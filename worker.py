import pika
import pika.exceptions
import json
import torch
import numpy as np
import joblib
import io
import sys
import time
import requests
from dotenv import load_dotenv
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
import traceback
import os

load_dotenv()

# Configuration
RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', 'localhost')
RABBITMQ_USER = os.getenv('RABBITMQ_DEFAULT_USER', 'guest')
RABBITMQ_PASS = os.getenv('RABBITMQ_DEFAULT_PASS', 'guest')
EXCHANGE = 'art.exchange'
ROUTING_KEY_TASKS = 'art.analyze'
ROUTING_KEY_RESULTS = 'art.result'
QUEUE_TASKS = 'art.analysis.queue'
QUEUE_RESULTS = 'art.results.queue'
MODEL_FILE = "minimalism_classifier.pkl"
# Must match extract_features.py's CLIP_MODEL and the model that built the artworks vectors.
# Upgrade: CLIP_MODEL=openai/clip-vit-large-patch14-336 (then re-extract + reload the DB).
CLIP_MODEL_NAME = os.getenv('CLIP_MODEL', 'openai/clip-vit-large-patch14')

# Blend weight for style classification: trained head vs zero-shot CLIP text prior.
# Default 1.0 = pure trained head. The MLP head is strong (~75% top-1), so the spiky
# zero-shot prior tends to hurt in-distribution and is left OFF by default. Try
# CLF_WEIGHT=0.9 if the zero-shot prior helps on real / out-of-distribution uploads.
CLF_WEIGHT = float(os.getenv('CLF_WEIGHT', '1.0'))

STYLE_TRANSLATIONS = {
    "Abstract_Expressionism": "Абстрактный экспрессионизм",
    "Action_painting": "Живопись действия",
    "Analytical_Cubism": "Аналитический кубизм",
    "Art_Nouveau_Modern": "Ар-нуво (Модерн)",
    "Baroque": "Барокко",
    "Color_Field_Painting": "Живопись цветового поля",
    "Contemporary_Realism": "Современный реализм",
    "Cubism": "Кубизм",
    "Early_Renaissance": "Раннее Возрождение",
    "Expressionism": "Экспрессионизм",
    "Fauvism": "Фовизм",
    "High_Renaissance": "Высокое Возрождение",
    "Impressionism": "Импрессионизм",
    "Mannerism_Late_Renaissance": "Маньеризм",
    "Minimalism": "Минимализм",
    "Naive_Art_Primitivism": "Наивное искусство",
    "New_Realism": "Новый реализм",
    "Northern_Renaissance": "Северное Возрождение",
    "Pointillism": "Пуантилизм",
    "Pop_Art": "Поп-арт",
    "Post_Impressionism": "Постимпрессионизм",
    "Realism": "Реализм",
    "Rococo": "Рококо",
    "Romanticism": "Романтизм",
    "Symbolism": "Символизм",
    "Synthetic_Cubism": "Синтетический кубизм",
    "Ukiyo_e": "Укиё-э"
}

print("Starting ML Worker...")

# Load Models
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading CLIP...")
model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

print("Loading Classifier...")
if os.path.exists(MODEL_FILE):
    model_data = joblib.load(MODEL_FILE)
    clf = model_data['classifier']
    style_classes = model_data['classes']

    # Compatibility patch for older pickles
    if not hasattr(clf, 'multi_class'):
        clf.multi_class = 'multinomial'
else:
    # Style classification is a core feature — fail fast instead of silently
    # publishing "successful" results with an empty style breakdown.
    print(f"CRITICAL: {MODEL_FILE} not found! Cannot start without the classifier.")
    sys.exit(1)


# Precompute one zero-shot text embedding per style from CLIP's text tower. Blended with
# the trained head at inference (see CLF_WEIGHT) to steady predictions on rare styles.
ZEROSHOT_TEMPLATES = [
    "a painting in the style of {s}",
    "an example of {s} art",
    "a {s} style artwork",
]
text_embeds = None
logit_scale = 1.0
if CLF_WEIGHT < 1.0:
    print("Precomputing zero-shot text embeddings...")
    _rows = []
    for _raw in style_classes:
        _human = str(_raw).replace('_', ' ').lower()
        _prompts = [t.format(s=_human) for t in ZEROSHOT_TEMPLATES]
        _tin = processor(text=_prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            _tfeat = model.get_text_features(**_tin)
        _tfeat = _tfeat / _tfeat.norm(p=2, dim=-1, keepdim=True)
        _rows.append(_tfeat.mean(dim=0))
    text_embeds = torch.stack(_rows)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    logit_scale = model.logit_scale.exp().item()


def get_colors(image, num_colors=5):
    try:
        img = image.copy()
        img.thumbnail((150, 150))
        ar = np.asarray(img)
        if len(ar.shape) != 3 or ar.shape[2] != 3:
            return ['#000000'] * num_colors
        shape = ar.shape
        ar = ar.reshape(np.prod(shape[:2]), shape[2])
        kmeans = KMeans(n_clusters=num_colors, n_init='auto', random_state=42).fit(ar)
        colors = kmeans.cluster_centers_.astype(int)
        hex_colors = ['#{:02x}{:02x}{:02x}'.format(np.clip(c[0], 0, 255), np.clip(c[1], 0, 255), np.clip(c[2], 0, 255)) for c in colors]
        return hex_colors
    except Exception as e:
        print(f"Palette extraction failed: {e}")
        return ['#000000'] * num_colors


# Warm up the inference pipeline at startup. PyTorch lazily initializes its CPU kernels
# and thread pool on the FIRST forward pass, which for CLIP ViT-Large is ~5-10s. Running
# one dummy image through the exact same path here pays that cost at boot (where nobody is
# waiting) instead of on the user's first real upload. Also warms the sklearn head + KMeans.
def warmup():
    print("Warming up CLIP (first CPU forward pass is slow)...")
    try:
        dummy = Image.new("RGB", (224, 224), (127, 127, 127))
        inputs = processor(images=dummy, return_tensors="pt").to(device)
        with torch.no_grad():
            feat = model.get_image_features(**inputs)
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        clf.predict_proba(feat.cpu().numpy().reshape(1, -1))
        get_colors(dummy)
        print("Warm-up complete.")
    except Exception as e:
        print(f"Warm-up skipped: {e}")


def publish_failure(ch, task_id, error_message):
    """Tell Java the task failed so it is marked FAILED instead of stuck PROCESSING."""
    failure = {"taskId": task_id, "status": "FAILED", "error": str(error_message)[:500]}
    ch.basic_publish(
        exchange=EXCHANGE,
        routing_key=ROUTING_KEY_RESULTS,
        body=json.dumps(failure),
        properties=pika.BasicProperties(delivery_mode=2)
    )


def process_task(ch, method, properties, body):
    task_id = None
    try:
        data = json.loads(body)
        task_id = data.get('taskId')
        image_url = data.get('imageUrl')

        print(f"Processing Task: {task_id} | URL: {image_url}")

        # 1. Download Image
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(image_url, headers=headers, timeout=30)
        response.raise_for_status()

        image = Image.open(io.BytesIO(response.content)).convert("RGB")

        # 2. Extract Embedding
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

        if not isinstance(image_features, torch.Tensor):
            if hasattr(image_features, "image_embeds"):
                image_features = image_features.image_embeds
            elif hasattr(image_features, "pooler_output"):
                image_features = image_features.pooler_output
            else:
                image_features = image_features[0]

        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        embedding = image_features.cpu().numpy().flatten().tolist()

        # 3. Classify Style (trained head, optionally blended with the zero-shot prior)
        query_emb = np.array(embedding).reshape(1, -1)
        probs = clf.predict_proba(query_emb)[0]

        if text_embeds is not None:
            sims = (image_features @ text_embeds.T).squeeze(0)
            zs_probs = torch.softmax(sims * logit_scale, dim=-1).cpu().numpy()
            probs = CLF_WEIGHT * probs + (1.0 - CLF_WEIGHT) * zs_probs
            probs = probs / probs.sum()

        style_probs = sorted(zip(style_classes, probs), key=lambda x: x[1], reverse=True)

        style_breakdown = [
            {
                "style": STYLE_TRANSLATIONS.get(s, s),
                "prob": f"{p:.1%}",
                "val": float(p * 100)
            } for s, p in style_probs[:5]
        ]

        # 4. Extract Palette
        palette = get_colors(image)

        # 5. Prepare Result
        result = {
            "taskId": task_id,
            "embedding": embedding,
            "palette": palette,
            "styleBreakdown": style_breakdown
        }

        # 6. Publish Result
        ch.basic_publish(
            exchange=EXCHANGE,
            routing_key=ROUTING_KEY_RESULTS,
            body=json.dumps(result),
            properties=pika.BasicProperties(delivery_mode=2)
        )

        print(f"Completed Task: {task_id}")
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        print(f"Error processing task {task_id}: {e}")
        traceback.print_exc()
        # Report failure to Java, then ack (the message is "handled"). If we cannot even
        # publish the failure, nack without requeue so it is not retried endlessly.
        try:
            if task_id:
                publish_failure(ch, task_id, e)
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as pub_err:
            print(f"Could not report failure for task {task_id}: {pub_err}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def main():
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    parameters = pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        credentials=credentials,
        heartbeat=600,                   # tolerate long CLIP inference between heartbeats
        blocked_connection_timeout=300,
    )

    while True:
        connection = None
        try:
            print("Connecting to RabbitMQ...")
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()

            # Ensure Exchange and Queues exist (idempotent; re-run on every reconnect)
            channel.exchange_declare(exchange=EXCHANGE, exchange_type='direct', durable=True)
            channel.queue_declare(queue=QUEUE_TASKS, durable=True)
            channel.queue_declare(queue=QUEUE_RESULTS, durable=True)
            channel.queue_bind(queue=QUEUE_TASKS, exchange=EXCHANGE, routing_key=ROUTING_KEY_TASKS)
            channel.queue_bind(queue=QUEUE_RESULTS, exchange=EXCHANGE, routing_key=ROUTING_KEY_RESULTS)

            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue=QUEUE_TASKS, on_message_callback=process_task)

            print("Worker is waiting for messages. To exit press CTRL+C")
            channel.start_consuming()

        except pika.exceptions.AMQPConnectionError as e:
            print(f"RabbitMQ connection error: {e}. Reconnecting in 5s...")
            time.sleep(5)
        except KeyboardInterrupt:
            print("Shutting down worker...")
            try:
                if connection and connection.is_open:
                    connection.close()
            except Exception:
                pass
            break
        except Exception as e:
            print(f"Unexpected worker error: {e}. Reconnecting in 5s...")
            traceback.print_exc()
            time.sleep(5)


if __name__ == "__main__":
    warmup()
    main()
