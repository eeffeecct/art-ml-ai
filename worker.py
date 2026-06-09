import pika
import json
import torch
import numpy as np
import joblib
import io
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
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"

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
    
    # Compatibility patch
    if not hasattr(clf, 'multi_class'):
        clf.multi_class = 'multinomial'
else:
    print(f"CRITICAL: {MODEL_FILE} not found! Worker will fail on classification.")
    clf = None

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
        hex_colors = ['#{:02x}{:02x}{:02x}'.format(np.clip(c[0],0,255), np.clip(c[1],0,255), np.clip(c[2],0,255)) for c in colors]
        return hex_colors
    except:
        return ['#000000'] * num_colors

def process_task(ch, method, properties, body):
    try:
        data = json.loads(body)
        task_id = data.get('taskId')
        image_url = data.get('imageUrl')
        
        print(f"Processing Task: {task_id} | URL: {image_url}")
        
        # 1. Download Image
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(image_url, headers=headers, timeout=10)
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
        
        # 3. Classify Style
        style_breakdown = []
        if clf:
            query_emb = np.array(embedding).reshape(1, -1)
            probs = clf.predict_proba(query_emb)[0]
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
        print(f"Error processing task: {e}")
        traceback.print_exc()
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

# RabbitMQ Connection
print("Connecting to RabbitMQ...")
credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
parameters = pika.ConnectionParameters(
    host=RABBITMQ_HOST,
    credentials=credentials
)
connection = pika.BlockingConnection(parameters)
channel = connection.channel()

# Ensure Exchange and Queues exist
channel.exchange_declare(exchange=EXCHANGE, exchange_type='direct', durable=True)
channel.queue_declare(queue=QUEUE_TASKS, durable=True)
channel.queue_declare(queue=QUEUE_RESULTS, durable=True)

# Bind Queues
channel.queue_bind(queue=QUEUE_TASKS, exchange=EXCHANGE, routing_key=ROUTING_KEY_TASKS)
channel.queue_bind(queue=QUEUE_RESULTS, exchange=EXCHANGE, routing_key=ROUTING_KEY_RESULTS)

channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=QUEUE_TASKS, on_message_callback=process_task)

print("Worker is waiting for messages. To exit press CTRL+C")
channel.start_consuming()
