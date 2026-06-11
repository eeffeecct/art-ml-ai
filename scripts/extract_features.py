import os
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# Configuration
EXCLUDE_DIRS = ['.git', '.idea', 'templates', 'uploads', '__pycache__']
# CLIP backbone. Keep IDENTICAL to worker.py's CLIP_MODEL — classifier, artworks vectors
# and the query must all live in the same embedding space. Upgrade lever: set
# CLIP_MODEL=openai/clip-vit-large-patch14-336 (sharper brushstroke detail) and re-extract
# everything. Projection dim stays 768, so the pgvector schema does not change.
MODEL_NAME = os.getenv('CLIP_MODEL', 'openai/clip-vit-large-patch14')
OUTPUT_FILE = "embeddings.npz"

def get_style_dirs():
    dataset_path = 'datasets'
    if not os.path.exists(dataset_path):
        return []
    dirs = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    return dirs

def extract_features():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    STYLE_DIRS = get_style_dirs()
    print(f"Found styles: {[os.path.basename(d) for d in STYLE_DIRS]}")

    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    all_embeddings = []
    all_labels = []
    all_paths = []

    for style_path in STYLE_DIRS:
        style_name = os.path.basename(style_path)
        print(f"Processing style: {style_name}")
        
        files = [f for f in os.listdir(style_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for filename in tqdm(files, desc=f"Style: {style_name}"):
            img_path = os.path.join(style_path, filename)
            try:
                image = Image.open(img_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs)
                
                if not isinstance(image_features, torch.Tensor):
                    if hasattr(image_features, "image_embeds"):
                        image_features = image_features.image_embeds
                    elif hasattr(image_features, "pooler_output"):
                        image_features = image_features.pooler_output
                    else:
                        image_features = torch.tensor(image_features)

                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                all_embeddings.append(image_features.cpu().numpy().flatten())
                all_labels.append(style_name)
                all_paths.append(img_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

    if not all_embeddings:
        print("No images found or processed successfully.")
        return

    np.savez_compressed(OUTPUT_FILE, 
                        embeddings=np.array(all_embeddings, dtype=np.float32), 
                        labels=np.array(all_labels), 
                        paths=np.array(all_paths))
    print(f"\nSuccess! Saved {len(all_embeddings)} embeddings to {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_features()
