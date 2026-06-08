# MINIMALISM.AI - ML Worker Service

This repository serves as the specialized Machine Learning Worker for the MINIMALISM.AI microservices ecosystem. It is responsible for processing art analysis tasks asynchronously via RabbitMQ.

## Project Overview

- **Role:** ML-Worker Service (Background process).
- **Primary Technologies:**
  - **Machine Learning:** PyTorch, Transformers (CLIP: `clip-vit-large-patch14`), scikit-learn (Logistic Regression, K-Means).
  - **Messaging:** RabbitMQ (via `pika`).
  - **Data Processing:** NumPy, Pillow (PIL), Requests.
- **Workflow:** 
  1. Listen to `art.analysis.queue` for tasks.
  2. Download image from provided S3 URL.
  3. Extract CLIP embeddings (768-dim vector).
  4. Perform multi-class style classification.
  5. Extract dominant color palette.
  6. Publish results to `art.results.queue`.

## Directory Structure

- `worker.py`: The main background worker that communicates with RabbitMQ.
- `rebuild.py`: Automation script to re-extract features and retrain the classifier locally.
- `extract_features.py`: Computes CLIP embeddings for local style folders.
- `train_classifier.py`: Trains the Logistic Regression model.
- `main.py`: Script to download the WikiArt dataset (for local development/training).
- `TechSpecs.txt`: Detailed technical specifications for the entire microservices architecture.

## Running the Worker

### 1. Prerequisites
Ensure you have a RabbitMQ instance running. You can set the host via the `RABBITMQ_HOST` environment variable (defaults to `localhost`).

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Prepare the Model
Ensure `minimalism_classifier.pkl` is present. If not, you need to train it:
```bash
python rebuild.py
```

### 3. Start the Worker
```bash
python worker.py
```

## Microservices Integration

This worker interacts with a **Java (Spring Boot) API Gateway**:
- **Task Input (JSON):** `{"taskId": "UUID", "imageUrl": "S3_URL", "mode": "auto"}`
- **Result Output (JSON):** `{"taskId": "UUID", "embedding": [768 floats], "palette": ["#HEX1", ...], "styleBreakdown": [{"style": "Name", "prob": "85.4%"}, ...]}`

The Java side is responsible for MinIO (S3-compatible) storage, PostgreSQL management (including `pgvector` similarity search), and the user-facing REST API.
