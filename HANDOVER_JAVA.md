# HANDOVER: ML-Worker Integration for Java/Spring Boot

This document contains the technical details required to integrate the Python ML-Worker with the Java API Gateway.

## 1. System Role
The Python service acts as an **asynchronous worker**. It does not have an HTTP API. It communicates exclusively via **RabbitMQ**.

## 2. RabbitMQ Configuration
- **Host:** Defined by `RABBITMQ_HOST` environment variable (default: `localhost`).
- **Input Queue (Java -> Python):** `art.analysis.queue` (Durable)
- **Output Queue (Python -> Java):** `art.results.queue` (Durable)

## 3. Data Contracts (JSON)

### Task Message (Sent by Java)
Java should publish a message to `art.analysis.queue`:
```json
{
  "taskId": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "imageUrl": "https://s3.amazonaws.com/your-bucket/uploads/image.jpg"
}
```

### Result Message (Sent by Python)
Python will publish a message to `art.results.queue`:
```json
{
  "taskId": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "embedding": [0.012, -0.045, ...], 
  "palette": ["#030303", "#f5f5f5", "#555555", "#aaaaaa", "#bbbbbb"],
  "styleBreakdown": [
    {"style": "Minimalism", "prob": "85.4%"},
    {"style": "Baroque", "prob": "12.1%"},
    {"style": "Impressionism", "prob": "2.5%"}
  ]
}
```
*Note: `embedding` is a float array of size 768 (CLIP-large).*

## 4. Requirements for Java Service (API Gateway)
1. **MinIO (S3-compatible) Integration:** Java must upload the user's file to MinIO and provide a public or pre-signed URL to Python.
2. **PostgreSQL + pgvector:** 
   - Store the `embedding` in a `vector(768)` column.
   - Use HNSW index for similarity search.
   - Sample SQL for finding top 6 similar images:
     ```sql
     SELECT id, artist, title, style, image_url 
     FROM artworks 
     ORDER BY embedding <=> cast(:query_embedding as vector) 
     ASC LIMIT 6;
     ```
3. **Task Management:** Java should track `taskId` status (PROCESSING -> COMPLETED/FAILED).

## 5. Deployment Notes
- Python worker uses `CLIP Model (openai/clip-vit-large-patch14)`.
- Pre-trained classifier is in `minimalism_classifier.pkl`.
- To start the worker: `python worker.py`.

## 6. Current Python Files
- `worker.py`: The RabbitMQ consumer/producer.
- `rebuild.py`: Full model training pipeline.
- `extract_features.py`: Feature extraction logic.
- `train_classifier.py`: Training logic.
- `requirements.txt`: Python dependencies.
