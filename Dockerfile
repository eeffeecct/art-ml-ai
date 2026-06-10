FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=5 \
    HF_HOME=/app/hf_cache

# CPU-only PyTorch first — the default PyPI wheel bundles CUDA and is several GB larger.
# (For a GPU server, swap this for the matching CUDA wheel and use a CUDA base image.)
RUN pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY worker.py .
COPY minimalism_classifier.pkl .

# The CLIP weights (~1.7 GB) are downloaded to HF_HOME on first run. docker-compose
# mounts a named volume there so they are downloaded once and reused across restarts.
CMD ["python", "worker.py"]
