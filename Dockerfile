FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY eval ./eval
COPY scripts ./scripts

# Persisted index + docstore live here; mount a volume in production.
ENV DATA_DIR=/data
RUN mkdir -p /data

EXPOSE 8000

# A long-lived process keeps the FAISS index warm in memory.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
