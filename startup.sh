#!/bin/bash
# Azure App Service startup script
# Uses gunicorn as the WSGI server with extended timeout for ML model loading
# Models (HuBERT ~400MB, Whisper ~150MB, SentenceTransformer ~90MB) take 30-60s on cold start

mkdir -p /home/LogFiles
exec > /home/LogFiles/gunicorn_master.log 2>&1
echo "Starting Application at $(date)"

gunicorn \
    --bind=0.0.0.0:8000 \
    --timeout=900 \
    --workers=1 \
    --threads=2 \
    --worker-class=gthread \
    --access-logfile=- \
    --error-logfile=- \
    api:app
