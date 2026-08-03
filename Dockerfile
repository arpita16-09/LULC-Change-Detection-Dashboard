# Hugging Face Spaces / generic Docker image for the LULC Transition API
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/* \
 && git lfs install

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY . /app

# Rebuild enhanced features from OSCD imagery (npy files are gitignored)
RUN python landcover_train/build_oscd_enhanced_dataset.py

ENV CORS_ALLOW_ALL=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app/backend

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
