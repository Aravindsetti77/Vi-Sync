FROM python:3.10-slim

WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Set cache path inside the app directory so it's accessible regardless of the user running the container
ENV FASTEMBED_CACHE_PATH=/app/model_cache
# Pre-download the fastembed model into the Docker image layer.
# This means container restarts / rebuilds (when only code changes) won't
# re-download ~80MB of model weights from HuggingFace.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')"

COPY . .

EXPOSE 8000

# Set PYTHONPATH so that uvicorn can find the backend module
ENV PYTHONPATH=/app:/app/backend

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
