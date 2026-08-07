FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformer model into the Docker image layer.
# This means container restarts / rebuilds (when only code changes) won't
# re-download ~80MB of model weights from HuggingFace.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

EXPOSE 8000

# Set PYTHONPATH so that uvicorn can find the backend module
ENV PYTHONPATH=/app:/app/backend

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
