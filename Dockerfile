# Hugging Face Spaces Docker SDK Image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HOST=0.0.0.0 \
    DATA_DIR=/app/data \
    INDEX_DIR=/app/data/indexes \
    PROCESSED_DIR=/app/data/processed \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-cache embedding model weights into Docker image at build time
RUN python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small'); AutoModel.from_pretrained('intfloat/multilingual-e5-small')"

# Copy application source code and pre-built index/processed artifacts
COPY config.py .
COPY app.py .
COPY data/ ./data/
COPY chunking/ ./chunking/
COPY retrieval/ ./retrieval/
COPY stt/ ./stt/
COPY guardrails/ ./guardrails/
COPY generation/ ./generation/
COPY pipeline/ ./pipeline/
COPY api/ ./api/
COPY benchmark/ ./benchmark/
COPY demo/ ./demo/

# Create user for Hugging Face Spaces security
RUN useradd -m -u 1000 user && \
    chown -R user:user /app
USER user

# Expose default Hugging Face Spaces port
EXPOSE 7860

# Launch FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
