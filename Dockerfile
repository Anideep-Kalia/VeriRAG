# Python 3.12 (3.14 lacks wheels for torch/chromadb/sentence-transformers).
FROM python:3.12-slim

WORKDIR /app

# build-essential covers any deps without prebuilt wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

# Embedding/reranker weights download on first run; mount storage/ to reuse the
# Chroma index built by `python -m app.ingest`.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
