#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np
import random
import os
import psutil

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder

# ==============================
# 🔧 CONFIG (TUNE THIS)
# ==============================
DATA_SCALE = 500        # increase → more load
PAIR_MULTIPLIER = 30    # cross-encoder stress
QUERY_REPEAT = 10       # retrieval stress
RUNS = 5                # benchmark runs

# ==============================
# 📦 DATASET
# ==============================
base_docs = [
    "RAG pipelines combine retrieval and generation for improved accuracy.",
    "Retrieval latency can impact real-time system performance.",
    "Semantic search uses embeddings to match meaning rather than keywords.",
    "Vector databases store embeddings for efficient similarity search.",
    "BM25 is a ranking function used in information retrieval.",
    "Cross-attention helps models focus on relevant context.",
    "Prompt engineering can improve RAG output quality.",
    "Token limits restrict how much context can be passed to LLMs.",
    "Query decomposition splits complex queries into sub-queries.",
    "Answer grounding reduces hallucination risk.",
]

documents = [
    Document(page_content=f"{i} {doc}")
    for i in range(DATA_SCALE)
    for doc in base_docs
]

# ==============================
# 🔌 INIT MODELS (LOCAL ONLY)
# ==============================
print("Loading models...")

embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"trust_remote_code": True}
)

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="benchmark"
)

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5

print("Models loaded.\n")

# ==============================
# 🔥 BENCHMARK FUNCTION
# ==============================
def run_pipeline(query):

    # 1. Retrieval (stress)
    queries = [query] * QUERY_REPEAT

    all_docs = []
    for q in queries:
        all_docs.extend(bm25_retriever.invoke(q))
        all_docs.extend(vector_retriever.invoke(q))

    # Deduplicate
    unique_docs = list({doc.page_content: doc for doc in all_docs}.values())

    # 2. Cross Encoder Stress (MAIN CPU TEST)
    pairs = [(query, doc.page_content) for doc in unique_docs]

    if not pairs:
        return

    pairs = pairs * PAIR_MULTIPLIER

    scores = cross_encoder.predict(pairs)

    # 3. Embedding Stress
    texts = [doc.page_content for doc in unique_docs]
    _ = embeddings.embed_documents(texts)


# ==============================
# 📊 SYSTEM STATS
# ==============================
def print_system_stats():
    process = psutil.Process(os.getpid())
    print(f"CPU Usage: {psutil.cpu_percent(interval=1)}%")
    print(f"Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")


# ==============================
# ⏱️ BENCHMARK RUNNER
# ==============================
def benchmark():

    query = "How does RAG improve accuracy?"

    # Warmup (IMPORTANT)
    print("Warmup run...")
    run_pipeline(query)

    times = []

    print("\nRunning benchmark...\n")

    for i in range(RUNS):
        start = time.perf_counter()

        run_pipeline(query)

        end = time.perf_counter()

        duration = end - start
        times.append(duration)

        print(f"Run {i+1}: {duration:.4f} sec")

    print("\n📊 RESULTS")
    print(f"Average: {sum(times)/len(times):.4f} sec")
    print(f"Min: {min(times):.4f} sec")
    print(f"Max: {max(times):.4f} sec")

    print("\n🖥️ System Stats:")
    print_system_stats()


# ==============================
# 🚀 START
# ==============================
if __name__ == "__main__":
    benchmark()