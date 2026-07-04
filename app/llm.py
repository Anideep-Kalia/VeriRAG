"""
This builds and exposes the shared model clients (the core), 
wraps every call in safe_llm_invoke for resilience, and 
tosses in three small parsers for the JSON those models return.
"""
import json
import random
import time
from functools import lru_cache

from pydantic import BaseModel

from app.config import settings
from app.providers import build_chat_model


# Structured-output schema for query rewriting/expansion (verbatim from notebook).
class QueryOutput(BaseModel):
    rewritten_query: str
    expanded_queries: list[str]
    step_back_query: str


# Provider-agnostic clients (provider chosen via settings.llm_provider).
# Primary path: grounded generation, faithfulness judging.
llm = build_chat_model(settings.llm_model)

# Fast path: query rewriting / expansion.
llm_flash = build_chat_model(settings.llm_flash_model, provider=settings.llm_flash_provider or None)
llm_compression = llm_flash

structured_llm_flash = llm_flash.with_structured_output(QueryOutput)


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"trust_remote_code": True},
    )


@lru_cache(maxsize=1)
def get_cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.reranker_model)


def safe_llm_invoke(llm, prompt, retries=5):
    """Retry an LLM .invoke with exponential backoff + jitter (verbatim)."""
    last_error = None

    for i in range(retries):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            last_error = e
            wait_time = (2 ** i) + random.uniform(0, 1)
            time.sleep(wait_time)

    raise last_error


# --- LLM-output parsers (shared by generation / faithfulness nodes) ---

def extract_text(response):
    content = response.content

    if isinstance(content, list):
        return " ".join([item.get("text", "") for item in content]).strip()

    return content.strip()


def parse_json_response(response, fallback):
    raw = extract_text(response)

    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")

        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except Exception:
                pass

    return fallback


def normalize_list(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    if isinstance(value, str) and value.strip():
        return [value.strip()]

    return []
