"""Central settings — loaded from .env, with notebook-identical defaults.

Every value here is extracted verbatim from the notebook so the refactored
service has parity. Real secrets live only in .env (git-ignored).
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- provider switch (gemini | openai | groq | ollama) ---
    llm_provider: str = "gemini"

    # --- API keys (blank defaults; real values only in .env) ---
    google_api_key: str = ""
    openai_api_key: str = ""
    groq_api_key: str = ""
    anthropic_api_key: str = ""        # placeholder; not wired as a provider yet
    ollama_base_url: str = "http://localhost:11434"

    # --- chat model names (interpreted within the chosen provider) ---
    # Defaults are Gemini for Week-1 parity. When switching provider, set these to
    # that provider's ids, e.g. openai -> gpt-4o / gpt-4o-mini, groq ->
    # llama-3.3-70b-versatile / llama-3.1-8b-instant, ollama -> llama3.1 / llama3.2.
    llm_model: str = "gemini-2.5-flash"        # primary: grounded generation + faithfulness judge
    llm_flash_model: str = "gemini-2.5-flash"  # fast: query rewrite / expansion
    llm_temperature: float = 0.3

    # --- local models (provider-independent; not part of the LLM switch) ---
    embedding_model: str = "nomic-ai/nomic-embed-text-v1"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- evaluation (Week 2): use a CHEAP judge; a full run is hundreds of calls ---
    eval_judge_model: str = "gemini-2.5-flash"   # RAGAS/DeepEval judge (built via provider factory)
    eval_fail_threshold: float = 0.6             # per-metric cutoff for failed_cases.csv

    # --- chunking / storage ---
    chunk_size: int = 1000
    chunk_overlap: int = 150
    chroma_path: str = "storage/chroma"
    chroma_collection: str = "verirag-pdf-chroma"
    pdf_dir: str = "documents/pdfs"
    chars_per_page: int = 3000          # page-number estimate from char offset

    # --- retrieval / rerank ---
    retrieval_top_k: int = 5
    rerank_top_k: int = 5
    rerank_upper_threshold: float = 0.7
    rerank_lower_threshold: float = 0.4

    # --- self-correction loop ---
    max_iterations: int = 2
    faithfulness_pass_score: float = 0.85
    max_faithfulness_retries: int = 2

    # --- tracing (opt-in; off by default so a missing key never adds latency) ---
    langchain_tracing: bool = False
    langchain_api_key: str = ""


settings = Settings()
