"""Central settings — loaded from .env, with notebook-identical defaults.

Every value here is extracted verbatim from the notebook so the refactored
service has parity. Real secrets live only in .env (git-ignored).
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # main: grounded generation + faithfulness node
    llm_provider: str = "gemini"        

    # fast: query rewrite / expansion
    llm_flash_provider: str = ""

    # RAGAS/DeepEval judge (built via provider factory)
    eval_judge_provider: str = ""


    # --- API keys (blank defaults; real values only in .env) ---
    google_api_key: str = ""
    openai_api_key: str = ""
    groq_api_key: str = ""
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"


    llm_model: str = "gemini-2.5-flash"
    llm_flash_model: str = "gemini-2.5-flash"  
    llm_temperature: float = 0.3


    embedding_model: str = "nomic-ai/nomic-embed-text-v1"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


    eval_judge_model: str = "gemini-2.5-flash"   
    eval_fail_threshold: float = 0.6


    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment: str = "verirag-eval"


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
