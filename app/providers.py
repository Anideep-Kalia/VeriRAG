"""Provider-agnostic chat-model factory.

Switch the active model with LLM_PROVIDER in .env (gemini | openai | groq | ollama).
Each builder lazily imports its LangChain integration, so only the selected
provider's package needs to be installed. Returns standard LangChain chat models,
so every node (ChatPromptTemplate / .invoke / with_structured_output) is unchanged.

ponytail: VeriRAG deliberately does NOT replicate the LLM Gateway's
routing/failover/cost layer — that project owns it. This is a single-active-model
switch, nothing more.
"""
from app.config import settings


def _gemini(model, temperature):
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.google_api_key,
        temperature=temperature,
    )


def _openai(model, temperature):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, api_key=settings.openai_api_key, temperature=temperature)


def _groq(model, temperature):
    from langchain_groq import ChatGroq

    return ChatGroq(model=model, api_key=settings.groq_api_key, temperature=temperature)


def _ollama(model, temperature):
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model, base_url=settings.ollama_base_url, temperature=temperature)


# Provider name -> builder. "google" is an alias for "gemini".
_BUILDERS = {
    "gemini": _gemini,
    "google": _gemini,
    "openai": _openai,
    "groq": _groq,
    "ollama": _ollama,
}


def build_chat_model(model: str, temperature: float | None = None):
    """Build a LangChain chat model for the provider named in settings.llm_provider."""
    provider = (settings.llm_provider or "gemini").lower()
    if provider not in _BUILDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER={provider!r}; choose one of {sorted(set(_BUILDERS))}"
        )
    temp = settings.llm_temperature if temperature is None else temperature
    return _BUILDERS[provider](model, temp)
