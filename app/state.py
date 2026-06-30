"""GraphState — the typed state every node reads/writes."""
from typing import List, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    query: str
    rewritten_query: str
    expanded_queries: List[str]
    step_back_query: str
    documents: List[Document]
    answer: str
    claims: List[str]
    iteration: int
    retrieval_feedback: dict
    faithfulness: dict
    faithfulness_score: float
    unsupported_claims: List[str]
    faithfulness_retry_count: int
    abstention_decision: str
    references: List[dict]             # [{document, page}] for grounded answers
