"""FastAPI request/response models for the /query endpoint."""
from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]                 # e.g. ["data1.pdf — p.3 (~line 12)"]
    faithfulness_score: float | None     # grounded float; None when not doc-judged
    abstained: bool
    iterations: int                      # retrieval self-correction loop count
