"""FastAPI serving layer: GET /health, POST /query.

The Chroma index + retrievers and the compiled graph are built ONCE at startup
(not per request).
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.graph import build_graph
from app.ingest import get_index
from app.schemas import QueryRequest, QueryResponse

_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    get_index()            # warm: load Chroma + rebuild BM25 once
    _graph = build_graph()  # compile graph once
    yield


app = FastAPI(title="VeriRAG", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        result = _graph.invoke({
            "query": req.question,
            "iteration": 0,
            "faithfulness_retry_count": 0,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"graph invocation failed: {e}")
    return _to_response(result)


def _to_response(result: dict) -> QueryResponse:
    answer = result.get("answer", "")
    if isinstance(answer, list):
        answer = " ".join(item.get("text", "") for item in answer).strip()

    abstained = result.get("abstention_decision") == "abstain"
    references = result.get("references", []) or []
    citations = [f'{r.get("document", "unknown")} — p.{r.get("page", "?")}' for r in references]

    return QueryResponse(
        answer=answer,
        citations=citations,
        faithfulness_score=None if abstained else result.get("faithfulness_score"),
        abstained=abstained,
        iterations=int(result.get("iteration", 0) or 0),
    )
