"""Cross-encoder rerank + relevance filter + retry-decision (verbatim logic)."""
import numpy as np

from app.config import settings
from app.llm import get_cross_encoder


def rerank_and_filter_node(state):

    docs = state.get("documents", [])
    query = state["query"]

    if not docs:
        return {
            "documents": [],
            "answer": "I don't know",
            "retrieval_feedback": {"reason": "no_docs"}
        }

    pairs = [(query, doc.page_content) for doc in docs]
    scores = get_cross_encoder().predict(pairs)
    scores = 1 / (1 + np.exp(-scores))

    doc_scores = list(zip(docs, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)

    top_docs = doc_scores[:settings.rerank_top_k]
    top_scores = [score for _, score in top_docs]
    max_score = max(top_scores)
    avg_score = sum(top_scores) / len(top_scores)

    upper_threshold = settings.rerank_upper_threshold
    lower_threshold = settings.rerank_lower_threshold

    kept = [doc for doc, _ in top_docs]

    if max_score < lower_threshold:
        # Very low relevance — keep the best docs but ask the graph to refine the query.
        return {
            "documents": kept,
            "retrieval_feedback": {
                "reason": "very_low_relevance",
                "max_score": float(max_score),
                "avg_score": float(avg_score),
            },
        }

    # Medium/high relevance — pass through without burning retry budget. Only
    # very_low_relevance retries; medium docs often hold the right content but
    # score lower on the QA-trained cross-encoder.
    return {"documents": kept}


def Retry_decision_logic(state):

    iteration = state.get("iteration", 0)
    max_iterations = settings.max_iterations

    if state.get("retrieval_feedback") is not None and iteration < max_iterations:
        return "retry"

    # If no docs → retry
    if not state.get("documents") and iteration < max_iterations:
        return "retry"

    return "generate"


def Refine_query_node(state):
    return {
        "iteration": state.get("iteration", 0) + 1
    }
