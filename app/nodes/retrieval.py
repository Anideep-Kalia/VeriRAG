"""Hybrid retrieval node — BM25 + vector, deduplicated (verbatim logic)."""
from app.ingest import get_index


def hybrid_retrieve_node(state):
    idx = get_index()
    queries = (
        [state["rewritten_query"]] +
        state["expanded_queries"] +
        [state["step_back_query"]]
    )

    all_docs = []
    for q in queries:
        all_docs.extend(idx.bm25_retriever.invoke(q))
        all_docs.extend(idx.vector_retriever.invoke(q))

    # Deduplicate by content
    unique_docs = list({doc.page_content: doc for doc in all_docs}.values())

    return {"documents": unique_docs}
