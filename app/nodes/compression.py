"""Context compression node — query-relevant sentence selection (verbatim logic)."""
import numpy as np
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity

from app.llm import get_embeddings


def compress_documents_node(state):

    query = state["query"]
    docs = state.get("documents", [])[:5]

    if not docs:
        return {"documents": []}

    embeddings = get_embeddings()

    # Embed query once
    query_embedding = embeddings.embed_query(query)

    compressed_docs = []

    for doc in docs:

        text = doc.page_content

        # Step 1: sentence splitting (simple but effective)
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 10]

        if not sentences:
            compressed_docs.append(doc)
            continue

        # Step 2: embed sentences
        sentence_embeddings = embeddings.embed_documents(sentences)

        # Step 3: similarity scoring
        scores = cosine_similarity(
            [query_embedding],
            sentence_embeddings
        )[0]

        # Step 4: pick top-k sentences
        top_k = min(6, len(sentences))

        top_indices = np.argsort(scores)[-top_k:]
        top_indices = sorted(top_indices)  # preserve order

        selected_sentences = [sentences[i] for i in top_indices]

        compressed_text = ". ".join(selected_sentences)

        # Preserve original metadata so page/source/start_index survive into the planner
        compressed_docs.append(Document(page_content=compressed_text, metadata=doc.metadata))

    return {"documents": compressed_docs}
