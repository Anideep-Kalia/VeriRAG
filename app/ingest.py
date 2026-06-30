"""PDF ingestion (build) + index loading (serve).

CLI:   python -m app.ingest [pdf_dir]   extract -> chunk -> embed -> persist Chroma
Serve: get_index() / load_index()       open persisted Chroma, rebuild BM25

The store lives in one canonical dir; re-running ingest rebuilds it in place. BM25
isn't persisted, so it's rebuilt at load time from the chunks stored in Chroma
(single source of truth -> retrieval parity).
"""
import shutil
from dataclasses import dataclass
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import settings
from app.llm import get_embeddings

PDF_FOLDER = Path(settings.pdf_dir)
INDEX_DIR = Path(settings.chroma_path) / "index"


@dataclass
class Index:
    """Retrieval singletons a graph run needs (built once, shared)."""
    vector_retriever: object
    bm25_retriever: object


_INDEX: Index | None = None


def _source_name(pdf_path: Path) -> str:
    """PDF path relative to the corpus folder (filename if it sits outside it)."""
    try:
        return str(pdf_path.resolve().relative_to(PDF_FOLDER.resolve()))
    except ValueError:
        return str(pdf_path)


def _pdf_to_documents(pdf_files: list[Path]) -> list[Document]:
    """One Document per PDF, text extracted with PyPDF."""
    docs = []
    for pdf_path in pdf_files:
        try:
            text = "\n\n".join((p.extract_text() or "") for p in PdfReader(str(pdf_path)).pages).strip()
        except Exception as e:
            print(f"[pdf] {pdf_path.name} failed: {e}")
            continue
        if not text:
            print(f"Warning: {pdf_path.name} produced no text — skipping.")
            continue
        docs.append(Document(page_content=text, metadata={"source": _source_name(pdf_path)}))
    return docs


def load_pdf_chunks(folder: Path):
    pdf_files = sorted(p for p in folder.rglob("*.pdf") if p.is_file())
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs in {folder.resolve()}. Add PDFs and rerun.")

    print(f"Extracting {len(pdf_files)} PDF(s)...")
    raw_documents = _pdf_to_documents(pdf_files)
    if not raw_documents:
        raise ValueError(f"PDFs found in {folder.resolve()} but no text could be extracted.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    chunks = filter_complex_metadata(splitter.split_documents(raw_documents))
    for i, doc in enumerate(chunks):
        doc.metadata["chunk_index"] = i
        doc.metadata["page"] = doc.metadata.get("start_index", 0) // settings.chars_per_page
    return chunks


def build_vectorstore(chunks) -> Chroma:
    """Rebuild the persisted Chroma store in place."""
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    return Chroma.from_documents(
        documents=chunks,
        collection_name=settings.chroma_collection,
        embedding=get_embeddings(),
        persist_directory=str(INDEX_DIR),
    )


def _chunks_from_chroma(vectorstore) -> list[Document]:
    """Reconstruct chunks from the persisted store to rebuild BM25 (not persisted)."""
    data = vectorstore.get(include=["documents", "metadatas"])
    return [
        Document(page_content=t, metadata=m or {})
        for t, m in zip(data.get("documents") or [], data.get("metadatas") or [])
    ]


def _make_index(vectorstore, chunks) -> Index:
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = settings.retrieval_top_k
    return Index(
        vector_retriever=vectorstore.as_retriever(search_kwargs={"k": settings.retrieval_top_k}),
        bm25_retriever=bm25,
    )


def build_index(pdf_dir=None) -> Index:
    """Full ingest: PDFs -> chunk -> embed -> persist Chroma."""
    global _INDEX
    chunks = load_pdf_chunks(Path(pdf_dir) if pdf_dir else PDF_FOLDER)
    vectorstore = build_vectorstore(chunks)
    print(f"Indexed {len(chunks)} chunks.")
    _INDEX = _make_index(vectorstore, chunks)
    return _INDEX


def load_index() -> Index:
    """Open the persisted store and rebuild BM25 (no PDF re-read)."""
    if not (INDEX_DIR / "chroma.sqlite3").exists():
        raise RuntimeError(f"No Chroma index at {INDEX_DIR.resolve()}. Run: python -m app.ingest")

    vectorstore = Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
        persist_directory=str(INDEX_DIR),
    )
    chunks = _chunks_from_chroma(vectorstore)
    if not chunks:
        raise RuntimeError(f"Chroma index at {INDEX_DIR} is empty. Run: python -m app.ingest")
    return _make_index(vectorstore, chunks)


def get_index() -> Index:
    """Cached accessor used by nodes + the service (built once per process)."""
    global _INDEX
    if _INDEX is None:
        _INDEX = load_index()
    return _INDEX


def main(argv=None):
    import sys

    args = argv if argv is not None else sys.argv[1:]
    build_index(args[0] if args else None)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
