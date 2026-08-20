from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from pdf_parser import parse_document
from chunking import chunk_table, chunk_text


_COLLECTION_NAME = "document_synthesizer"
_PERSIST_DIRECTORY = Path(__file__).resolve().parent / "chroma_db"
_EMBEDDING_MODEL = None
_BM25_INDEX = None
_ALL_DOCS_CACHE = None


def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Return a ChromaDB-safe metadata payload by dropping non-serializable values."""
    sanitized: Dict[str, Any] = {}
    for key, value in metadata.items():
        if key in {"source_doc", "page_number", "chunk_type", "parent_context"}:
            sanitized[key] = value
        elif isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[key] = value
    return sanitized


def _load_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("Install sentence-transformers to enable embeddings") from exc

        _EMBEDDING_MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _EMBEDDING_MODEL


def _get_collection() -> Tuple[Any, Any]:
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("Install chromadb to enable vector storage") from exc

    client = chromadb.PersistentClient(path=str(_PERSIST_DIRECTORY))
    try:
        collection = client.get_collection(name=_COLLECTION_NAME)
    except Exception:
        collection = client.create_collection(name=_COLLECTION_NAME)
    return client, collection


def _get_all_documents() -> List[Dict[str, Any]]:
    """Fetch all indexed documents from ChromaDB."""
    global _ALL_DOCS_CACHE
    if _ALL_DOCS_CACHE is not None:
        return _ALL_DOCS_CACHE

    _, collection = _get_collection()
    data = collection.get(include=["documents", "metadatas"])

    docs: List[Dict[str, Any]] = []
    ids = data.get("ids", [])
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    for doc_id, doc, meta in zip(ids, documents, metadatas):
        docs.append({"id": doc_id, "content": doc, "metadata": meta or {}})
    
    _ALL_DOCS_CACHE = docs
    return docs


def _load_bm25_index():
    """Build and cache the BM25 index on first load."""
    global _BM25_INDEX
    if _BM25_INDEX is None:
        all_docs = _get_all_documents()
        if all_docs:
            try:
                from rank_bm25 import BM25Okapi
                corpus_tokens = [d["content"].lower().split() for d in all_docs]
                _BM25_INDEX = BM25Okapi(corpus_tokens)
            except Exception:
                _BM25_INDEX = None
    return _BM25_INDEX


def clear_cache():
    """Clear BM25 cache when new docs are ingested."""
    global _BM25_INDEX, _ALL_DOCS_CACHE
    _BM25_INDEX = None
    _ALL_DOCS_CACHE = None


def ingest_document(pdf_path: str | Path) -> List[Dict[str, Any]]:
    """Parse, chunk, embed, and store a document in ChromaDB."""
    document = parse_document(pdf_path)
    chunks: List[Dict[str, Any]] = []

    for page in document.get("pages", []):
        if page.get("text"):
            chunks.extend(
                chunk_text(
                    page["text"],
                    source_doc=document["source_doc"],
                    page_number=page["page_number"],
                )
            )

    for table in document.get("tables", []):
        chunks.append(
            chunk_table(
                table["df"],
                source_doc=document["source_doc"],
                page_number=table["page"],
            )
        )

    if not chunks:
        return []

    texts = [chunk["content"] for chunk in chunks]
    metadatas = [_sanitize_metadata(chunk["metadata"]) for chunk in chunks]
    ids = [f"{document['source_doc']}:{index}" for index in range(len(chunks))]

    model = _load_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True).tolist()
    _, collection = _get_collection()
    collection.add(documents=texts, metadatas=metadatas, embeddings=embeddings, ids=ids)
    clear_cache()
    return chunks
