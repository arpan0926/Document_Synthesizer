from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from pdf_parser import parse_document
from chunking import chunk_table, chunk_text


_COLLECTION_NAME = "document_synthesizer"
_PERSIST_DIRECTORY = Path(__file__).resolve().parent / "chroma_db"
_EMBEDDING_MODEL = None


def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Return a ChromaDB-safe metadata payload by dropping non-serializable values."""
    sanitized: Dict[str, Any] = {}
    for key, value in metadata.items():
        if key in {"source_doc", "page_number", "chunk_type"}:
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

        _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
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
    return chunks
