from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document

from vector_store import _get_collection, _load_embedding_model

_CROSS_ENCODER = None


def _load_cross_encoder():
    """Lazy-load the cross-encoder model for reranking."""
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("Install sentence-transformers to enable cross-encoder reranking") from exc

        _CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _CROSS_ENCODER


def to_langchain_documents(chunks: List[Dict[str, Any]]) -> List[Document]:
    """Convert chunk dictionaries into LangChain Document objects."""
    documents: List[Document] = []
    for chunk in chunks:
        metadata = dict(chunk.get("metadata", {}))
        documents.append(Document(page_content=chunk.get("content", ""), metadata=metadata))
    return documents


def retrieve(query: str, top_k: int = 10, rerank: bool = True) -> List[Dict[str, Any]]:
    """Retrieve candidates from ChromaDB and optionally rerank them with a cross-encoder.

    This is a two-stage flow when `rerank` is True:
    1. Use ChromaDB's vector search to fetch up to `min(top_k, 10)` candidate chunks by
       embedding distance.
    2. Rerank those candidates with a cross-encoder over (query, chunk_content)
       pairs and return the top 3 results.

    If `rerank` is False, return the top 3 candidates ordered by raw ChromaDB
    distance (ascending).
    """
    _, collection = _get_collection()
    model = _load_embedding_model()

    query_embedding = model.encode([query], convert_to_numpy=True)[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, 10),
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        return []

    if not rerank:
        scored = [
            {
                "content": doc,
                "metadata": metadata,
                "distance": float(distance),
                "rerank_score": None,
            }
            for doc, metadata, distance in zip(documents, metadatas, distances)
        ]
        scored.sort(key=lambda item: item["distance"])  # lower distance = more similar
        return scored[:3]

    cross_encoder = _load_cross_encoder()
    pairs = [(query, doc) for doc in documents]
    scores = cross_encoder.predict(pairs)

    scored = [
        {
            "content": doc,
            "metadata": metadata,
            "distance": float(distance),
            "rerank_score": float(score),
        }
        for doc, metadata, distance, score in zip(documents, metadatas, distances, scores)
    ]

    scored.sort(key=lambda item: item["rerank_score"], reverse=True)
    return scored[:3]
