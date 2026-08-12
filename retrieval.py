from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document

from vector_store import _get_collection, _load_embedding_model


def to_langchain_documents(chunks: List[Dict[str, Any]]) -> List[Document]:
    """Convert chunk dictionaries into LangChain Document objects."""
    documents: List[Document] = []
    for chunk in chunks:
        metadata = dict(chunk.get("metadata", {}))
        documents.append(Document(page_content=chunk.get("content", ""), metadata=metadata))
    return documents


def retrieve(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Run vector retrieval and rerank the top candidates with a lightweight LangChain-style flow."""
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

    ranked = sorted(zip(documents, metadatas, distances), key=lambda item: item[2])
    reranked: List[Dict[str, Any]] = []
    for doc, metadata, distance in ranked[:3]:
        reranked.append(
            {
                "content": doc,
                "metadata": metadata,
                "distance": float(distance),
                "rerank_score": float(-distance),
            }
        )

    return reranked
