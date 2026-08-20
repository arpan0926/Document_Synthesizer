from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document

from vector_store import _get_collection, _load_embedding_model, _get_all_documents, _load_bm25_index

_FLASHRANK_RANKER = None


def _load_flashrank_reranker():
    """Lazy-load CPU-optimized FlashRank reranker (<50ms latency)."""
    global _FLASHRANK_RANKER
    if _FLASHRANK_RANKER is None:
        try:
            from flashrank import Ranker
        except ImportError as exc:
            raise RuntimeError("Install flashrank to enable lightweight reranking") from exc

        _FLASHRANK_RANKER = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    return _FLASHRANK_RANKER


def to_langchain_documents(chunks: List[Dict[str, Any]]) -> List[Document]:
    """Convert chunk dictionaries into LangChain Document objects using parent context."""
    documents: List[Document] = []
    for chunk in chunks:
        metadata = dict(chunk.get("metadata", {}))
        # Use parent context window for LLM generation if present
        content = metadata.get("parent_context", chunk.get("content", ""))
        documents.append(Document(page_content=content, metadata=metadata))
    return documents


def _reciprocal_rank_fusion(
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Combine sparse and dense candidate lists using Reciprocal Rank Fusion (RRF)."""
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    for rank, doc in enumerate(vector_results, start=1):
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_map[doc_id] for doc_id, _ in sorted_docs]


def retrieve(query: str, top_k: int = 40, rerank: bool = True) -> List[Dict[str, Any]]:
    """Two-stage retrieval pipeline:
    1. Expanded Dense Vector Search (candidate_k = 40) + Sparse BM25 Search.
    2. Reciprocal Rank Fusion (RRF).
    3. FlashRank stage-2 reranking returning top 3 candidates.
    """
    _, collection = _get_collection()
    model = _load_embedding_model()

    # 1. First-Stage Dense Vector Search
    query_embedding = model.encode(query, convert_to_numpy=True).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    vector_ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    vector_candidates = [
        {
            "id": v_id,
            "content": doc,
            "metadata": metadata,
            "distance": float(dist),
        }
        for v_id, doc, metadata, dist in zip(vector_ids, documents, metadatas, distances)
    ]

    # 2. First-Stage Sparse BM25 Search
    all_docs = _get_all_documents()
    bm25 = _load_bm25_index()
    bm25_candidates = []
    
    if all_docs and bm25:
        try:
            query_tokens = query.lower().split()
            bm25_scores = bm25.get_scores(query_tokens)

            top_bm25_indices = sorted(
                range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
            )[:top_k]
            bm25_candidates = [
                {
                    "id": all_docs[idx]["id"],
                    "content": all_docs[idx]["content"],
                    "metadata": all_docs[idx]["metadata"],
                    "distance": 0.0,
                }
                for idx in top_bm25_indices
                if bm25_scores[idx] > 0
            ]
        except Exception:
            bm25_candidates = []

    # 3. Reciprocal Rank Fusion (RRF)
    fused_candidates = (
        _reciprocal_rank_fusion(vector_candidates, bm25_candidates)
        if bm25_candidates
        else vector_candidates
    )

    if not fused_candidates:
        return []

    if not rerank:
        return fused_candidates[:3]

    # 4. Stage-2 Fast Reranking via FlashRank
    try:
        from flashrank import RerankRequest

        ranker = _load_flashrank_reranker()
        passages = [
            {"id": item["id"], "text": item["content"], "meta": item["metadata"]}
            for item in fused_candidates
        ]
        rerank_req = RerankRequest(query=query, passages=passages)
        reranked_raw = ranker.rerank(rerank_req)

        scored = [
            {
                "content": item["text"],
                "metadata": item["meta"],
                "distance": 0.0,
                "rerank_score": float(item.get("score", 0.0)),
            }
            for item in reranked_raw
        ]
        scored.sort(key=lambda item: item["rerank_score"], reverse=True)
        return scored[:3]
    except Exception:
        return fused_candidates[:3]
