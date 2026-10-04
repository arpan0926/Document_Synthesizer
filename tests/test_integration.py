import os
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fpdf import FPDF

import generation
import pdf_parser
import retrieval
import vector_store


class DummyEmbeddingModel:
    def encode(self, texts: str | List[str], convert_to_numpy: bool = False) -> Any:
        import numpy as np
        if isinstance(texts, str):
            arr = [float(len(texts))] + [0.0] * 383
            return np.array(arr) if convert_to_numpy else arr
        arr = [[float(len(text))] + [0.0] * 383 for text in texts]
        return np.array(arr) if convert_to_numpy else arr


class DummyFlashRankRanker:
    def rerank(self, rerank_req: Any) -> List[Dict[str, Any]]:
        passages = rerank_req.passages
        scored = []
        for p in passages:
            text = p["text"]
            score = 0.1
            if "92 percent" in text.lower() or "92%" in text.lower():
                score = 0.99
            elif "accuracy" in text.lower() and "experiment" in rerank_req.query.lower():
                score = 0.8
            scored.append({
                "id": p["id"],
                "text": text,
                "meta": p["meta"],
                "score": score
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored


@pytest.fixture(autouse=True)
def use_temp_chroma(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Use an isolated temporary ChromaDB storage for each test."""
    monkeypatch.setattr(vector_store, "_PERSIST_DIRECTORY", tmp_path / "chroma_db")
    monkeypatch.setattr(vector_store, "_COLLECTION_NAME", f"test_collection_{os.urandom(4).hex()}")
    monkeypatch.setattr(vector_store, "_EMBEDDING_MODEL", None)
    monkeypatch.setattr(retrieval, "_FLASHRANK_RANKER", None)
    monkeypatch.setattr(vector_store, "_load_embedding_model", lambda: DummyEmbeddingModel())
    monkeypatch.setattr(retrieval, "_load_embedding_model", lambda: DummyEmbeddingModel())
    monkeypatch.setattr(retrieval, "_load_flashrank_reranker", lambda: DummyFlashRankRanker())


def create_test_pdf(pdf_path: Path) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    text = (
        "This is a synthetic test PDF. "
        "The experiment achieved 92 percent accuracy on the validation set. "
        "This sentence is distinctive and searchable."
    )
    pdf.multi_cell(0, 10, text)
    pdf.output(str(pdf_path))


def test_ingest_retrieve_pipeline(tmp_path: Path) -> None:
    pdf_path = tmp_path / "test_doc.pdf"
    create_test_pdf(pdf_path)

    chunks = vector_store.ingest_document(pdf_path)
    assert chunks, "Ingestion should produce at least one chunk"

    query = "What accuracy did the experiment achieve?"
    results = retrieval.retrieve(query)

    assert results, "Retrieve should return at least one candidate"
    top = results[0]

    assert "92 percent" in top["content"].lower() or "92%" in top["content"].lower()
    assert top["metadata"]["source_doc"] == pdf_path.name
    assert top["metadata"]["page_number"] == 1


def test_reingestion_replaces_existing_document(tmp_path: Path) -> None:
    pdf_path = tmp_path / "test_doc.pdf"
    create_test_pdf(pdf_path)

    first_chunks = vector_store.ingest_document(pdf_path)
    second_chunks = vector_store.ingest_document(pdf_path)
    _, collection = vector_store._get_collection()

    assert len(first_chunks) == len(second_chunks)
    assert collection.count() == len(second_chunks)


def test_empty_index_returns_no_results() -> None:
    assert retrieval.retrieve("anything", top_k=3, rerank=False) == []


def test_retrieve_rejects_invalid_query() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        retrieval.retrieve("", top_k=3, rerank=False)


def test_parse_missing_pdf_reports_failure(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Unable to read PDF"):
        pdf_parser.parse_document(tmp_path / "missing.pdf")


def test_answer_query_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "test_doc.pdf"
    create_test_pdf(pdf_path)

    vector_store.ingest_document(pdf_path)

    def fake_call_ollama(prompt: str) -> str:
        return "The model reports 92 percent accuracy [test_doc.pdf, 1]."

    monkeypatch.setattr(generation, "_call_ollama", fake_call_ollama)

    result = generation.answer_query("What accuracy did the experiment achieve?")

    assert set(result.keys()) == {"answer", "context_chunks", "citations", "verification"}
    assert result["answer"] == "The model reports 92 percent accuracy [test_doc.pdf, 1]."
    assert result["citations"] == [{"source_doc": "test_doc.pdf", "page_number": 1}]
    assert result["verification"]["valid"] == [{"source_doc": "test_doc.pdf", "page_number": 1}]
    assert result["verification"]["flagged"] == []


def test_ollama_malformed_response_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class MalformedResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Dict[str, Any]:
            return {"unexpected": "payload"}

    monkeypatch.setattr(generation.requests, "post", lambda *args, **kwargs: MalformedResponse())

    with pytest.raises(RuntimeError, match="non-empty 'response'"):
        generation._call_ollama("test")
