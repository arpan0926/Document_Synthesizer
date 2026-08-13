import os
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fpdf import FPDF

import generation
import retrieval
import vector_store


class DummyEmbeddingModel:
    def encode(self, texts: List[str], convert_to_numpy: bool = False) -> List[List[float]]:
        return [[float(len(text))] for text in texts]


class DummyCrossEncoder:
    def predict(self, pairs: List[tuple[str, str]]) -> List[float]:
        scores: List[float] = []
        for query, chunk in pairs:
            if "92 percent" in chunk.lower() or "92%" in chunk.lower():
                scores.append(0.99)
            elif "accuracy" in chunk.lower() and "experiment" in query.lower():
                scores.append(0.8)
            else:
                scores.append(0.1)
        return scores


@pytest.fixture(autouse=True)
def use_temp_chroma(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Use an isolated temporary ChromaDB storage for each test."""
    monkeypatch.setattr(vector_store, "_PERSIST_DIRECTORY", tmp_path / "chroma_db")
    monkeypatch.setattr(vector_store, "_COLLECTION_NAME", f"test_collection_{os.urandom(4).hex()}")
    monkeypatch.setattr(vector_store, "_EMBEDDING_MODEL", None)
    monkeypatch.setattr(retrieval, "_CROSS_ENCODER", None)
    monkeypatch.setattr(vector_store, "_load_embedding_model", lambda: DummyEmbeddingModel())
    monkeypatch.setattr(retrieval, "_load_cross_encoder", lambda: DummyCrossEncoder())


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


def test_answer_query_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "test_doc.pdf"
    create_test_pdf(pdf_path)

    vector_store.ingest_document(pdf_path)

    def fake_call_huggingface(prompt: str) -> str:
        return "The model reports 92 percent accuracy [test_doc.pdf, 1]."

    monkeypatch.setattr(generation, "_call_huggingface", fake_call_huggingface)

    result = generation.answer_query("What accuracy did the experiment achieve?")

    assert set(result.keys()) == {"answer", "context_chunks", "citations", "verification"}
    assert result["answer"] == "The model reports 92 percent accuracy [test_doc.pdf, 1]."
    assert result["citations"] == [{"source_doc": "test_doc.pdf", "page_number": 1}]
    assert result["verification"]["valid"] == [{"source_doc": "test_doc.pdf", "page_number": 1}]
    assert result["verification"]["flagged"] == []
