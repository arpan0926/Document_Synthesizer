from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List

import requests

# Load .env from project root if present (optional, requires python-dotenv)
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is declared at runtime
    load_dotenv = None

root = Path(__file__).resolve().parent
env_path = root / ".env"
if load_dotenv is not None and env_path.exists():
    load_dotenv(dotenv_path=env_path)

from retrieval import retrieve, to_langchain_documents


def build_answer_prompt(question: str, chunks: List[Dict[str, Any]]) -> Any:
    """Build a LangChain-style prompt object for answer generation."""
    from langchain_core.prompts import PromptTemplate

    context = "\n\n".join(
        f"[{index}] Source: {chunk.get('metadata', {}).get('source_doc')} Page: {chunk.get('metadata', {}).get('page_number')}\n{chunk.get('content', '')}"
        for index, chunk in enumerate(chunks, start=1)
    )

    template = (
        "You are a careful research assistant. Answer the user's query using ONLY the provided context chunks. "
        "For every factual claim, cite the source as [source_doc, page_number]. If a statement cannot be verified by the context, say so.\n\n"
        "Question: {question}\n\nContext:\n{context}\n\nAnswer:"
    )
    return PromptTemplate(template=template, input_variables=["question", "context"])


def _build_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    prompt = build_answer_prompt(query, chunks)
    context = "\n\n".join(
        f"[{index}] Source: {chunk.get('metadata', {}).get('source_doc')} Page: {chunk.get('metadata', {}).get('page_number')}\n{chunk.get('content', '')}"
        for index, chunk in enumerate(chunks, start=1)
    )
    return prompt.format(question=query, context=context)


def _call_ollama(prompt: str) -> str:
    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }
    try:
        response = requests.post(
            os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/generate",
            json=payload,
            timeout=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")),
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise RuntimeError("Ollama inference request failed; verify the configured service is reachable") from exc
    except ValueError as exc:
        raise RuntimeError("Ollama returned malformed JSON") from exc

    answer = data.get("response") if isinstance(data, dict) else None
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("Ollama response did not contain a non-empty 'response' string")
    return answer


def _extract_citations(text: str) -> List[Dict[str, Any]]:
    pattern = re.compile(r"\[(?P<source>[^,\]]+),\s*(?P<page>\d+)\]")
    return [
        {"source_doc": match.group("source"), "page_number": int(match.group("page"))}
        for match in pattern.finditer(text)
    ]


def verify_citations(citations: List[Dict[str, Any]], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid: List[Dict[str, Any]] = []
    flagged: List[Dict[str, Any]] = []
    allowed = {
        (chunk.get("metadata", {}).get("source_doc"), chunk.get("metadata", {}).get("page_number"))
        for chunk in chunks
    }

    for citation in citations:
        if (citation["source_doc"], citation["page_number"]) in allowed:
            valid.append(citation)
        else:
            flagged.append(citation)

    return {"valid": valid, "flagged": flagged}


def answer_query(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Retrieve context, generate an answer, and verify citation claims."""
    chunks = retrieve(query, top_k=top_k)
    prompt = _build_prompt(query, chunks)
    answer_text = _call_ollama(prompt)
    citations = _extract_citations(answer_text)
    verification = verify_citations(citations, chunks)

    return {
        "answer": answer_text,
        "context_chunks": chunks,
        "citations": citations,
        "verification": verification,
    }
