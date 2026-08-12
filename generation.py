from __future__ import annotations

import os
import re
from typing import Any, Dict, List

import requests

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


def _call_huggingface(prompt: str) -> str:
    token = os.getenv("HF_API_TOKEN")
    if not token:
        return "No Hugging Face token configured; using fallback summary."

    model = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 300, "temperature": 0.1}}
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and data:
            return data[0].get("generated_text", "")
        if isinstance(data, dict):
            return data.get("generated_text", "")
    except Exception:
        return "Inference call failed; using fallback summary."
    return "Fallback summary due to an empty response."


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
    answer_text = _call_huggingface(prompt)
    citations = _extract_citations(answer_text)
    verification = verify_citations(citations, chunks)

    return {
        "answer": answer_text,
        "context_chunks": chunks,
        "citations": citations,
        "verification": verification,
    }
