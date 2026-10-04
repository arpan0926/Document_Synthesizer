from __future__ import annotations

import re
from typing import Any, Dict, List


def _estimate_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\w+|[\.,;:!?]", text)))


def _split_paragraphs(text: str) -> List[str]:
    if not text:
        return []

    paragraphs = re.split(r"\n\s*\n", text.strip())
    cleaned: List[str] = []
    for paragraph in paragraphs:
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if paragraph:
            cleaned.append(paragraph)
    return cleaned


def _split_large_paragraph(paragraph: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    """Split a single large paragraph into smaller chunks of bounded token count."""
    words = paragraph.split(" ")
    sub_chunks = []
    start = 0
    while start < len(words):
        end = start
        tokens = 0
        while end < len(words):
            word_tokens = _estimate_tokens(words[end])
            if tokens + word_tokens + 1 <= max_tokens or end == start:
                tokens += word_tokens + 1
                end += 1
            else:
                break
        sub_chunks.append(" ".join(words[start:end]))
        
        if end >= len(words):
            break
            
        # backtrack for overlap
        overlap_tokens_count = 0
        overlap_start = end
        while overlap_start > start:
            word_tokens = _estimate_tokens(words[overlap_start - 1])
            if overlap_tokens_count + word_tokens + 1 <= overlap_tokens:
                overlap_tokens_count += word_tokens + 1
                overlap_start -= 1
            else:
                break
        if overlap_start < end:
            start = overlap_start
        else:
            start = end
    return sub_chunks


def chunk_text(
    text: str,
    source_doc: str,
    page_number: int,
    max_tokens: int = 256,
    overlap_tokens: int = 32,
) -> List[Dict[str, Any]]:
    """Chunk text by paragraph/section while keeping token size bounded."""
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    chunks: List[Dict[str, Any]] = []
    start_index = 0
    parent_context = text.strip()  # Full-page context attached to child metadata

    while start_index < len(paragraphs):
        end_index = start_index
        token_count = 0

        while end_index < len(paragraphs):
            paragraph = paragraphs[end_index]
            paragraph_tokens = _estimate_tokens(paragraph)
            if token_count + paragraph_tokens + 1 <= max_tokens:
                token_count += paragraph_tokens + 1
                end_index += 1
            else:
                break

        chunk_paragraphs = paragraphs[start_index:end_index]
        if not chunk_paragraphs:
            # First paragraph is too large to fit in max_tokens
            large_paragraph = paragraphs[start_index]
            sub_chunks = _split_large_paragraph(large_paragraph, max_tokens, overlap_tokens)
            for sub_content in sub_chunks:
                chunks.append(
                    {
                        "content": sub_content,
                        "metadata": {
                            "source_doc": source_doc,
                            "page_number": page_number,
                            "chunk_type": "text",
                            "parent_context": parent_context,
                        },
                    }
                )
            start_index += 1
            continue

        content = " ".join(chunk_paragraphs)
        chunks.append(
            {
                "content": content,
                "metadata": {
                    "source_doc": source_doc,
                    "page_number": page_number,
                    "chunk_type": "text",
                    "parent_context": parent_context,
                },
            }
        )

        if end_index >= len(paragraphs):
            break

        overlap_paragraphs: List[str] = []
        overlap_count = 0
        for paragraph in reversed(chunk_paragraphs):
            paragraph_tokens = _estimate_tokens(paragraph)
            if overlap_count + paragraph_tokens + 1 <= overlap_tokens:
                overlap_paragraphs.insert(0, paragraph)
                overlap_count += paragraph_tokens + 1
            else:
                break

        if overlap_paragraphs:
            start_index = max(start_index + 1, end_index - len(overlap_paragraphs))
        else:
            start_index = end_index

    return chunks


def dataframe_to_markdown(df: Any) -> str:
    """Serialize a pandas DataFrame to a compact markdown table string."""
    headers = [str(column) for column in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy().tolist()]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def chunk_table(df: Any, source_doc: str, page_number: int) -> Dict[str, Any]:
    """Keep each table as a single chunk and attach markdown parent_context."""
    md_table = dataframe_to_markdown(df)
    return {
        "content": md_table,
        "metadata": {
            "source_doc": source_doc,
            "page_number": page_number,
            "chunk_type": "table",
            "parent_context": md_table,
            "df": df,
        },
    }
