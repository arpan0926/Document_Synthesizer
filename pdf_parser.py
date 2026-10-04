from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

try:
    import pdfplumber
except ImportError:  # pragma: no cover - optional runtime dependency
    pdfplumber = None

try:
    import camelot
except ImportError:  # pragma: no cover - optional runtime dependency
    camelot = None

from chunking import dataframe_to_markdown


def parse_document(pdf_path: str | Path) -> Dict[str, Any]:
    """Parse a PDF into page text and table candidates.

    The parser preserves page order and returns a document structure that can be
    consumed by the chunking and ingestion modules.
    """
    pdf_path = Path(pdf_path)
    pages: List[Dict[str, Any]] = []
    tables: List[Dict[str, Any]] = []

    if pdfplumber is None:
        raise RuntimeError("pdfplumber is required to parse PDF documents")

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                raw_text = page.extract_text() or ""

                extracted_tables = page.extract_tables() or []
                table_markdowns: List[str] = []

                for table in extracted_tables:
                    if not table or len(table) < 2:
                        continue
                    headers = [str(cell or "").strip().replace("\n", " ") for cell in table[0]]
                    header_line = "| " + " | ".join(headers) + " |"
                    separator = "| " + " | ".join(["---"] * len(headers)) + " |"

                    rows: List[str] = []
                    for row in table[1:]:
                        cleaned_row = [str(cell or "").strip().replace("\n", " ") for cell in row]
                        rows.append("| " + " | ".join(cleaned_row) + " |")

                    table_md = "\n".join([header_line, separator] + rows)
                    table_markdowns.append(table_md)

                pages.append({
                    "page_number": page_number,
                    "text": raw_text.strip(),
                })
                for table_markdown in table_markdowns:
                    tables.append({"content": table_markdown, "page": page_number})
    except (FileNotFoundError, PermissionError, OSError, ValueError) as exc:
        raise RuntimeError(f"Unable to read PDF '{pdf_path}': {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"PDF parsing failed for '{pdf_path}': {exc}") from exc

    if not tables and camelot is not None:
        for flavor in ("lattice", "stream"):
            try:
                extracted_tables = camelot.read_pdf(str(pdf_path), flavor=flavor, pages="all")
            except (OSError, ValueError, RuntimeError) as exc:
                if flavor == "stream":
                    raise RuntimeError(f"Fallback table extraction failed for '{pdf_path}': {exc}") from exc
                continue

            for table in extracted_tables:
                df = getattr(table, "df", None)
                if df is not None and not df.empty:
                    tables.append(
                        {
                            "df": df,
                            "page": int(getattr(table, "page", 1)),
                            "content": dataframe_to_markdown(df),
                        }
                    )
            if tables:
                break

    return {
        "source_doc": pdf_path.name,
        "pages": pages,
        "tables": tables,
        "text_chunks": [],
        "table_chunks": [
            {"content": item.get("content") or dataframe_to_markdown(item["df"]), "page": item["page"]}
            for item in tables
        ],
    }
