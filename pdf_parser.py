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


def parse_document(pdf_path: str | Path) -> Dict[str, Any]:
    """Parse a PDF into page text and table candidates.

    The parser preserves page order and returns a document structure that can be
    consumed by the chunking and ingestion modules.
    """
    pdf_path = Path(pdf_path)
    pages: List[Dict[str, Any]] = []
    tables: List[Dict[str, Any]] = []

    if pdfplumber is None:
        return {"source_doc": pdf_path.name, "pages": pages, "tables": tables}

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                raw_text = page.extract_text() or ""
                
                # Extract tables via pdfplumber and convert to Markdown
                extracted_tables = page.extract_tables() or []
                table_markdowns: List[str] = []
                
                for table in extracted_tables:
                    if not table or len(table) < 2:
                        continue
                    headers = [str(cell or "").strip().replace("\n", " ") for cell in table[0]]
                    header_line = "| " + " | ".join(headers) + " |"
                    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
                    
                    rows = []
                    for row in table[1:]:
                        cleaned_row = [str(cell or "").strip().replace("\n", " ") for cell in row]
                        rows.append("| " + " | ".join(cleaned_row) + " |")
                        
                    table_md = "\n".join([header_line, separator] + rows)
                    table_markdowns.append(table_md)
                
                full_page_text = raw_text.strip()
                if table_markdowns:
                    full_page_text += "\n\n### Extracted Tables ###\n" + "\n\n".join(table_markdowns)
                    
                pages.append({
                    "page_number": page_number,
                    "text": full_page_text,
                })
    except Exception:
        pages = []

    if camelot is not None:
        for flavor in ("lattice", "stream"):
            try:
                extracted_tables = camelot.read_pdf(str(pdf_path), flavor=flavor, pages="all")
            except Exception:
                extracted_tables = []

            for table in extracted_tables:
                try:
                    df = table.df
                    if df.empty:
                        continue
                    tables.append({"df": df, "page": int(getattr(table, "page", 1))})
                except Exception:
                    continue

            if tables:
                break

    return {
        "source_doc": pdf_path.name,
        "pages": pages,
        "tables": tables,
        "text_chunks": [],
        "table_chunks": [{"df": item["df"], "page": item["page"]} for item in tables],
    }
