from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from vector_store import ingest_document
from generation import answer_query


def ingest_folder(folder_path: str | Path) -> List[str]:
    """Ingest every PDF in a folder and return the ingested filenames."""
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(folder)

    pdfs = sorted(folder.glob("*.pdf"))
    for pdf_path in pdfs:
        ingest_document(pdf_path)
    return [pdf_path.name for pdf_path in pdfs]


def interactive_loop() -> None:
    """Run a simple interactive query loop until the user exits."""
    print("Document Synthesizer ready. Type 'quit' to exit.")
    while True:
        query = input("Query> ").strip()
        if query.lower() in {"", "quit", "exit"}:
            break
        result = answer_query(query)
        print("\nAnswer:")
        print(result["answer"])
        print("\nVerification:")
        print(result["verification"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Complex Document Synthesizer")
    parser.add_argument("--ingest-folder", type=str, help="Folder containing PDF documents to ingest")
    parser.add_argument("--query", type=str, help="Single query to run")
    args = parser.parse_args()

    if args.ingest_folder:
        ingested = ingest_folder(args.ingest_folder)
        print(f"Ingested {len(ingested)} PDFs: {', '.join(ingested)}")

    if args.query:
        result = answer_query(args.query)
        print(result["answer"])
        return

    interactive_loop()


if __name__ == "__main__":
    main()
