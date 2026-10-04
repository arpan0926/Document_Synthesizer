# Document Synthesizer

A retrieval-augmented generation (RAG) pipeline for question-answering over collections of complex, multi-page PDF documents (research papers, reports) with tables and mixed content. Built around a two-stage retrieval architecture (dense vector search + cross-encoder reranking) and citation verification to reduce hallucinated source attribution.

## What it does

Given a folder of PDFs, the system:
1. Parses each document into per-page text and extracts tables separately.
2. Chunks text into small, embedding-friendly units while preserving full-page context for downstream use.
3. Embeds and stores chunks in a persistent ChromaDB vector store.
4. On a query, retrieves candidate chunks via dense vector search, reranks them with a cross-encoder, and generates an answer that must cite `[source_doc, page_number]` for every claim.
5. Verifies each generated citation against the actually-retrieved chunks and flags any that don't match, to surface potential hallucinations.

## Architecture

```
PDF files
   │
   ▼
pdf_parser.py    → extracts per-page text (pdfplumber) + tables (camelot, lattice/stream)
   │
   ▼
chunking.py      → splits page text into ~256-token paragraph-aligned chunks with overlap;
                    attaches full-page text as parent_context metadata (small-to-big retrieval);
                    tables are kept as a single markdown-serialized chunk
   │
   ▼
vector_store.py  → embeds chunks (BAAI/bge-small-en-v1.5) and stores in ChromaDB
   │
   ▼
retrieval.py     → stage 1: dense vector search (top 10 by embedding distance)
                    stage 2: cross-encoder reranking (ms-marco-MiniLM-L-6-v2) → top 3
   │
   ▼
generation.py    → builds a grounded prompt, calls an LLM (Hugging Face Inference API),
                    extracts [source, page] citations from the answer, and verifies
                    each one against the retrieved chunk metadata
   │
   ▼
main.py          → CLI: ingest a folder of PDFs, run a single query, or an interactive loop
```

### Design decisions worth calling out

- **Small-to-big chunking**: chunks used for embedding/matching are small (~256 tokens) for precision, but each chunk carries its full source page as `parent_context`, so a generation step can expand context without re-retrieving.
- **Two-stage retrieval**: vector search is cheap but imprecise; a cross-encoder reranker over (query, chunk) pairs is expensive but more accurate. Both are implemented so the trade-off is measurable (see Evaluation below) rather than assumed.
- **Citation verification**: the model is instructed to cite `[source_doc, page_number]` for every claim. `verify_citations()` checks each citation against the metadata of chunks that were actually retrieved, separating `valid` from `flagged` citations — a lightweight, deterministic guard against citation hallucination.
- **Table handling**: tables are extracted independently of text (via `camelot`) and stored as their own chunks, serialized to markdown for embedding while keeping the original DataFrame in metadata for potential structured use later.

## Evaluation

An evaluation harness (`eval_set.json` → `eval_results.json`) tests retrieval against 30 hand-written queries spanning 4 arXiv papers, checking whether the expected `(source_doc, page_number)` appears in the top-3 retrieved chunks.

| Metric | Vector search only | + Cross-encoder reranking |
|---|---|---|
| Recall@3 | 33.3% | 40.0% |
| MRR | 0.300 | 0.306 |
| Mean latency | ~18 ms | ~756 ms |
| P95 latency | ~23 ms | ~804 ms |

**Takeaways:**
- Reranking gives a modest recall improvement (~7 points absolute) at roughly a **40x latency cost**, which is a real trade-off to weigh depending on the target use case (interactive vs. batch).
- Misses are frequently "near misses" — the correct source document is retrieved, but the specific page is off, suggesting chunk-boundary effects and single-page ground truth may both be contributing factors. This is an active area for improvement (see below).

Re-run the evaluation with:
```bash
python eval.py
```

## Known limitations / next steps

- Recall@3 (~40%) has clear room to improve — candidate fixes: a larger embedding model than `bge-small`, hybrid dense+BM25 retrieval (a sparse document index already exists in `vector_store.py` but isn't wired into `retrieval.py` yet), and chunk-size/overlap tuning.
- PDF parsing failures are currently swallowed silently (broad `except: pass`); this should be replaced with logging so ingestion failures are visible.
- Text and table retrieval aren't currently weighted differently, so a table-relevant query can be crowded out by text chunks or vice versa.
- The Hugging Face Inference API is used for generation; response quality and availability depend on the configured model and API rate limits.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root (this file is git-ignored and should never be committed):
```
HF_API_TOKEN=your_huggingface_token
HF_MODEL=meta-llama/Llama-3.1-8B-Instruct   # optional override
```

## Usage

Ingest a folder of PDFs:
```bash
python main.py --ingest-folder ./data
```

Run a single query:
```bash
python main.py --query "What is the RECAST framework and what does it do?"
```

Or start an interactive session:
```bash
python main.py
```

## Project structure

```
├── main.py           # CLI entry point (ingest / query / interactive loop)
├── pdf_parser.py      # PDF → page text + tables
├── chunking.py        # Text/table chunking with parent-page context
├── vector_store.py    # Embedding + ChromaDB persistence
├── retrieval.py        # Dense retrieval + cross-encoder reranking
├── generation.py       # Prompting, LLM calls, citation extraction/verification
├── eval_set.json       # 30 evaluation queries with expected (doc, page)
├── eval_results.json   # Recall@3 / MRR / latency results
├── data/                # Source PDFs (not committed)
└── chroma_db/           # Persisted vector store (not committed)
```

## Tech stack

Python · ChromaDB · Sentence-Transformers (`bge-small-en-v1.5`) · Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) · pdfplumber · camelot · LangChain (prompting) · Hugging Face Inference API
