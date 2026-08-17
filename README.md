# Document Synthesizer

A baseline Retrieval-Augmented Generation (RAG) pipeline for complex PDF documents, including text and tables.

## What it does
- Ingests multi-page PDFs using `pdfplumber`
- Detects tables with `camelot-py`
- Chunks text by paragraph while preserving page context
- Embeds chunks with `sentence-transformers`
- Stores vectors in ChromaDB
- Retrieves and reranks results for query answering
- Builds citation-aware prompts for generation

## Project structure
- `pdf_parser.py`: PDF parsing and table extraction
- `chunking.py`: Text and table chunking
- `vector_store.py`: Embedding and ChromaDB storage
- `retrieval.py`: Retrieval and reranking helpers
- `generation.py`: Answer generation + citation verification
- `main.py`: CLI for ingestion and interactive querying

## Setup
1. Create and activate a virtual environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
2. Install dependencies
```powershell
python -m pip install -r requirements.txt
```

## Usage
1. Place PDFs in a folder, e.g. `data/`
2. Ingest:
```powershell
python main.py --ingest-folder data
```
3. Query interactively:
```powershell
python main.py
```

## Notes
- Use text-based PDFs for best results
-- Set `HF_API_TOKEN` environment variable for Hugging Face generation
	and optionally set `HF_MODEL` to `meta-llama/Llama-3.1-8B-Instruct` for best results.

## Tests
```powershell
python -m unittest discover -s tests -v
```
