# Document Synthesizer

A local retrieval-augmented generation pipeline for question-answering over PDF collections. It extracts page text and tables, indexes chunks in ChromaDB, combines dense retrieval with BM25 using reciprocal-rank fusion, reranks candidates with FlashRank, and generates cited answers using a locally hosted Ollama model.

## Requirements and setup

- Python 3.10 or newer
- Ollama running locally, with the configured model downloaded
- The project dependencies from `requirements.txt`

```bash
python -m venv .venv
# Activate the environment for your shell, then:
python -m pip install -r requirements.txt
ollama pull qwen2.5:7b-instruct
```

Optional environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server base URL |
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` | Model used for answer generation |
| `OLLAMA_TIMEOUT_SECONDS` | `120` | Inference request timeout |

## CLI

Ingest PDFs from a folder:

```bash
python main.py --ingest-folder ./data
```

Ask a single question or start the interactive loop:

```bash
python main.py --query "What is the RECAST framework?"
python main.py
```

## HTTP API

Start the FastAPI service:

```bash
python app.py
# Or run Uvicorn directly:
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` for the chat interface or `/docs` for the API explorer.

- `GET /health` reports vector-store availability and indexed chunk count.
- `POST /ingest` accepts a PDF as multipart field `file`; uploads are limited to 50 MB.
- `POST /query` accepts JSON with `query`, optional `candidate_k` (default 40), and optional `result_k` (default 3).

The API has no authentication. Keep it on a trusted interface or add an authentication layer before exposing it to an untrusted network.

## Retrieval evaluation

Run the evaluator with a verified dataset and the same candidate pool used by the query path:

```bash
python tests/evaluate_retrieval.py --eval-file eval_set.json --candidate-k 40
```

The current `eval_set.json` contains 14 queries (not 10) across the four PDFs. Each expected page was checked against the local PDF's extracted text and uses the physical, 1-based page numbering stored by the parser. One run with `candidate_k=40` produced:

| Metric | Dense + BM25, no reranking | Dense + BM25 + FlashRank |
|---|---:|---:|
| Recall@3 (source and page) | 50.0% | 64.3% |
| MRR | 0.333 | 0.500 |
| Mean latency | 78 ms | 20,939 ms |
| P95 latency | 363 ms | 93,408 ms |

These are a single run on the current machine and index, not a general performance guarantee. The first reranked query took 93.4 seconds; the remaining 13 averaged 15.36 seconds. Reranking latency therefore includes a substantial cold-start cost, but remained high on subsequent queries too. Re-measure after model warm-up and on the intended deployment hardware before making a production latency decision. The result file is `eval_results.json`.

## Tests

```bash
python -m pytest
```

## Limitations

- Ollama, the embedding model, and FlashRank model must be available to the runtime; model loading can require network access on first use.
- Citation verification confirms that a cited document/page was retrieved, not that the cited passage entails the associated claim.
- Scanned PDFs without a text layer may produce little or no useful extracted text.
- The API currently runs ingestion and inference in-process; it has no authentication, durable job queue, or multi-worker coordination.
