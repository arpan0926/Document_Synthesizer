from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from generation import answer_query
from vector_store import _get_collection, ingest_document


logger = logging.getLogger(__name__)
app = FastAPI(title="Document Synthesizer", version="1.0.0")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
FRONTEND_FILE = Path(__file__).resolve().parent / "frontend" / "index.html"


class QueryRequest(BaseModel):
	query: str = Field(min_length=1, max_length=4000)
	candidate_k: int = Field(default=40, ge=1, le=500)
	result_k: int = Field(default=3, ge=1, le=20)

	@field_validator("query")
	@classmethod
	def query_must_contain_text(cls, value: str) -> str:
		if not value.strip():
			raise ValueError("query must contain non-whitespace text")
		return value


class QueryResponse(BaseModel):
	answer: str
	context_chunks: List[Dict[str, Any]]
	citations: List[Dict[str, Any]]
	verification: Dict[str, List[Dict[str, Any]]]


class IngestResponse(BaseModel):
	source_doc: str
	chunk_count: int


@app.exception_handler(Exception)
async def handle_unexpected_error(_request, exc: Exception) -> JSONResponse:
	logger.exception("Unhandled API request failure", exc_info=exc)
	return JSONResponse(
		status_code=500,
		content={"error": {"code": "internal_error", "message": "Request failed; check service logs"}},
	)


@app.get("/health")
async def health() -> JSONResponse:
	try:
		_, collection = await run_in_threadpool(_get_collection)
		return JSONResponse({"status": "ok", "indexed_chunks": await run_in_threadpool(collection.count)})
	except Exception as exc:
		logger.exception("Health check failed", exc_info=exc)
		return JSONResponse(
			status_code=503,
			content={"error": {"code": "unavailable", "message": "Vector store is unavailable"}},
		)


@app.get("/", include_in_schema=False)
async def frontend() -> FileResponse:
	return FileResponse(FRONTEND_FILE)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
	filename = Path(file.filename or "").name
	if not filename or Path(filename).suffix.lower() != ".pdf":
		raise HTTPException(status_code=400, detail="Upload must be a PDF file")

	try:
		with tempfile.TemporaryDirectory(prefix="document-synthesizer-") as temp_dir:
			pdf_path = Path(temp_dir) / filename
			size = 0
			with pdf_path.open("wb") as destination:
				while data := await file.read(1024 * 1024):
					size += len(data)
					if size > MAX_UPLOAD_BYTES:
						raise HTTPException(status_code=413, detail="PDF exceeds the 50 MB upload limit")
					destination.write(data)

			chunks = await run_in_threadpool(ingest_document, pdf_path)
		return IngestResponse(source_doc=filename, chunk_count=len(chunks))
	finally:
		await file.close()


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
	result = await run_in_threadpool(
		answer_query,
		request.query,
		candidate_k=request.candidate_k,
		result_k=request.result_k,
	)
	return QueryResponse(**jsonable_encoder(result))


if __name__ == "__main__":
	import uvicorn

	uvicorn.run(app, host="127.0.0.1", port=8000)
