"""
=============================================================================
FastAPI Application — REST API
=============================================================================
Endpoints:
  POST /extract        — Extract structured data from receipt image
  POST /query          — Answer natural language questions about receipts
  GET  /receipts       — List all stored receipts
  GET  /receipts/{id}  — Get a specific receipt
  GET  /health         — Health check
=============================================================================
"""

import io
import json
import os
import sys
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# Lazy imports (loaded on startup)
extractor  = None
store      = None
qa_engine  = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    global extractor, store, qa_engine
    from src.extraction.extractor import ReceiptExtractor
    from src.storage.store import ReceiptStore
    from src.api.qa_engine import QAEngine

    MODEL_DIR = os.environ.get("MODEL_DIR", "outputs/checkpoints/final_lora")
    extractor  = ReceiptExtractor(model_dir=MODEL_DIR)
    store      = ReceiptStore()
    qa_engine  = QAEngine(store)
    print("✓ All components loaded.")
    yield
    store.close()


app = FastAPI(
    title="Receipt Information Extraction API",
    description="Extract structured data from receipt images and query via NL",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    receipt_id: Optional[str] = None   # None = search across all receipts


class ExtractionResponse(BaseModel):
    receipt_db_id: str
    vendor: str
    date: str
    total: str
    receipt_id: str
    confidence: dict
    raw_spans: dict


class QueryResponse(BaseModel):
    question: str
    answer: str
    source_receipts: list
    reasoning: str


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": extractor is not None}


@app.post("/extract", response_model=ExtractionResponse)
async def extract_receipt(file: UploadFile = File(...)):
    """
    Upload a receipt image and extract structured information.
    Returns: vendor, date, total, receipt_id, and confidence scores.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files are accepted.")

    contents = await file.read()
    image    = Image.open(io.BytesIO(contents)).convert("RGB")

    try:
        result = extractor.extract(image)
    except Exception as e:
        raise HTTPException(500, f"Extraction failed: {str(e)}")

    # Persist to database
    db_id = store.save(result)

    return ExtractionResponse(
        receipt_db_id=db_id,
        vendor=result.get("vendor", ""),
        date=result.get("date", ""),
        total=result.get("total", ""),
        receipt_id=result.get("receipt_id", ""),
        confidence=result.get("confidence", {}),
        raw_spans=result.get("raw_spans", {}),
    )


@app.post("/query", response_model=QueryResponse)
async def query_receipts(request: QueryRequest):
    """
    Answer natural language questions about receipts.
    Optionally scope to a specific receipt by receipt_id.
    """
    try:
        answer = qa_engine.answer(request.question, request.receipt_id)
    except Exception as e:
        raise HTTPException(500, f"QA failed: {str(e)}")
    return answer


@app.get("/receipts")
def list_receipts():
    """List all stored receipts."""
    return store.get_all()


@app.get("/receipts/{receipt_db_id}")
def get_receipt(receipt_db_id: str):
    """Get full details of a specific receipt."""
    result = store.get(receipt_db_id)
    if not result:
        raise HTTPException(404, "Receipt not found.")
    return result


@app.get("/receipts/search/")
def search_receipts(q: str = Query(..., description="Full-text search query")):
    """Search receipts by vendor, date, or total."""
    return store.search(q)
