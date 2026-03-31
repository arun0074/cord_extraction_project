"""
=============================================================================
FastAPI Application — REST API
=============================================================================
"""

# ── Must be set before ANY other import ──────────────────────────────────
import os
os.environ["FLAGS_use_mkldnn"]    = "0"
os.environ["PADDLE_DISABLE_ONEDNN"] = "1"

# ── Standard imports ─────────────────────────────────────────────────────
import io
import re
import traceback
import numpy as np
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# ── Surya imports (module level — not inside lifespan) ───────────────────
from surya.ocr import run_ocr as surya_run_ocr
from surya.model.detection.model   import load_model     as load_det_model, \
                                          load_processor  as load_det_processor
from surya.model.recognition.model     import load_model     as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor

# ── Global component references ──────────────────────────────────────────
extractor     = None
store         = None
qa_engine     = None
det_model     = None
det_processor = None
rec_model     = None
rec_processor = None


# ── Startup / shutdown ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global extractor, store, qa_engine
    global det_model, det_processor, rec_model, rec_processor

    from src.extraction.extractor import ReceiptExtractor
    from src.storage.store import ReceiptStore
    from src.api.qa_engine import QAEngine

    MODEL_DIR = os.environ.get("MODEL_DIR", "outputs/checkpoints/final_lora")

    extractor = ReceiptExtractor(model_dir=MODEL_DIR)
    store     = ReceiptStore()
    qa_engine = QAEngine(store)

    det_processor = load_det_processor()
    det_model     = load_det_model()
    rec_model     = load_rec_model()
    rec_processor = load_rec_processor()

    print("✓ All components loaded.")
    yield
    store.close()


# ── App ──────────────────────────────────────────────────────────────────

app = FastAPI(title="Receipt API", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ──────────────────────────────────────────────────────

class ExtractionResponse(BaseModel):
    receipt_db_id: str
    vendor:        str
    date:          str
    total:         str
    receipt_id:    str
    confidence:    dict
    raw_spans:     dict

class QueryRequest(BaseModel):
    question:   str
    receipt_id: Optional[str] = None


# ── OCR helper ───────────────────────────────────────────────────────────

def run_ocr(image: Image.Image):
    """Run Surya OCR and return (words, boxes) in LayoutLMv3 [0-1000] scale."""
    w, h = image.size

    predictions = surya_run_ocr(
        [image], [["en"]],
        det_model, det_processor,
        rec_model, rec_processor,
    )

    words, boxes = [], []
    for line in predictions[0].text_lines:
        txt = line.text.strip()
        if not txt:
            continue

        x0, y0, x1, y1 = line.bbox
        line_words = txt.split()
        word_w = (x1 - x0) / max(len(line_words), 1)

        for i, word in enumerate(line_words):
            wx0 = int((x0 + i       * word_w) / w * 1000)
            wx1 = int((x0 + (i + 1) * word_w) / w * 1000)
            wy0 = int(y0 / h * 1000)
            wy1 = int(y1 / h * 1000)
            words.append(word)
            boxes.append([
                max(0, min(wx0, 1000)),
                max(0, min(wy0, 1000)),
                max(0, min(wx1, 1000)),
                max(0, min(wy1, 1000)),
            ])

    return words, boxes


# ── Regex fallback (when model confidence is low) ────────────────────────

def fallback_extract(words):
    text = " ".join(words)
    result = {}

    m = re.search(r"\d+\.\d{2}", text)
    if m:
        result["total"] = m.group()

    m = re.search(r"\d{2}[/-]\d{2}[/-]\d{2,4}", text)
    if m:
        result["date"] = m.group()

    if words:
        result["vendor"] = words[0]

    return result


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": extractor is not None}


@app.get("/receipts/")
def list_receipts():
    return store.get_all()


@app.get("/receipts/{receipt_db_id}")
def get_receipt(receipt_db_id: str):
    result = store.get(receipt_db_id)
    if not result:
        raise HTTPException(404, "Receipt not found.")
    return result


@app.get("/receipts/search/")
def search_receipts(q: str = Query(..., description="Full-text search")):
    return store.search(q)


@app.post("/extract", response_model=ExtractionResponse)
async def extract_receipt(file: UploadFile = File(...)):
    """Upload a receipt image and extract structured information."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files are accepted.")

    try:
        contents = await file.read()
        image    = Image.open(io.BytesIO(contents)).convert("RGB")

        # OCR
        words, boxes = run_ocr(image)
        print("\nWORDS:", words[:20])

        # LayoutLMv3 extraction
        result = extractor.extract(image, words=words, boxes=boxes)
        print("MODEL RESULT:", result)

        # Regex fallback for any empty fields
        if not result.get("total"):
            fallback = fallback_extract(words)
            result.update({k: v for k, v in fallback.items() if not result.get(k)})

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))

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


@app.post("/query")
async def query_receipts(request: QueryRequest):
    """Answer natural language questions about receipts."""
    try:
        answer = qa_engine.answer(request.question, request.receipt_id)
    except Exception as e:
        raise HTTPException(500, f"QA failed: {str(e)}")
    return answer
