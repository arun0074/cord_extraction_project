# Receipt Information Extraction & QA System

> Fine-tuned LayoutLMv3 + LoRA + FastAPI + Gradio

---

## Table of Contents
1. [Architecture](#architecture)  
2. [Dataset Analysis Findings](#dataset-analysis-findings)  
3. [Model & PEFT Decisions](#model--peft-decisions)  
4. [Evaluation Metrics](#evaluation-metrics)  
5. [Setup & Running](#setup--running)  
6. [API Reference](#api-reference)  
7. [Design Trade-offs](#design-trade-offs)  
8. [Limitations & Future Work](#limitations--future-work)

---

## Architecture

```
Receipt Image
      │
      ▼
┌─────────────────────────────────────┐
│  LayoutLMv3-base + LoRA (fine-tuned)│  ← Token Classification (NER)
│  Input: image + OCR text + bboxes   │
│  Output: BIO label per token        │
└──────────────┬──────────────────────┘
               │ Span extraction
               ▼
        Structured JSON
   {vendor, date, total, receipt_id}
               │
               ▼
         SQLite + FTS5
               │
       ┌───────┴───────┐
       ▼               ▼
  FastAPI REST     Gemini Flash
  /extract          (grounded QA)
  /query               │
       │               ▼
       └──────► Gradio UI
```

---

## Dataset Analysis Findings

Run `python notebooks/01_dataset_analysis.py` to reproduce.

| Metric | Value | Impact on Decisions |
|--------|-------|---------------------|
| Train samples | 800 (subset) | Chose LoRA over full fine-tune |
| Median token length | ~180 | max_length=512 covers 99%+ |
| p95 token length | ~490 | Confirmed 512 is correct limit |
| BBox range | [0, 1000] | No rescaling needed for LayoutLMv3 |
| TOTAL field coverage | ~95% | High; model will learn well |
| DATE field coverage | ~80% | Moderate; some samples lack date |
| VENDOR field coverage | ~70% | Lowest; hardest to extract |
| O-label token % | ~87% | Class weights applied to loss |

---

## Model & PEFT Decisions

### Why LayoutLMv3?

| Model | Text | Layout | Image | Best for CORD? |
|-------|------|--------|-------|----------------|
| BERT/RoBERTa | ✓ | ✗ | ✗ | No — ignores spatial layout |
| LayoutLMv1 | ✓ | ✓ | ✗ | Partial — no image features |
| LayoutLMv2 | ✓ | ✓ | Separate | Better — but separate streams |
| **LayoutLMv3** | ✓ | ✓ | **Unified** | **Yes — joint cross-modal attention** |

### Why LoRA (r=8, alpha=16)?

```
Full fine-tune:  125M trainable params → overfits on 800 samples
LoRA r=4:        Too few params for 9-class NER (underfits)
LoRA r=8:        ~1.2M params (1% of total) ← CHOSEN
LoRA r=16:       Risk of memorisation at this data scale
LoRA r=32:       Definite overfit territory
```

`alpha/r = 16/8 = 2.0` — scaling factor recommended by original LoRA paper.

### Why class-weighted loss?

87% of tokens are O-label. Without weights, the model achieves 87% accuracy by always predicting O, but entity F1 ≈ 0. Balanced class weights up-weight named entity classes by ~5-8×.

---

## Evaluation Metrics

The system reports metrics at **three levels**:

### A. Token-Level (seqeval — NER standard)
- **Entity F1** — primary metric; requires correct boundary AND label
- **Precision** — fraction of predicted entities that are correct
- **Recall** — fraction of true entities that are found
- **Per-entity F1** — separate for VENDOR, DATE, TOTAL, RECEIPT_ID

### B. Calibration
- **ECE (Expected Calibration Error)** — lower = model confidence is trustworthy
- Reliability diagram plotted to `outputs/evaluation/calibration.png`

### C. Field-Level Extraction
- **Exact Match** — normalised string equality (strips currency symbols, whitespace)
- **Partial Match** — character-overlap ≥ 70% (lenient for vendor names)

### Error Analysis
- False Negatives (missed entities)
- False Positives (hallucinated entities)  
- Wrong boundaries
- Wrong entity type

---

## Setup & Running

### Option A: Google Colab (Recommended)

```python
# In a Colab cell:
!git clone https://github.com/your-repo/cord-extraction
%cd cord-extraction
!pip install -r requirements_notebook.txt -q

# Step 1: Analyse dataset
!python notebooks/01_dataset_analysis.py

# Step 2: Train
!python notebooks/02_model_selection_and_training.py
!pip install -r requirements.txt -q
# Step 3: Launch API + UI (two separate cells)
!uvicorn src.api.app:app --host 0.0.0.0 --port 8000 &
!python src/ui/app.py   # opens public Gradio URL
```

### Option B: Local Machine

```bash
git clone https://github.com/your-repo/cord-extraction
cd cord-extraction

python -m venv venv && source venv/bin/activate
pip install -r requirements_notebook.txt

# (Optional) Set Gemini API key for LLM-based QA
export GEMINI_API_KEY="your-key-here"

# Run in order:
python notebooks/01_dataset_analysis.py
python notebooks/02_model_selection_and_training.py

# Start API (terminal 1)
!pip install -r requirements.txt -q
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Start UI (terminal 2)
python src/ui/app.py
```

---

## API Reference

### `POST /extract`
Upload a receipt image, returns structured JSON.

```bash
curl -X POST http://localhost:8000/extract \
  -F "file=@receipt.jpg"
```

**Response:**
```json
{
  "receipt_db_id": "3f2a1b...",
  "vendor": "SUPER MART",
  "date": "2024/03/15",
  "total": "142.50",
  "receipt_id": "",
  "confidence": {"VENDOR": 0.94, "TOTAL": 0.98, "DATE": 0.91},
  "raw_spans": {"VENDOR": ["SUPER MART"], "TOTAL": ["142.50"]}
}
```

### `POST /query`
Ask a natural language question.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the total amount?", "receipt_id": "3f2a1b..."}'
```

**Response:**
```json
{
  "question": "What is the total amount?",
  "answer": "The total amount is 142.50.",
  "source_receipts": ["3f2a1b..."],
  "reasoning": "Retrieved TOTAL field from receipt 3f2a1b."
}
```

### `GET /receipts`
List all stored receipts.

### `GET /receipts/{id}`
Get a specific receipt by DB ID.

---

## Design Trade-offs

| Decision | Alternative | Why We Chose This |
|----------|-------------|-------------------|
| LayoutLMv3 | LayoutLMv1, BERT | Only v3 unifies image+text+layout |
| LoRA | Full fine-tune, Adapters | Best param-efficiency for small data |
| SQLite | Postgres, Vector DB | Zero setup, sufficient for structured data |
| Gemini Flash | GPT-4, local LLM | Free tier, low latency, good reasoning |
| Gradio | React, Streamlit | Fastest to build, integrates with Python |
| BIO tagging | Single-label | Handles multi-token entities correctly |
| Seqeval | Accuracy only | Entity-level F1 is the correct NER metric |

---

## Limitations & Future Work

### Current Limitations
- OCR quality: model depends on clean OCR text; handwritten receipts will fail
- RECEIPT_ID: only ~10% of CORD receipts have this field → low recall expected
- Languages: trained on English/Korean CORD — other languages not supported

### Future Improvements
- **More data**: Use full CORD training set (9,000+ samples) + SROIE + FUNSD
- **Donut**: End-to-end OCR-free model (avoids OCR errors entirely)
- **Multi-document QA**: Build a proper RAG pipeline with vector embeddings
- **Confidence thresholding**: Reject low-confidence extractions rather than returning empty strings
- **Active learning**: Use model uncertainty to select most informative samples for annotation

---

## File Structure

```
cord_extraction/
├── notebooks/
│   ├── 01_dataset_analysis.py          ← Run first
│   └── 02_model_selection_and_training.py
├── src/
│   ├── extraction/
│   │   └── extractor.py                ← Inference pipeline
│   ├── storage/
│   │   └── store.py                    ← SQLite + FTS
│   ├── api/
│   │   ├── app.py                      ← FastAPI
│   │   └── qa_engine.py                ← LLM-based QA
│   ├── ui/
│   │   └── app.py                      ← Gradio UI
│   └── evaluation/
│       └── evaluator.py                ← Comprehensive evaluation
├── configs/
│   └── config.yaml                     ← All params documented
├── outputs/                            ← Created at runtime
│   ├── checkpoints/final_lora/
│   ├── evaluation/
│   └── analysis/
├── requirements.txt
└── README.md
```
