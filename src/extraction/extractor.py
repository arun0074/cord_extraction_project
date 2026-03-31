"""
=============================================================================
Extraction Pipeline — Inference
=============================================================================
Loads the fine-tuned LoRA model and runs inference on new receipt images.
Returns structured JSON with VENDOR, DATE, TOTAL, RECEIPT_ID.
=============================================================================
"""

import json
import os
import re
import torch
import numpy as np
from typing import Dict, List, Optional
from PIL import Image

from transformers import AutoTokenizer,AutoProcessor

from peft import PeftModel, PeftConfig
from transformers import LayoutLMv3ForTokenClassification

# ── Label schema (must match training) ────────────────────────────────────
LABEL_LIST = [
    "O",
    "B-VENDOR", "I-VENDOR",
    "B-DATE",   "I-DATE",
    "B-TOTAL",  "I-TOTAL",
    "B-RECEIPT_ID", "I-RECEIPT_ID",
]
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}
LABEL2ID = {l: i for i, l in ID2LABEL.items()}


class ReceiptExtractor:
    """
    End-to-end inference pipeline for receipt information extraction.
    
    Inputs:  PIL Image or path to receipt image
    Outputs: {"vendor": str, "date": str, "total": str, "receipt_id": str,
              "raw_spans": {...}, "confidence": {...}}
    """

    def __init__(
        self,
        model_dir: str = "outputs/checkpoints/final_lora",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {model_dir} on {self.device} …")

        # Load LoRA config to get base model name
        peft_config  = PeftConfig.from_pretrained(model_dir)
        base_model   = LayoutLMv3ForTokenClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            num_labels=len(LABEL_LIST),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        self.model   = PeftModel.from_pretrained(base_model, model_dir)
        self.model    = self.model.to(self.device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            peft_config.base_model_name_or_path,
            apply_ocr=False,   # use built-in Tesseract OCR if no text provided
        )
        # ✅ Load tokenizer from the base model instead
        self.tokenizer = AutoTokenizer.from_pretrained(
            peft_config.base_model_name_or_path   # "microsoft/layoutlmv3-base"
        )
        print("✓ Model loaded successfully.")

    def extract(
        self,
        image: Image.Image,
        words: Optional[List[str]] = None,
        boxes: Optional[List[List[int]]] = None,
    ) -> Dict:
        """
        Extract structured fields from a receipt image.

        Args:
            image:  PIL Image of the receipt
            words:  Optional pre-computed OCR words (skips built-in OCR)
            boxes:  Corresponding bounding boxes [x0, y0, x1, y1] in [0,1000] scale
        """
        # ── Preprocess ────────────────────────────────────────────────────
        if words is not None and boxes is not None:
            encoding = self.processor(
                image, words, boxes=boxes,
                return_tensors="pt",
                truncation=True, max_length=512,
                padding="max_length",
            )
        else:
            # Let the processor run OCR automatically.
            encoding = self.processor(
                image,
                return_tensors="pt",
                truncation=True, max_length=512,
                padding="max_length",
            )
            # FIX (Bug 3): recover the real OCR words from the processor's
            # internal tokenizer, NOT subword tokens from convert_ids_to_tokens.
            # LayoutLMv3Processor stores the pre-tokenisation word list here:
            words = self.processor.tokenizer.batch_decode(
                encoding["input_ids"][0].tolist(),
                skip_special_tokens=True,
            )
            # A more reliable source when apply_ocr=True is the processor's
            # last OCR result (available as processor.tokenizer.word_ids won't
            # work, but the words are accessible via the encoding itself):
            # We pull word_ids BEFORE moving tensors to device (see Bug 1 fix).

        # FIX (Bug 1 & 2): call word_ids() on the *encoding object* right here,
        # before converting it to a plain tensor dict. This is the only place
        # where the BatchEncoding object still has the .word_ids() method.
        try:
            word_ids = encoding.word_ids(batch_index=0)  # List[int | None]
        except Exception:
            # Fallback: if the processor doesn't expose word_ids (e.g. slow
            # tokenizer), build a dummy 1-to-1 mapping so inference still runs.
            seq_len  = encoding["input_ids"].shape[1]
            word_ids = list(range(seq_len))

        # Also recover the true word list from the encoding when using auto-OCR,
        # because batch_decode above gives us subword noise.
        if words is None or not any(words):
            # words will have been set in the else-branch above; re-derive
            # using word_ids to pull the first token per word and decode.
            n_words  = max((w for w in word_ids if w is not None), default=-1) + 1
            word_ids_arr = np.array(
                [w if w is not None else -1 for w in word_ids]
            )
            words = []
            for wid in range(n_words):
                tok_positions = np.where(word_ids_arr == wid)[0]
                if len(tok_positions):
                    first_tok = encoding["input_ids"][0, tok_positions[0]].item()
                    words.append(
                        self.processor.tokenizer.decode([first_tok],
                                                        skip_special_tokens=True)
                    )

        # ── Move tensors to device ────────────────────────────────────────
        encoding_tensors = {
            k: v.to(self.device)
            for k, v in encoding.items()
            if isinstance(v, torch.Tensor)
        }

        # ── Inference ─────────────────────────────────────────────────────
        with torch.no_grad():
            outputs = self.model(**encoding_tensors)

        logits      = outputs.logits.squeeze(0)          # [seq_len, num_labels]
        probs       = torch.softmax(logits, dim=-1)
        pred_ids    = logits.argmax(-1).tolist()
        pred_labels = [ID2LABEL[i] for i in pred_ids]
        confidence  = probs.max(-1).values.tolist()

        # ── Map subword predictions → word level (take first subword) ─────
        word_pred_labels: Dict[int, str]   = {}
        word_confidence:  Dict[int, float] = {}

        for tok_idx, (label, conf) in enumerate(zip(pred_labels, confidence)):
            if tok_idx < len(word_ids) and word_ids[tok_idx] is not None:
                wid = word_ids[tok_idx]
                if wid not in word_pred_labels:      # first subword only
                    word_pred_labels[wid] = label
                    word_confidence[wid]  = conf

        # ── Build word-level label / confidence sequences ─────────────────
        word_label_seq = [
            word_pred_labels.get(i, "O") for i in range(len(words))
        ]
        word_conf_seq = [
            word_confidence.get(i, 0.0) for i in range(len(words))
        ]

        raw_spans = self._extract_spans_with_confidence(
            word_label_seq, words, word_conf_seq
        )

        # ── Build final output ────────────────────────────────────────────
        result = {
            "vendor":     self._best_span(raw_spans, "VENDOR"),
            "date":       self._best_span(raw_spans, "DATE"),
            "total":      self._best_span(raw_spans, "TOTAL"),
            "receipt_id": self._best_span(raw_spans, "RECEIPT_ID"),
            "raw_spans":  {k: [s["text"] for s in v]
                           for k, v in raw_spans.items()},
            "confidence": {k: round(max(s["confidence"] for s in v), 3)
                           for k, v in raw_spans.items() if v},
            "all_words":  words[:50],  # preview only
        }
        return result

    # ── Helpers ───────────────────────────────────────────────────────────

    def _extract_spans_with_confidence(
        self,
        labels: List[str],
        words:  List[str],
        confs:  List[float],
    ) -> Dict:
        spans: Dict[str, list] = {}
        current_type:  Optional[str] = None
        current_words: List[str]     = []
        current_confs: List[float]   = []

        def _flush():
            if current_type:
                spans.setdefault(current_type, []).append({
                    "text":       " ".join(current_words),
                    "confidence": float(np.mean(current_confs)),
                })

        for label, word, conf in zip(labels, words, confs):
            if label == "O":
                _flush()
                current_type = None; current_words = []; current_confs = []

            elif label.startswith("B-"):
                _flush()
                current_type  = label[2:]
                current_words = [word]
                current_confs = [conf]

            elif label.startswith("I-") and current_type == label[2:]:
                current_words.append(word)
                current_confs.append(conf)

            else:
                # I- tag with no matching B- (broken sequence) — start fresh
                _flush()
                current_type  = label[2:] if "-" in label else None
                current_words = [word] if current_type else []
                current_confs = [conf]  if current_type else []

        _flush()
        return spans

    def _best_span(self, spans: Dict, field: str) -> str:
        """Return the highest-confidence extracted span for a field."""
        candidates = spans.get(field, [])
        if not candidates:
            return ""
        return max(candidates, key=lambda x: x["confidence"])["text"]

    def extract_batch(self, images: List[Image.Image]) -> List[Dict]:
        return [self.extract(img) for img in images]
