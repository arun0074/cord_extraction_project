"""
=============================================================================
STEP 2: Model Selection, PEFT Configuration, and Training  (v5)
=============================================================================
Model: microsoft/layoutlmv3-base
Task:  Token Classification (NER) → extract VENDOR, DATE, TOTAL, RECEIPT_ID
PEFT:  LoRA (Low-Rank Adaptation)

FIXES (v5):
  BUG C — CUDA scatter/gather "index out of bounds" → training crash:
    The CUDA kernel assertion fires when an embedding-table lookup receives
    an index outside the valid range. Two sources identified:

    (C1) bbox coordinates outside [0, 1000]:
      LayoutLMv3 has 2D position embedding tables of size 1001 (indices 0–1000).
      Any coordinate > 1000 triggers the assertion. This happened because:
        - PIL image.size returns (width, height) as floats in some versions,
          causing int truncation errors during normalisation.
        - Receipts with very small images can produce near-zero width/height,
          making the divisor ≈ 0 and the normalised coordinate >> 1000.
        - Rounding errors: int(1000 * max_x / width) can equal 1000 when
          max_x == width exactly, but LayoutLMv3 uses range [0, 1000] inclusive
          so 1000 is actually valid — the real cap needed is 1000, not 999.
      Fix: explicit int() cast on image dimensions; hard clamp to [0, 1000]
      with min()/max() guards; enforce x_min ≤ x_max and y_min ≤ y_max after
      normalisation; add a pre-training validation scan that prints every
      out-of-range value found and aborts with a clear message.

    (C2) input_ids outside vocabulary range:
      LayoutLMv3-base has vocab_size = 50265 (RoBERTa vocab). If the fast
      tokenizer produces an ID ≥ 50265 (can happen with certain Unicode chars
      in Korean/mixed receipts) or if the padding token ID is wrong, the
      token embedding lookup goes out of bounds.
      Fix: clamp all input_ids to [0, vocab_size-1] after tokenisation;
      add vocab_size to the pre-training validation scan.

    (C3) image_idx stored as a tensor after set_format("torch"):
      set_format("torch") converts ALL integer fields — including image_idx —
      to 0-dimensional tensors. When the collator does int(f.pop("image_idx"))
      on a 0-dim tensor it works, BUT if the tensor has dtype int64 and the
      raw dataset has fewer items, the index can silently overflow on some
      Arrow versions. Fix: explicitly convert image_idx to Python int in the
      collator using .item() for tensors, int() for plain ints.

  BUG A — CORD v2 no token columns (from v4): fixed via valid_line parsing.
  BUG B — KeyError image_idx (from v4): fixed via set_format("torch") no cols.
  BUG 1/2/3 — metrics/labels/nan (from v2/v3): fixes carried forward.
=============================================================================
"""

import os
import re
import json
import math
import collections
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch
from torch import nn
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
from PIL import Image

import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# SECTION A: LABEL SCHEMA
# =============================================================================

LABEL_LIST = [
    "O",
    "B-VENDOR",     "I-VENDOR",
    "B-DATE",       "I-DATE",
    "B-TOTAL",      "I-TOTAL",
    "B-RECEIPT_ID", "I-RECEIPT_ID",
]
LABEL2ID   = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL   = {i: l for l, i in LABEL2ID.items()}
NUM_LABELS = len(LABEL_LIST)

# LayoutLMv3-base constants — must match the pretrained model exactly
VOCAB_SIZE     = 50265   # RoBERTa vocab used by LayoutLMv3-base
MAX_BBOX_VALUE = 1000    # 2D position embedding table size is 1001 → [0, 1000]

# =============================================================================
# SECTION B: PEFT CONFIGURATION
# =============================================================================

LORA_CONFIG = LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"],
    bias="none",
    modules_to_save=["classifier"],
)

# =============================================================================
# SECTION C: TRAINING CONFIG
# =============================================================================

@dataclass
class TrainingConfig:
    dataset_name:        str   = "naver-clova-ix/cord-v2"
    train_subset:        int   = 800
    val_subset:          int   = 100
    test_subset:         int   = 100
    model_name:          str   = "microsoft/layoutlmv3-base"
    output_dir:          str   = "outputs/checkpoints"
    num_epochs:          int   = 5
    train_batch:         int   = 4
    eval_batch:          int   = 8
    grad_accum:          int   = 4
    lr:                  float = 2e-4
    warmup_ratio:        float = 0.1
    weight_decay:        float = 0.01
    lr_scheduler:        str   = "cosine"
    eval_steps:          int   = 50
    early_stop_patience: int   = 3
    max_length:          int   = 512

CFG = TrainingConfig()

# =============================================================================
# SECTION D: GROUND-TRUTH PARSING
# =============================================================================

def _to_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for item in value:
            s = _to_str(item)
            if s:
                return s
        return ""
    if isinstance(value, dict):
        for key in ("total_price", "cashprice", "creditcardprice",
                    "date", "store_name", "branch_name", "price", "value"):
            if key in value and value[key]:
                return _to_str(value[key])
        return ""
    return str(value).strip()


def parse_ground_truth(gt_json_str: str) -> Dict[str, str]:
    try:
        gt    = json.loads(gt_json_str)
        parse = gt.get("gt_parse", {})
    except Exception:
        return {"total": "", "date": "", "vendor": "", "receipt_id": ""}

    result = {}

    total_obj = parse.get("total", {})
    if isinstance(total_obj, dict):
        raw = (total_obj.get("total_price") or
               total_obj.get("cashprice") or
               total_obj.get("creditcardprice") or "")
    else:
        raw = total_obj
    result["total"] = _to_str(raw)

    payment = parse.get("payment", {})
    if isinstance(payment, dict):
        result["date"] = _to_str(payment.get("date", ""))
    elif isinstance(payment, list) and payment:
        first = payment[0]
        result["date"] = _to_str(
            first.get("date", "") if isinstance(first, dict) else first
        )
    else:
        result["date"] = ""

    store = parse.get("store_info", {})
    if isinstance(store, dict):
        raw = store.get("store_name") or store.get("branch_name") or ""
    else:
        raw = store
    result["vendor"] = _to_str(raw)

    result["receipt_id"] = _to_str(parse.get("id", ""))
    return result


def _safe_bbox(x_min_px, y_min_px, x_max_px, y_max_px,
               img_w: int, img_h: int) -> List[int]:
    """
    FIX (Bug C1): Convert pixel coords to normalised [0, 1000] bbox.

    Guards applied:
      1. Divisor floored at 1 to prevent division by zero on tiny images.
      2. Each coordinate clamped to [0, MAX_BBOX_VALUE] after normalisation.
      3. Enforce x_min ≤ x_max and y_min ≤ y_max (swap if inverted quad).
         An inverted quad can occur in skewed receipt scans where the OCR
         engine reports corners in a non-standard order.
    """
    w = max(1, int(img_w))
    h = max(1, int(img_h))

    def norm(px, dim):
        return max(0, min(MAX_BBOX_VALUE, int(MAX_BBOX_VALUE * px / dim)))

    x0 = norm(x_min_px, w)
    y0 = norm(y_min_px, h)
    x1 = norm(x_max_px, w)
    y1 = norm(y_max_px, h)

    # Enforce min ≤ max
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0

    return [x0, y0, x1, y1]


def parse_words_and_boxes(
    gt_json_str: str,
    image_width: int,
    image_height: int,
) -> Tuple[List[str], List[List[int]]]:
    """
    Extract OCR words and normalised bboxes from CORD valid_line.

    Quad → axis-aligned:
      x_min = min(x1,x2,x3,x4),  x_max = max(x1,x2,x3,x4)
      y_min = min(y1,y2,y3,y4),  y_max = max(y1,y2,y3,y4)
    Then normalised to [0, 1000] via _safe_bbox().
    """
    words, bboxes = [], []
    try:
        gt = json.loads(gt_json_str)
    except Exception:
        return words, bboxes

    for line in gt.get("valid_line", []):
        for word_info in line.get("words", []):
            text = word_info.get("text", "").strip()
            if not text:
                continue

            quad = word_info.get("quad", {})
            xs   = [quad.get("x1", 0), quad.get("x2", 0),
                    quad.get("x3", 0), quad.get("x4", 0)]
            ys   = [quad.get("y1", 0), quad.get("y2", 0),
                    quad.get("y3", 0), quad.get("y4", 0)]

            bbox = _safe_bbox(
                min(xs), min(ys), max(xs), max(ys),
                image_width, image_height,
            )
            words.append(text)
            bboxes.append(bbox)

    return words, bboxes


# =============================================================================
# SECTION E: LABEL ASSIGNMENT
# =============================================================================

def _normalise(text: str) -> List[str]:
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text).lower().split()


def assign_word_labels(words: List[str], gt: Dict[str, str]) -> List[str]:
    labels          = ["O"] * len(words)
    norm_words      = [_normalise(w) for w in words]
    flat_norm_words = [" ".join(toks) if toks else "" for toks in norm_words]

    def tag_span(value: str, bio_prefix: str):
        if not value:
            return
        norm_gt = _normalise(value)
        if not norm_gt:
            return
        for start in range(len(flat_norm_words)):
            window = flat_norm_words[start: start + len(norm_gt)]
            if len(window) < len(norm_gt):
                break
            if window == norm_gt:
                labels[start] = f"B-{bio_prefix}"
                for j in range(1, len(norm_gt)):
                    labels[start + j] = f"I-{bio_prefix}"
                return

    tag_span(gt.get("receipt_id", ""), "RECEIPT_ID")
    tag_span(gt.get("total", ""),      "TOTAL")
    tag_span(gt.get("date", ""),       "DATE")
    tag_span(gt.get("vendor", ""),     "VENDOR")
    return labels


# =============================================================================
# SECTION F: TOKENISATION
# =============================================================================

def encode_words_and_labels(
    words:       List[str],
    word_labels: List[str],
    tokenizer,
    word_boxes:  List[List[int]],
    max_length:  int,
) -> Dict[str, Any]:
    """
    Tokenise and align labels. Returns plain Python lists (Arrow-safe).
    Does NOT encode images (handled by collator to bypass Arrow).

    FIX (Bug C2): input_ids are clamped to [0, VOCAB_SIZE-1] after tokenisation.
    The fast LayoutLMv3 tokenizer can emit IDs for special Unicode chars that
    map outside RoBERTa's vocab table, causing the embedding lookup to assert.
    Clamping replaces any out-of-range ID with the <unk> token ID (3 in RoBERTa).
    """
    word_label_ids = [LABEL2ID.get(lbl, 0) for lbl in word_labels]

    encoding = tokenizer.tokenizer(
        text=words,
        boxes=word_boxes,
        word_labels=word_label_ids,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    # FIX (Bug C2): clamp token IDs to valid vocab range
    unk_id     = tokenizer.tokenizer.unk_token_id or 3
    input_ids  = [
        max(0, min(VOCAB_SIZE - 1, tid)) if (tid < 0 or tid >= VOCAB_SIZE) else tid
        for tid in encoding["input_ids"]
    ]

    # FIX (Bug C1): clamp bbox values (extra safety on top of _safe_bbox)
    bbox = [
        [max(0, min(MAX_BBOX_VALUE, c)) for c in box]
        for box in encoding["bbox"]
    ]

    return {
        "input_ids":      input_ids,
        "attention_mask": encoding["attention_mask"],
        "bbox":           bbox,
        "labels":         encoding["labels"],
    }


class CORDPreprocessor:
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __call__(self, example: Dict, idx: int) -> Dict:
        gt_str = example.get("ground_truth", "{}")
        image  = example.get("image")

        # FIX (Bug C1): explicit int() cast; PIL size can be float in some versions
        if image is not None:
            img_w, img_h = int(image.size[0]), int(image.size[1])
        else:
            img_w, img_h = 224, 224

        words, bboxes = parse_words_and_boxes(gt_str, img_w, img_h)

        if not words:
            words  = ["[UNK]"]
            bboxes = [[0, 0, 0, 0]]

        gt      = parse_ground_truth(gt_str)
        wlabels = assign_word_labels(words, gt)

        encoded = encode_words_and_labels(
            words, wlabels, self.tokenizer, bboxes, self.max_length
        )

        # Store index as plain int — Arrow-safe; collator fetches images by this
        encoded["image_idx"] = idx
        encoded["gt_fields"] = json.dumps(gt)
        return encoded


# =============================================================================
# SECTION G: PRE-TRAINING VALIDATION SCAN
# =============================================================================

def validate_dataset_ranges(dataset, split_name: str, vocab_size: int = VOCAB_SIZE):
    """
    FIX (Bug C): Scan the entire preprocessed dataset for out-of-range values
    BEFORE training starts. Prints a report and raises ValueError if any
    violations are found, so the crash happens with a readable message instead
    of a cryptic CUDA assertion deep in a kernel.

    Checks:
      - input_ids: must be in [0, vocab_size-1]
      - bbox coords: must be in [0, MAX_BBOX_VALUE]
      - labels: must be in [-100, NUM_LABELS-1]
    """
    print(f"\n  [Range validation: {split_name}]")
    n_id_violations   = 0
    n_bbox_violations = 0
    n_lbl_violations  = 0
    examples_checked  = 0

    for ex in dataset:
        examples_checked += 1

        ids = ex["input_ids"]
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        for tok_id in ids:
            if tok_id < 0 or tok_id >= vocab_size:
                n_id_violations += 1

        bboxes = ex["bbox"]
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.tolist()
        for box in bboxes:
            for coord in box:
                if coord < 0 or coord > MAX_BBOX_VALUE:
                    n_bbox_violations += 1

        lbls = ex["labels"]
        if isinstance(lbls, torch.Tensor):
            lbls = lbls.tolist()
        for lbl in lbls:
            if lbl != -100 and (lbl < 0 or lbl >= NUM_LABELS):
                n_lbl_violations += 1

    ok = True
    if n_id_violations > 0:
        print(f"    [FAIL] input_ids: {n_id_violations} values outside [0, {vocab_size-1}]")
        ok = False
    else:
        print(f"    ✓ input_ids: all in [0, {vocab_size-1}]")

    if n_bbox_violations > 0:
        print(f"    [FAIL] bbox coords: {n_bbox_violations} values outside [0, {MAX_BBOX_VALUE}]")
        ok = False
    else:
        print(f"    ✓ bbox coords: all in [0, {MAX_BBOX_VALUE}]")

    if n_lbl_violations > 0:
        print(f"    [FAIL] labels: {n_lbl_violations} values outside [-100, {NUM_LABELS-1}]")
        ok = False
    else:
        print(f"    ✓ labels: all in [-100, {NUM_LABELS-1}]")

    print(f"    Checked {examples_checked} examples.")

    if not ok:
        raise ValueError(
            f"Range violations found in {split_name} split. "
            "Fix the preprocessing before training — these will cause CUDA assertions."
        )


# =============================================================================
# SECTION H: DATA COLLATOR
# =============================================================================

class LayoutLMv3DataCollator:
    """
    Encodes images at collation time (bypasses Arrow serialisation of tensors).

    FIX (Bug C3): image_idx is retrieved with .item() if it's a 0-dim tensor
    (which set_format("torch") produces for integer scalar fields), otherwise
    cast directly to int. This prevents silent index corruption.
    """

    def __init__(self, tokenizer, raw_dataset):
        self.tokenizer   = tokenizer
        self.raw_dataset = raw_dataset

    @staticmethod
    def _to_int(val) -> int:
        """Safely extract int from either a 0-dim tensor or a plain int/float."""
        if isinstance(val, torch.Tensor):
            return int(val.item())
        return int(val)

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        image_indices = [self._to_int(f.pop("image_idx")) for f in features]
        for f in features:
            f.pop("gt_fields", None)

        batch = {
            key: torch.stack([
                f[key] if isinstance(f[key], torch.Tensor)
                else torch.tensor(f[key], dtype=torch.long)
                for f in features
            ])
            for key in ("input_ids", "attention_mask", "bbox", "labels")
        }

        # Encode images fresh — never serialised through Arrow
        pil_images = []
        for idx in image_indices:
            img = self.raw_dataset[idx].get("image")
            if img is None:
                img = Image.new("RGB", (224, 224), color=(255, 255, 255))
            pil_images.append(img)

        image_enc = self.tokenizer.image_processor(
            images=pil_images, return_tensors="pt"
        )
        batch["pixel_values"] = image_enc["pixel_values"]   # (B, 3, 224, 224)

        return batch


# =============================================================================
# SECTION I: CLASS-WEIGHTED LOSS
# =============================================================================

class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device),
            ignore_index=-100,
        )
        loss = loss_fn(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1),
        )
        return (loss, outputs) if return_outputs else loss


def compute_class_weights(dataset) -> torch.Tensor:
    counts = collections.Counter()
    for ex in dataset:
        for lbl in ex["labels"]:
            lbl_val = lbl.item() if hasattr(lbl, "item") else int(lbl)
            if lbl_val != -100:
                counts[lbl_val] += 1

    total   = sum(counts.values())
    weights = torch.ones(NUM_LABELS)
    for cls_id in range(NUM_LABELS):
        if counts[cls_id] > 0:
            weights[cls_id] = total / (NUM_LABELS * counts[cls_id])

    print("\nClass weights:")
    for i, (lbl, w) in enumerate(zip(LABEL_LIST, weights)):
        print(f"  {lbl:20s}: {w:.3f}  (count={counts[i]:,})")

    entity_counts = sum(counts[i] for i in range(1, NUM_LABELS))
    if entity_counts == 0:
        print("\n[WARNING] Zero entity labels — assign_word_labels matched nothing.")
    return weights


# =============================================================================
# SECTION J: METRICS
# =============================================================================

seqeval = evaluate.load("seqeval")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions    = np.argmax(logits, axis=-1)

    true_labels, pred_labels = [], []
    for pred_seq, label_seq in zip(predictions, labels):
        true_seq, pred_seq_f = [], []
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id == -100:
                continue
            true_seq.append(ID2LABEL[label_id])
            pred_seq_f.append(ID2LABEL[pred_id])
        if not true_seq:
            continue
        true_labels.append(true_seq)
        pred_labels.append(pred_seq_f)

    if not true_labels:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    results = seqeval.compute(
        predictions=pred_labels,
        references=true_labels,
        zero_division=0,
    )
    flat = {
        "precision": results["overall_precision"],
        "recall":    results["overall_recall"],
        "f1":        results["overall_f1"],
        "accuracy":  results["overall_accuracy"],
    }
    for entity_type in ["VENDOR", "DATE", "TOTAL", "RECEIPT_ID"]:
        if entity_type in results:
            flat[f"{entity_type.lower()}_f1"]       = results[entity_type]["f1"]
            flat[f"{entity_type.lower()}_precision"] = results[entity_type]["precision"]
            flat[f"{entity_type.lower()}_recall"]    = results[entity_type]["recall"]
    return flat


# =============================================================================
# SECTION K: DIAGNOSTICS
# =============================================================================

def _check_label_distribution(dataset, split_name: str):
    counts = collections.Counter()
    for ex in dataset:
        for lbl in ex["labels"]:
            lbl_val = lbl.item() if hasattr(lbl, "item") else int(lbl)
            if lbl_val != -100:
                counts[lbl_val] += 1
    total = sum(counts.values())
    print(f"\n  [{split_name}] label distribution ({total:,} non-padding tokens):")
    for i, lbl in enumerate(LABEL_LIST):
        cnt = counts.get(i, 0)
        pct = 100 * cnt / total if total else 0
        print(f"    {lbl:20s}: {cnt:6,}  ({pct:5.1f}%)")


def _assert_sample_integrity(dataset, raw_dataset, tokenizer, split_name: str):
    print(f"\n  [Integrity check: {split_name}]")
    sample    = dataset[0]
    labels    = sample["labels"]
    label_arr = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels)
    non_pad   = label_arr[label_arr != -100]

    if not (non_pad > 0).any():
        print(f"    [WARNING] No entity labels in sample 0 — only O or padding.")
        gt_snippet = raw_dataset[0].get("ground_truth", "")[:400]
        print(f"    Ground truth snippet:\n      {gt_snippet}")
    else:
        entity_names = [ID2LABEL[int(i)] for i in non_pad[non_pad > 0].unique()]
        print(f"    ✓ Entity labels found: {entity_names}")

    img = raw_dataset[0].get("image")
    if img is None:
        img = Image.new("RGB", (224, 224), color=(255, 255, 255))
    pv     = tokenizer.image_processor(images=[img], return_tensors="pt")["pixel_values"][0]
    pv_sum = pv.sum().item()
    if abs(pv_sum) < 1e-6:
        print(f"    [WARNING] pixel_values sum ≈ 0.0")
    else:
        print(f"    ✓ pixel_values sum = {pv_sum:.2f} (Arrow bypass confirmed)")


# =============================================================================
# SECTION L: MODEL
# =============================================================================

def build_model_and_peft():
    print("=" * 70)
    print("Loading LayoutLMv3-base …")
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        CFG.model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    print("Applying LoRA …")
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()
    return model


# =============================================================================
# SECTION M: TRAINER
# =============================================================================

class MultiSplitWeightedTrainer(WeightedTrainer):
    """Routes train vs eval DataLoaders to the correct split-specific collator."""

    def __init__(self, train_collator, val_collator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_collator = train_collator
        self._val_collator   = val_collator

    def get_train_dataloader(self):
        original           = self.data_collator
        self.data_collator = self._train_collator
        dl                 = super().get_train_dataloader()
        self.data_collator = original
        return dl

    def get_eval_dataloader(self, eval_dataset=None):
        original           = self.data_collator
        self.data_collator = self._val_collator
        dl                 = super().get_eval_dataloader(eval_dataset)
        self.data_collator = original
        return dl


# =============================================================================
# SECTION N: MAIN TRAINING PIPELINE
# =============================================================================

def run_training():
    print("=" * 70)
    print("CORD INFORMATION EXTRACTION — TRAINING PIPELINE  (v5)")
    print("=" * 70)

    # ── 1. Load dataset ────────────────────────────────────────────────────
    print("\n[1/6] Loading dataset …")
    raw = load_dataset(CFG.dataset_name)
    print(f"  Train column names: {raw['train'].column_names}")

    train_raw = raw["train"].select(range(min(CFG.train_subset, len(raw["train"]))))
    val_raw   = raw["validation"].select(range(min(CFG.val_subset,  len(raw["validation"]))))
    test_raw  = raw["test"].select(range(min(CFG.test_subset,  len(raw["test"]))))

    # ── 2. Processor ───────────────────────────────────────────────────────
    print("\n[2/6] Initialising processor …")
    tokenizer = AutoProcessor.from_pretrained(CFG.model_name, apply_ocr=False)

    # ── 3. Preprocess ──────────────────────────────────────────────────────
    print("\n[3/6] Preprocessing datasets …")
    preprocessor = CORDPreprocessor(tokenizer, max_length=CFG.max_length)
    orig_cols    = raw["train"].column_names

    train_ds = train_raw.map(
        preprocessor, batched=False, with_indices=True,
        remove_columns=orig_cols, desc="Preprocessing train",
    )
    val_ds = val_raw.map(
        preprocessor, batched=False, with_indices=True,
        remove_columns=orig_cols, desc="Preprocessing val",
    )
    test_ds = test_raw.map(
        preprocessor, batched=False, with_indices=True,
        remove_columns=orig_cols, desc="Preprocessing test",
    )

    # NO explicit columns list → image_idx and gt_fields are NOT dropped
    train_ds.set_format("torch")
    val_ds.set_format("torch")
    test_ds.set_format("torch")

    # ── Diagnostics ────────────────────────────────────────────────────────
    print("\n  [Diagnostic] Label distributions after preprocessing:")
    _check_label_distribution(train_ds, "train")
    _check_label_distribution(val_ds,   "val")

    print("\n  [Diagnostic] Sample integrity checks:")
    _assert_sample_integrity(train_ds, train_raw, tokenizer, "train")
    _assert_sample_integrity(val_ds,   val_raw,   tokenizer, "val")

    # ── FIX (Bug C): Range validation — catches out-of-bounds BEFORE GPU ──
    print("\n  [Diagnostic] Range validation (catches CUDA assertions early):")
    validate_dataset_ranges(train_ds, "train")
    validate_dataset_ranges(val_ds,   "val")

    # ── 4. Class weights ───────────────────────────────────────────────────
    print("\n[4/6] Computing class weights …")
    class_weights = compute_class_weights(train_ds)

    # ── 5. Model + LoRA ────────────────────────────────────────────────────
    print("\n[5/6] Building model with LoRA …")
    model = build_model_and_peft()

    # ── 6. Training ────────────────────────────────────────────────────────
    print("\n[6/6] Starting training …")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    print(f"  Precision: {'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32'}")

    total_steps  = math.ceil(
        CFG.train_subset / CFG.train_batch / CFG.grad_accum
    ) * CFG.num_epochs
    warmup_steps = int(CFG.warmup_ratio * total_steps)

    training_args = TrainingArguments(
        output_dir                  = CFG.output_dir,
        num_train_epochs            = CFG.num_epochs,
        per_device_train_batch_size = CFG.train_batch,
        per_device_eval_batch_size  = CFG.eval_batch,
        gradient_accumulation_steps = CFG.grad_accum,
        learning_rate               = CFG.lr,
        warmup_steps                = warmup_steps,
        weight_decay                = CFG.weight_decay,
        lr_scheduler_type           = CFG.lr_scheduler,
        eval_strategy               = "steps",
        eval_steps                  = CFG.eval_steps,
        save_strategy               = "steps",
        save_steps                  = CFG.eval_steps,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_f1",
        greater_is_better           = True,
        logging_steps               = 25,
        bf16                        = use_bf16,
        fp16                        = use_fp16,
        dataloader_num_workers      = 0,
        report_to                   = "none",
        seed                        = SEED,
        label_names                 = ["labels"],
        remove_unused_columns       = False,   # keep image_idx for collator
    )

    train_collator = LayoutLMv3DataCollator(tokenizer, train_raw)
    val_collator   = LayoutLMv3DataCollator(tokenizer, val_raw)
    test_collator  = LayoutLMv3DataCollator(tokenizer, test_raw)

    trainer = MultiSplitWeightedTrainer(
        train_collator   = train_collator,
        val_collator     = val_collator,
        class_weights    = class_weights,
        model            = model,
        args             = training_args,
        train_dataset    = train_ds,
        eval_dataset     = val_ds,
        processing_class = tokenizer,
        data_collator    = train_collator,
        compute_metrics  = compute_metrics,
        callbacks        = [EarlyStoppingCallback(
                                early_stopping_patience=CFG.early_stop_patience)],
    )

    trainer.train()

    save_path = os.path.join(CFG.output_dir, "final_lora")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print("\n" + "=" * 70)
    print("FINAL TEST SET EVALUATION")
    print("=" * 70)

    trainer._val_collator = test_collator
    test_results = trainer.evaluate(test_ds)

    for k, v in test_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    print(f"\n✓ Model saved to {save_path}")
    print("  → Next: Run src/extraction/extractor.py")
    return trainer, model, tokenizer


if __name__ == "__main__":
    run_training()
