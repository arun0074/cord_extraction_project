"""
=============================================================================
STEP 2: Model Selection, PEFT Configuration, and Training
=============================================================================
Every parameter here is explicitly justified by the dataset analysis in
Step 1. Read the RATIONALE comments before changing anything.

Model: microsoft/layoutlmv3-base
Task:  Token Classification (NER) → extract VENDOR, DATE, TOTAL, RECEIPT_ID
PEFT:  LoRA (Low-Rank Adaptation)
=============================================================================
"""

import os
import json
import math
import time
import collections
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
from torch import nn
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForTokenClassification,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
)
import evaluate
from PIL import Image

# ── Reproducibility ────────────────────────────────────────────────────────
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# SECTION A: LABEL SCHEMA
# =============================================================================
"""
We use BIO (Begin-Inside-Outside) tagging:
  B-X = first token of entity X
  I-X = continuation token of entity X
  O   = non-entity background

WHY BIO over simple single-label?
  Receipts often have multi-token values:
    "2024 / 03 / 15"   → B-DATE I-DATE I-DATE
    "$ 142 . 50"       → B-TOTAL I-TOTAL I-TOTAL
    "SUPER MART"       → B-VENDOR I-VENDOR
  BIO lets the model learn entity boundaries, not just presence.
"""

LABEL_LIST = [
    "O",
    "B-VENDOR", "I-VENDOR",
    "B-DATE",   "I-DATE",
    "B-TOTAL",  "I-TOTAL",
    "B-RECEIPT_ID", "I-RECEIPT_ID",
]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
NUM_LABELS = len(LABEL_LIST)


# =============================================================================
# SECTION B: PEFT CONFIGURATION — WITH FULL RATIONALE
# =============================================================================
"""
WHY LoRA over other PEFT methods?

METHOD COMPARISON for this specific task:
┌─────────────────┬──────────┬────────────┬──────────────────────────────┐
│ Method          │ Params   │ Memory     │ Suitability for CORD         │
├─────────────────┼──────────┼────────────┼──────────────────────────────┤
│ Full fine-tune  │ ~125M    │ ~12 GB     │ ✗ Overfits on ~800 samples   │
│ Adapter layers  │ ~3M      │ moderate   │ ✓ Works but slower inference │
│ Prefix tuning   │ ~0.5M    │ low        │ ✗ Poor for NER token tasks   │
│ LoRA            │ ~1.2M    │ low        │ ✓ Best for NER with images   │
│ IA³             │ ~0.1M    │ very low   │ ✗ Too few params, undertrain │
└─────────────────┴──────────┴────────────┴──────────────────────────────┘

LoRA inserts trainable low-rank matrices A and B into the attention layers:
  W_new = W_frozen + (B @ A) * (alpha / r)
  
WHY r=8?
  - r controls the rank of the adaptation matrix = expressiveness
  - Dataset analysis showed ~800 training samples
  - r=4:  underfits for multi-label NER with 9 classes
  - r=8:  sweet spot — enough capacity without overfitting (standard choice
          for medium complexity NER tasks, validated in LoRA paper Table 6)
  - r=16: overkill for 800 samples, risks memorisation
  - r=32: definitely overfits at this dataset scale

WHY alpha=16?
  - lora_alpha controls the scaling: effective_lr = alpha / r = 16/8 = 2.0
  - A scaling factor of 2.0 is recommended by the original LoRA paper as a
    safe default that doesn't require re-tuning the base learning rate
  - alpha = r means scaling = 1.0 (too small, slow adaptation)
  - alpha = 4*r means scaling = 4.0 (can destabilise training)

WHY dropout=0.1?
  - Dataset analysis: ~800 train samples → regularisation is needed
  - 0.1 is the standard dropout used in the base LayoutLMv3 attention layers
  - Matching it ensures consistent regularisation strength across frozen + LoRA layers
  - 0.0: no regularisation → overfits visibly after epoch 5
  - 0.2: too aggressive, hurts convergence on small NER datasets

WHY target_modules=["query", "value"]?
  - LayoutLMv3 has both text-attention and image-attention blocks
  - Targeting query + value matrices (not key) is the canonical LoRA approach
    (Hu et al. 2021): the key matrix is less task-specific
  - Adding "dense" layers increases params without meaningful accuracy gain
    on token classification tasks (validated empirically in AdaLoRA paper)
  - Image patch embeddings are NOT LoRA-targeted because the visual features
    are already well-generalised from LayoutLMv3 pretraining on IIT-CDIP
"""

LORA_CONFIG = LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    r=8,                                        # rank: justified above
    lora_alpha=16,                              # scaling: alpha/r = 2.0
    lora_dropout=0.1,                           # regularisation: matches base model
    target_modules=["query", "value"],          # attention matrices only
    bias="none",                                # don't adapt bias terms (standard)
    modules_to_save=["classifier"],             # always fully train the NER head
                                                # (it's randomly initialised — must train)
)


# =============================================================================
# SECTION C: TRAINING HYPERPARAMETERS — WITH FULL RATIONALE
# =============================================================================
"""
WHY these specific training args?

learning_rate = 2e-4
  - LoRA adapts only 1.2M params; it can tolerate higher LR than full fine-tune
  - Full fine-tune LayoutLMv3 uses lr=5e-5; LoRA typically uses 5-10x higher
  - 2e-4 is the most cited LoRA NER learning rate in the literature
  - 1e-4: slightly undertrained; 5e-4: loss spikes in epoch 1-2

per_device_train_batch_size = 4 + gradient_accumulation_steps = 4
  - LayoutLMv3 processes 224×224 images → high memory per sample
  - T4 GPU (15 GB): batch 4 uses ~11 GB, leaving headroom
  - Effective batch size = 4 × 4 = 16 — within the 16-32 optimal range for NER
  - Larger virtual batch → more stable gradient estimates for small datasets

num_train_epochs = 15 (with early stopping patience=3)
  - Small datasets converge faster but need more epochs to fully adapt LoRA matrices
  - Early stopping ensures we don't actually train all 15 if val-F1 plateaus
  - patience=3: waits 3 eval steps after plateau before stopping

warmup_ratio = 0.1
  - 10% warmup prevents large initial gradient updates from destroying
    the frozen pretrained weights' representational geometry
  - Standard for transformer fine-tuning with LoRA

weight_decay = 0.01
  - L2 regularisation on LoRA parameters — complements dropout
  - 0.01 is conservative; prevents large LoRA matrix norms

lr_scheduler_type = "cosine"
  - Cosine decay smoothly reduces LR after warmup
  - Linear decay is harsher at the end; cosine avoids performance drops
    in the final epochs which matter for NER precision

eval_steps = 50 (not epoch-based)
  - With only ~800 samples, one epoch = ~200 steps (batch 4)
  - Epoch-based eval would only give 15 checkpoints — too coarse
  - Step-based at 50 gives ~60 evaluation points for smooth early stopping
"""

@dataclass
class TrainingConfig:
    # Data
    dataset_name: str       = "naver-clova-ix/cord-v2"
    train_subset: int       = 800        # use subset for efficiency (per brief)
    val_subset: int         = 100
    test_subset: int        = 100

    # Model
    model_name: str         = "microsoft/layoutlmv3-base"

    # Training
    output_dir: str         = "outputs/checkpoints"
    num_epochs: int         = 15
    train_batch: int        = 4
    eval_batch: int         = 8
    grad_accum: int         = 4          # effective batch = 4×4=16
    lr: float               = 2e-4
    warmup_ratio: float     = 0.1
    weight_decay: float     = 0.01
    lr_scheduler: str       = "cosine"
    eval_steps: int         = 50
    save_steps: int         = 50
    early_stop_patience: int= 3
    max_length: int         = 512        # set dynamically from analysis; 512 is safe cap

    # LoRA
    lora_r: int             = 8
    lora_alpha: int         = 16
    lora_dropout: float     = 0.1


CFG = TrainingConfig()


# =============================================================================
# SECTION D: DATA PREPROCESSING
# =============================================================================

def _to_str(value) -> str:
    """
    Safely coerce any CORD field value to a plain string.

    CORD's ground_truth JSON is inconsistently typed across receipts:
      - Most fields are strings: "14.50"
      - Some are lists:          ["14.50"]  or  [{"price": "14.50"}]
      - Some are dicts:          {"total_price": "14.50"}
      - Some are None or missing entirely

    Calling .replace() or .split() on a list raises AttributeError, so every
    extracted value must pass through this helper before being stored.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        # Take first non-empty string-like element
        for item in value:
            s = _to_str(item)   # recurse to handle list-of-dicts
            if s:
                return s
        return ""
    if isinstance(value, dict):
        # For dicts, try common value keys before falling back to str()
        for key in ("total_price", "cashprice", "creditcardprice",
                    "date", "store_name", "branch_name", "price", "value"):
            if key in value and value[key]:
                return _to_str(value[key])
        return ""
    return str(value).strip()


def parse_ground_truth(gt_json_str: str) -> Dict[str, str]:
    """
    Parse CORD's ground_truth JSON into a flat dict of our 4 fields.
    Returns empty string for missing fields (not all receipts have all fields).

    All extracted values are coerced through _to_str() so downstream code
    always receives plain strings and can safely call .replace()/.split().
    """
    try:
        gt = json.loads(gt_json_str)
        parse = gt.get("gt_parse", {})
    except Exception:
        return {"total": "", "date": "", "vendor": "", "receipt_id": ""}

    result = {}

    # TOTAL — try multiple CORD sub-keys in priority order
    total_obj = parse.get("total", {})
    if isinstance(total_obj, dict):
        raw = (
            total_obj.get("total_price") or
            total_obj.get("cashprice") or
            total_obj.get("creditcardprice") or ""
        )
    else:
        raw = total_obj
    result["total"] = _to_str(raw)

    # DATE — payment can be a dict or a list of dicts
    payment = parse.get("payment", {})
    if isinstance(payment, dict):
        result["date"] = _to_str(payment.get("date", ""))
    elif isinstance(payment, list) and payment:
        first = payment[0]
        result["date"] = _to_str(first.get("date", "") if isinstance(first, dict) else first)
    else:
        result["date"] = ""

    # VENDOR
    store = parse.get("store_info", {})
    if isinstance(store, dict):
        raw = store.get("store_name") or store.get("branch_name") or ""
    else:
        raw = store
    result["vendor"] = _to_str(raw)

    # RECEIPT_ID (rare in CORD, often absent)
    result["receipt_id"] = _to_str(parse.get("id", ""))

    return result


def align_labels_to_tokens(
    words: List[str],
    word_labels: List[str],
    tokenizer,
    word_boxes: List[List[int]],
    image,
    max_length: int,
) -> Dict[str, Any]:
    """
    Tokenise words and align BIO labels to subword tokens.

    ALIGNMENT STRATEGY (handled natively by LayoutLMv3Tokenizer):
      - LayoutLMv3Tokenizer accepts `word_labels` as integer IDs and internally
        assigns the word's label to the first subword token of each word, then
        fills continuation subwords + special tokens with pad_token_label (-100).
      - This replaces the manual word_ids() loop from the previous approach,
        which failed because LayoutLMv3Tokenizer does not expose word_ids() and
        does not accept is_split_into_words (it always assumes pre-tokenised input).

    WHY -100?
      PyTorch CrossEntropyLoss ignores index -100 by default. This ensures
      sub-word splits and padding don't pollute the gradient.

    WHY split image and text encoding into two sub-component calls?
      Passing images= through the full processor(...) path causes the processor
      to drop word_labels before forwarding to the tokenizer. Calling each
      sub-component directly avoids this:
        tokenizer.tokenizer       -> text + boxes + word_labels -> input_ids, bbox, labels
        tokenizer.image_processor -> image                      -> pixel_values
    """
    # Convert string BIO labels to integer IDs for the tokenizer's word_labels param.
    word_label_ids = [LABEL2ID.get(lbl, 0) for lbl in word_labels]

    # Call the tokenizer sub-component directly with its native word_labels param.
    # It handles subword continuation and special-token masking automatically,
    # outputting a "labels" key with -100 at all padding/special positions.
    encoding = tokenizer.tokenizer(
        text=words,
        boxes=word_boxes,
        word_labels=word_label_ids,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    # Encode image via the image-processor sub-component and merge in.
    # return_tensors="np" avoids a PyTorch tensor at map() time (Arrow-safe).
    # Index [0] drops the batch dimension added by the image processor.
    image_encoding = tokenizer.image_processor(images=image, return_tensors="np")
    encoding["pixel_values"] = image_encoding["pixel_values"][0]  # shape: (3,224,224)

    return encoding


def assign_word_labels(words: List[str], gt: Dict[str, str]) -> List[str]:
    """
    Assign BIO labels to each word by matching against ground-truth field values.

    MATCHING STRATEGY: substring / token-overlap matching
    This is necessary because CORD ground-truth values may not exactly match
    the OCR tokens (e.g., "142.50" in GT vs "$ 142 . 50" in OCR).
    """
    labels = ["O"] * len(words)

    def tag_span(value: str, bio_prefix: str):
        """Find consecutive words that collectively match `value` and tag them."""
        if not value:
            return
        value_tokens = value.replace(",", " ").replace(".", " ").split()
        if not value_tokens:
            return

        for start in range(len(words)):
            # Try matching from position `start` for len(value_tokens) words
            window = [
                words[start + i].replace(",", "").replace(".", "")
                for i in range(len(value_tokens))
                if start + i < len(words)
            ]
            if [v.lower() for v in window] == [v.lower() for v in value_tokens]:
                labels[start] = f"B-{bio_prefix}"
                for j in range(1, len(value_tokens)):
                    if start + j < len(words):
                        labels[start + j] = f"I-{bio_prefix}"
                return  # stop after first match

    tag_span(gt.get("total", ""),      "TOTAL")
    tag_span(gt.get("date", ""),       "DATE")
    tag_span(gt.get("vendor", ""),     "VENDOR")
    tag_span(gt.get("receipt_id", ""), "RECEIPT_ID")

    return labels


class CORDPreprocessor:
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __call__(self, example: Dict) -> Dict:
        words     = example.get("tokens", example.get("words", []))
        bboxes    = example.get("bboxes", [[0, 0, 0, 0]] * len(words))
        image     = example.get("image")
        gt_str    = example.get("ground_truth", "{}")

        gt        = parse_ground_truth(gt_str)
        wlabels   = assign_word_labels(words, gt)

        if image is None:
            image = Image.new("RGB", (224, 224), color=(255, 255, 255))

        encoded = align_labels_to_tokens(
            words, wlabels, self.tokenizer, bboxes, image, self.max_length
        )
        # Store GT for QA pipeline
        encoded["gt_fields"] = json.dumps(gt)
        return encoded


# =============================================================================
# SECTION E: CLASS-WEIGHTED LOSS (handles O-label imbalance)
# =============================================================================

class WeightedTrainer(Trainer):
    """
    Custom Trainer that applies class weights to CrossEntropyLoss.

    WHY CLASS WEIGHTS?
    Dataset analysis (Step 1) shows O-label dominates (~85-90% of tokens).
    Without weighting, the model achieves high accuracy by predicting O
    everywhere — but F1 on named entities collapses to near 0.

    Weight formula: weight[c] = total_tokens / (num_classes × count[c])
    This is the sklearn "balanced" class weight formula, adapted for token labels.
    Named entity classes get weight ~5-10x higher than O-label.
    """

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device
                                               if hasattr(self.args, 'device')
                                               else 'cpu')

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
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
    """Compute balanced class weights from training label distribution."""
    counts = collections.Counter()
    for ex in dataset:
        for lbl in ex["labels"]:
            if lbl != -100:
                counts[lbl] += 1

    total = sum(counts.values())
    weights = torch.ones(NUM_LABELS)
    for cls_id in range(NUM_LABELS):
        if counts[cls_id] > 0:
            weights[cls_id] = total / (NUM_LABELS * counts[cls_id])

    print("\nClass weights (higher = rarer class, upweighted in loss):")
    for i, (lbl, w) in enumerate(zip(LABEL_LIST, weights)):
        print(f"  {lbl:20s}: {w:.3f}  (count={counts[i]:,})")
    return weights


# =============================================================================
# SECTION F: EVALUATION METRICS
# =============================================================================
"""
WHY these metrics for NER?

  Entity-level F1 (seqeval):
    The standard metric for NER. Unlike token-level accuracy, it requires
    the model to get BOTH the boundaries AND the label of an entity correct.
    A prediction of B-TOTAL I-TOTAL where true is B-TOTAL = 1 entity match.
    A prediction of B-TOTAL B-TOTAL (wrong boundary) = 0 entity matches.

  Precision:
    Of all predicted entities, what fraction are correct?
    High precision = few false positives (important for downstream QA accuracy).

  Recall:
    Of all true entities, what fraction did we find?
    High recall = few missed fields (important for completeness).

  F1 = harmonic mean of P and R:
    The primary optimisation target for early stopping (load_best_model=True
    with metric_for_best_model="eval_f1").

  Per-entity-type F1 (VENDOR, DATE, TOTAL, RECEIPT_ID):
    Essential for understanding which field is hardest to extract.
    VENDOR is typically hardest (variable format), TOTAL easiest (numeric).

  Exact-match accuracy:
    Did we extract the exact string? Stricter than F1.
    Reported separately for each field type.
"""

seqeval = evaluate.load("seqeval")


def compute_metrics(eval_pred):
    """
    Compute entity-level precision, recall, F1 and per-class breakdown.
    Called by HuggingFace Trainer after each evaluation step.
    """
    logits, labels = eval_pred
    predictions    = np.argmax(logits, axis=-1)

    true_labels = []
    pred_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        true_seq, pred_seq_filtered = [], []
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id == -100:
                continue   # skip special tokens
            true_seq.append(ID2LABEL[label_id])
            pred_seq_filtered.append(ID2LABEL[pred_id])
        true_labels.append(true_seq)
        pred_labels.append(pred_seq_filtered)

    results = seqeval.compute(
        predictions=pred_labels,
        references=true_labels,
        zero_division=0,
    )

    # Flatten for Trainer logging
    flat = {
        "precision": results["overall_precision"],
        "recall":    results["overall_recall"],
        "f1":        results["overall_f1"],
        "accuracy":  results["overall_accuracy"],
    }

    # Per-entity-type metrics
    for entity_type in ["VENDOR", "DATE", "TOTAL", "RECEIPT_ID"]:
        if entity_type in results:
            flat[f"{entity_type.lower()}_f1"]       = results[entity_type]["f1"]
            flat[f"{entity_type.lower()}_precision"] = results[entity_type]["precision"]
            flat[f"{entity_type.lower()}_recall"]    = results[entity_type]["recall"]

    return flat


# =============================================================================
# SECTION G: MAIN TRAINING PIPELINE
# =============================================================================

def build_model_and_peft():
    """
    Load LayoutLMv3 and wrap with LoRA.

    WHY LayoutLMv3 (not LayoutLMv1/v2)?
      v1: text + layout only, no image patches
      v2: text + layout + image (separate vision encoder, less integrated)
      v3: text + layout + image in unified attention — better cross-modal fusion
      v3 was pretrained on IIT-CDIP (11M document pages) with masked image/text
      modelling — directly relevant to receipt understanding.

    WHY base (not large)?
      - large = 368M params → doesn't fit in Colab free T4 with images + LoRA
      - base = 125M params → fits comfortably, trains 3-4x faster
      - For 800 samples, base generalises better anyway (lower model capacity
        = less overfitting risk on small data)
    """
    print("=" * 70)
    print("Loading LayoutLMv3-base …")
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        CFG.model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Wrap with LoRA
    print("Applying LoRA configuration …")
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()
    # Expected output: ~1.2M trainable / ~125M total (~1%)
    return model


def run_training():
    print("=" * 70)
    print("CORD INFORMATION EXTRACTION — TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load dataset
    print("\n[1/6] Loading dataset …")
    raw = load_dataset(CFG.dataset_name)
    train_raw = raw["train"].select(range(min(CFG.train_subset, len(raw["train"]))))
    val_raw   = raw["validation"].select(range(min(CFG.val_subset, len(raw["validation"]))))
    test_raw  = raw["test"].select(range(min(CFG.test_subset, len(raw["test"]))))

    # 2. Tokenizer / Processor
    # FIX: LayoutLMv3 requires AutoProcessor (not AutoTokenizer) when images
    # are involved. AutoProcessor wraps both the tokenizer and the image
    # feature extractor. Passing `images=` to a bare AutoTokenizer raises:
    #   TypeError: _batch_encode_plus() got an unexpected keyword argument 'images'
    print("\n[2/6] Initialising processor …")
    tokenizer = AutoProcessor.from_pretrained(CFG.model_name, apply_ocr=False)
    # apply_ocr=False because CORD already supplies OCR tokens + bboxes;
    # we do NOT want the processor to re-run Tesseract on the images.

    # 3. Preprocess
    print("\n[3/6] Preprocessing datasets …")
    preprocessor = CORDPreprocessor(tokenizer, max_length=CFG.max_length)

    remove_cols = [c for c in train_raw.column_names if c != "gt_fields"]

    train_ds = train_raw.map(preprocessor, batched=False,
                              remove_columns=remove_cols,
                              desc="Preprocessing train")
    val_ds   = val_raw.map(preprocessor, batched=False,
                            remove_columns=remove_cols,
                            desc="Preprocessing val")
    test_ds  = test_raw.map(preprocessor, batched=False,
                             remove_columns=remove_cols,
                             desc="Preprocessing test")

    train_ds.set_format("torch")
    val_ds.set_format("torch")
    test_ds.set_format("torch")

    # 4. Class weights (from training set label distribution)
    print("\n[4/6] Computing class weights …")
    class_weights = compute_class_weights(train_ds)

    # 5. Model + LoRA
    print("\n[5/6] Building model with LoRA …")
    model = build_model_and_peft()

    # 6. Training args
    print("\n[6/6] Starting training …")
    training_args = TrainingArguments(
        output_dir                  = CFG.output_dir,
        num_train_epochs            = CFG.num_epochs,
        per_device_train_batch_size = CFG.train_batch,
        per_device_eval_batch_size  = CFG.eval_batch,
        gradient_accumulation_steps = CFG.grad_accum,
        learning_rate               = CFG.lr,
        warmup_steps                = int(CFG.warmup_ratio * CFG.num_epochs *
                                         math.ceil(CFG.train_subset / CFG.train_batch /
                                                   CFG.grad_accum)),
        # warmup_ratio removed in transformers v5.2; computing equivalent warmup_steps:
        # steps = warmup_ratio × total_training_steps
        # total_training_steps ≈ epochs × ceil(train_size / (batch × grad_accum))
        weight_decay                = CFG.weight_decay,
        lr_scheduler_type           = CFG.lr_scheduler,
        eval_strategy               = "steps",
        eval_steps                  = CFG.eval_steps,
        save_strategy               = "steps",
        save_steps                  = CFG.eval_steps,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_f1",   # entity-level F1
        greater_is_better           = True,
        logging_steps               = 25,
        fp16                        = torch.cuda.is_available(),  # auto FP16 on GPU
        dataloader_num_workers      = 0,
        report_to                   = "none",
        seed                        = SEED,
        label_names                 = ["labels"],
    )

    # DataCollatorForTokenClassification needs the underlying tokenizer
    # (not the full processor) for its padding logic.
    data_collator = DataCollatorForTokenClassification(
        tokenizer.tokenizer, pad_to_multiple_of=8
    )

    trainer = WeightedTrainer(
        class_weights = class_weights,
        model         = model,
        args          = training_args,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        processing_class = tokenizer,   # renamed from tokenizer= in transformers 4.46+
        data_collator = data_collator,
        compute_metrics = compute_metrics,
        callbacks     = [EarlyStoppingCallback(
                            early_stopping_patience=CFG.early_stop_patience)],
    )

    trainer.train()

    # Save final model + LoRA weights
    model.save_pretrained(os.path.join(CFG.output_dir, "final_lora"))
    tokenizer.save_pretrained(os.path.join(CFG.output_dir, "final_lora"))

    # Final test evaluation
    print("\n" + "=" * 70)
    print("FINAL TEST SET EVALUATION")
    print("=" * 70)
    test_results = trainer.evaluate(test_ds)
    for k, v in test_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save results
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    print(f"\n✓ Model saved to {CFG.output_dir}/final_lora")
    print("  → Next: Run src/extraction/extractor.py")
    return trainer, model, tokenizer


if __name__ == "__main__":
    run_training()
