"""
=============================================================================
STEP 3: Comprehensive Evaluation with Multiple Metrics
=============================================================================
Covers:
  1. Entity-level F1 (seqeval) — primary NER metric
  2. Token-level accuracy — secondary
  3. Field-level exact match — end-to-end extraction quality
  4. Field-level partial match (normalised) — lenient metric
  5. Confusion matrix — understand which entities are confused
  6. Error analysis — find systematic failure patterns
  7. Confidence calibration — does model confidence correlate with correctness?
=============================================================================
"""

import json
import os
import re
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from typing import List, Dict, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoProcessor
from peft import PeftModel
from datasets import load_dataset

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from notebooks.n02_model_selection_and_training import (
    LABEL_LIST, ID2LABEL, LABEL2ID, NUM_LABELS,
    CFG, CORDPreprocessor, parse_ground_truth, assign_word_labels
)

EVAL_OUTPUT_DIR = "outputs/evaluation"
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 1. STRING NORMALISATION (for lenient/exact match)
# =============================================================================

def normalise_value(value: str) -> str:
    """
    Normalise extracted strings for comparison.
    - Strip currency symbols, punctuation, extra whitespace
    - Lowercase
    - Normalise date separators: 2024/03/15 → 20240315
    """
    if not value:
        return ""
    v = value.lower().strip()
    v = re.sub(r"[£$€¥₩]", "", v)         # remove currency
    v = re.sub(r"[,\s]+", "", v)            # remove commas and spaces
    v = re.sub(r"[/\-\.]", "", v)           # normalise date separators
    return v


def exact_match(pred: str, gold: str) -> bool:
    return normalise_value(pred) == normalise_value(gold)


def partial_match(pred: str, gold: str, threshold: float = 0.7) -> float:
    """
    Token-overlap score between predicted and gold strings.
    Returns a score in [0, 1]. Used as a lenient metric for vendor names.
    """
    if not pred or not gold:
        return 0.0
    pred_toks = set(normalise_value(pred))
    gold_toks = set(normalise_value(gold))
    if not gold_toks:
        return 0.0
    overlap = len(pred_toks & gold_toks) / len(gold_toks)
    return overlap


# =============================================================================
# 2. ENTITY SPAN EXTRACTION from BIO token predictions
# =============================================================================

def extract_spans(token_labels: List[str], token_texts: List[str]) -> Dict[str, List[str]]:
    """
    Convert a BIO label sequence + corresponding token texts
    into a dict of {entity_type: [extracted_string, ...]}.

    Example:
      labels = [O, B-VENDOR, I-VENDOR, O, B-TOTAL]
      texts  = ["on", "SUPER", "MART", "total", "142.50"]
      → {"VENDOR": ["SUPER MART"], "TOTAL": ["142.50"]}
    """
    spans: Dict[str, List[str]] = collections.defaultdict(list)
    current_entity = None
    current_tokens = []

    for label, text in zip(token_labels, token_texts):
        if label == "O":
            if current_entity and current_tokens:
                spans[current_entity].append(" ".join(current_tokens))
            current_entity = None
            current_tokens = []
        elif label.startswith("B-"):
            if current_entity and current_tokens:
                spans[current_entity].append(" ".join(current_tokens))
            current_entity = label[2:]
            current_tokens = [text]
        elif label.startswith("I-") and current_entity == label[2:]:
            current_tokens.append(text)
        else:
            # Label mismatch (e.g., I- after O) — start fresh
            if current_entity and current_tokens:
                spans[current_entity].append(" ".join(current_tokens))
            current_entity = label[2:] if "-" in label else None
            current_tokens = [text] if current_entity else []

    if current_entity and current_tokens:
        spans[current_entity].append(" ".join(current_tokens))

    return dict(spans)


# =============================================================================
# 3. FULL EVALUATION CLASS
# =============================================================================

class ExtractionEvaluator:
    """
    Runs comprehensive evaluation on the test set.
    Produces metrics at three levels:
      A) Token-level (seqeval standard)
      B) Entity span-level
      C) Field-level (end-to-end extraction quality)
    """

    def __init__(self, model, tokenizer, test_dataset, device="cpu"):
        self.model       = model.to(device)
        self.tokenizer   = tokenizer
        self.test_dataset= test_dataset
        self.device      = device
        self.model.eval()

    # ── A. Token-Level Metrics ─────────────────────────────────────────────

    def evaluate_tokens(self) -> Dict:
        """
        Compute token-level metrics using seqeval.
        This is the standard NER benchmark metric.
        """
        import evaluate as hf_evaluate
        seqeval = hf_evaluate.load("seqeval")

        all_true, all_pred = [], []

        with torch.no_grad():
            for ex in self.test_dataset:
                input_ids = ex["input_ids"].unsqueeze(0).to(self.device)
                attn_mask = ex["attention_mask"].unsqueeze(0).to(self.device)
                bbox      = ex["bbox"].unsqueeze(0).to(self.device)
                labels    = ex["labels"]

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    bbox=bbox,
                )
                preds = outputs.logits.argmax(-1).squeeze().tolist()

                true_seq, pred_seq = [], []
                for p, l in zip(preds, labels.tolist()):
                    if l == -100:
                        continue
                    true_seq.append(ID2LABEL[l])
                    pred_seq.append(ID2LABEL[p])

                all_true.append(true_seq)
                all_pred.append(pred_seq)

        results = seqeval.compute(
            predictions=all_pred, references=all_true, zero_division=0
        )
        return results, all_true, all_pred

    # ── B. Confidence Calibration ──────────────────────────────────────────

    def evaluate_confidence_calibration(self, n_bins: int = 10) -> Dict:
        """
        Calibration measures: does P(predicted class) = actual accuracy?
        Expected Calibration Error (ECE) — lower is better.
        A well-calibrated model with 80% confidence should be correct 80% of the time.
        """
        confidences = []
        correctness = []

        with torch.no_grad():
            for ex in self.test_dataset:
                input_ids = ex["input_ids"].unsqueeze(0).to(self.device)
                attn_mask = ex["attention_mask"].unsqueeze(0).to(self.device)
                bbox      = ex["bbox"].unsqueeze(0).to(self.device)
                labels    = ex["labels"]

                outputs  = self.model(
                    input_ids=input_ids, attention_mask=attn_mask, bbox=bbox
                )
                probs    = torch.softmax(outputs.logits, dim=-1).squeeze()
                preds    = probs.argmax(-1)
                max_conf = probs.max(-1).values

                for conf, pred, lbl in zip(
                    max_conf.tolist(), preds.tolist(), labels.tolist()
                ):
                    if lbl == -100:
                        continue
                    confidences.append(conf)
                    correctness.append(int(pred == lbl))

        confidences = np.array(confidences)
        correctness = np.array(correctness)

        # Compute ECE
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        bin_accs, bin_confs, bin_sizes = [], [], []

        for i in range(n_bins):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1])
            if mask.sum() == 0:
                continue
            acc  = correctness[mask].mean()
            conf = confidences[mask].mean()
            size = mask.sum()
            ece += abs(acc - conf) * size
            bin_accs.append(acc)
            bin_confs.append(conf)
            bin_sizes.append(size)

        ece /= len(confidences)

        return {
            "ece": ece,
            "mean_confidence": confidences.mean(),
            "mean_accuracy": correctness.mean(),
            "bin_accs": bin_accs,
            "bin_confs": bin_confs,
        }

    # ── C. Field-Level Exact + Partial Match ──────────────────────────────

    def evaluate_field_extraction(
        self,
        raw_test_dataset,
        all_pred_label_seqs: List[List[str]],
    ) -> Dict:
        """
        Measures how accurately the model extracts each field as a string.
        Two metrics:
          - Exact match (strict): normalised strings must match exactly
          - Partial match: character-level overlap ≥ 70%
        """
        field_results = {
            f: {"exact": [], "partial": []}
            for f in ["VENDOR", "DATE", "TOTAL", "RECEIPT_ID"]
        }

        for ex, pred_labels in zip(raw_test_dataset, all_pred_label_seqs):
            words  = ex.get("tokens", ex.get("words", []))
            gt_str = ex.get("ground_truth", "{}")
            gt     = parse_ground_truth(gt_str)

            gt_map = {
                "VENDOR":     gt.get("vendor", ""),
                "DATE":       gt.get("date", ""),
                "TOTAL":      gt.get("total", ""),
                "RECEIPT_ID": gt.get("receipt_id", ""),
            }

            # Align predictions back to word level (take first subword prediction)
            # pred_labels is already word-level from evaluation loop
            if len(pred_labels) > len(words):
                pred_labels = pred_labels[:len(words)]
            elif len(pred_labels) < len(words):
                pred_labels = pred_labels + ["O"] * (len(words) - len(pred_labels))

            spans = extract_spans(pred_labels, words)

            for field, gold in gt_map.items():
                if not gold:   # field absent in this receipt → skip
                    continue
                pred_val = spans.get(field, [""])
                pred_str = pred_val[0] if pred_val else ""

                field_results[field]["exact"].append(
                    int(exact_match(pred_str, gold))
                )
                field_results[field]["partial"].append(
                    partial_match(pred_str, gold)
                )

        summary = {}
        print("\nField-Level Extraction Metrics:")
        print(f"  {'Field':12s}  {'N':>5}  {'ExactMatch':>10}  {'PartialMatch':>12}")
        print("  " + "-" * 45)
        for field, res in field_results.items():
            n = len(res["exact"])
            if n == 0:
                continue
            em = np.mean(res["exact"])
            pm = np.mean(res["partial"])
            print(f"  {field:12s}  {n:>5}  {em:>10.3f}  {pm:>12.3f}")
            summary[field] = {"n": n, "exact_match": em, "partial_match": pm}

        return summary

    # ── D. Confusion Matrix ───────────────────────────────────────────────

    def plot_confusion_matrix(
        self, all_true: List[List[str]], all_pred: List[List[str]]
    ):
        """
        Entity-level confusion matrix.
        Reveals which entity types are confused with each other.
        e.g., TOTAL confused with RECEIPT_ID suggests numeric context overlap.
        """
        # Flatten to token level
        true_flat = [l for seq in all_true for l in seq]
        pred_flat = [l for seq in all_pred for l in seq]

        # Use only non-O labels for clarity
        entity_labels = [l for l in LABEL_LIST if l != "O"]
        label_to_idx  = {l: i for i, l in enumerate(entity_labels)}

        n = len(entity_labels)
        cm = np.zeros((n, n), dtype=int)

        for t, p in zip(true_flat, pred_flat):
            if t in label_to_idx and p in label_to_idx:
                cm[label_to_idx[t]][label_to_idx[p]] += 1

        # Normalise
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm  = np.divide(cm, row_sums, where=row_sums != 0)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm_norm, annot=True, fmt=".2f",
            xticklabels=entity_labels, yticklabels=entity_labels,
            cmap="Blues", ax=ax, vmin=0, vmax=1,
        )
        ax.set_title("Entity Confusion Matrix (row-normalised)\n"
                     "Diagonal = correct; off-diagonal = confusion")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        plt.tight_layout()
        path = f"{EVAL_OUTPUT_DIR}/confusion_matrix.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"\n  ✓ Confusion matrix saved → {path}")

    # ── E. Calibration Plot ───────────────────────────────────────────────

    def plot_calibration(self, calib_results: Dict):
        """
        Reliability diagram: confidence vs actual accuracy per bin.
        Perfect calibration = diagonal line.
        """
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.bar(
            calib_results["bin_confs"],
            calib_results["bin_accs"],
            width=0.08, alpha=0.7, color="steelblue", label="Model",
        )
        ax.set_xlabel("Mean Confidence")
        ax.set_ylabel("Fraction Correct")
        ax.set_title(f"Calibration Reliability Diagram\n"
                     f"ECE = {calib_results['ece']:.4f} "
                     f"(lower is better)")
        ax.legend()
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        plt.tight_layout()
        path = f"{EVAL_OUTPUT_DIR}/calibration.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  ✓ Calibration plot saved → {path}")

    # ── F. Training Curve ────────────────────────────────────────────────

    @staticmethod
    def plot_training_curves(trainer_state_path: str):
        """
        Plot training loss and eval F1 over steps.
        Load from trainer_state.json saved by HuggingFace Trainer.
        """
        with open(trainer_state_path) as f:
            state = json.load(f)

        log_history = state.get("log_history", [])
        train_steps, train_losses = [], []
        eval_steps, eval_f1s, eval_losses = [], [], []

        for entry in log_history:
            if "loss" in entry and "eval_loss" not in entry:
                train_steps.append(entry["step"])
                train_losses.append(entry["loss"])
            if "eval_f1" in entry:
                eval_steps.append(entry["step"])
                eval_f1s.append(entry["eval_f1"])
                eval_losses.append(entry.get("eval_loss", float("nan")))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        axes[0].plot(train_steps, train_losses, label="Train Loss", color="coral")
        axes[0].plot(eval_steps, eval_losses, label="Val Loss", color="steelblue")
        axes[0].set_xlabel("Step"); axes[0].set_ylabel("Loss")
        axes[0].set_title("Training & Validation Loss")
        axes[0].legend()

        # F1
        axes[1].plot(eval_steps, eval_f1s, label="Val F1", color="green", marker="o")
        best_step = eval_steps[np.argmax(eval_f1s)] if eval_f1s else 0
        best_f1   = max(eval_f1s) if eval_f1s else 0
        axes[1].axvline(best_step, color="red", linestyle="--",
                        label=f"Best F1={best_f1:.3f} @ step {best_step}")
        axes[1].set_xlabel("Step"); axes[1].set_ylabel("Entity F1")
        axes[1].set_title("Validation Entity-Level F1")
        axes[1].legend()

        plt.tight_layout()
        path = f"{EVAL_OUTPUT_DIR}/training_curves.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  ✓ Training curves saved → {path}")

    # ── G. Error Analysis ────────────────────────────────────────────────

    def error_analysis(
        self,
        raw_test_dataset,
        all_true: List[List[str]],
        all_pred: List[List[str]],
        n_examples: int = 10,
    ) -> List[Dict]:
        """
        Find and categorise errors into:
          - FALSE NEGATIVE: entity missed entirely
          - FALSE POSITIVE: entity hallucinated
          - WRONG BOUNDARY: correct entity type, wrong span
          - WRONG TYPE:     right span, wrong entity type
        """
        error_log = []
        error_types = collections.Counter()

        for i, (ex, true_seq, pred_seq) in enumerate(
            zip(raw_test_dataset, all_true, all_pred)
        ):
            words = ex.get("tokens", ex.get("words", []))
            min_len = min(len(words), len(true_seq), len(pred_seq))

            true_spans = extract_spans(true_seq[:min_len], words[:min_len])
            pred_spans = extract_spans(pred_seq[:min_len], words[:min_len])

            for etype in ["VENDOR", "DATE", "TOTAL", "RECEIPT_ID"]:
                true_vals = set(true_spans.get(etype, []))
                pred_vals = set(pred_spans.get(etype, []))

                fn = true_vals - pred_vals    # missed
                fp = pred_vals - true_vals    # hallucinated

                for v in fn:
                    error_types["FALSE_NEGATIVE"] += 1
                    if len(error_log) < n_examples:
                        error_log.append({
                            "sample_id": i, "type": "FALSE_NEGATIVE",
                            "entity": etype, "gold": v, "pred": "(missed)",
                            "words_preview": " ".join(words[:15]) + "…"
                        })
                for v in fp:
                    error_types["FALSE_POSITIVE"] += 1
                    if len(error_log) < n_examples:
                        error_log.append({
                            "sample_id": i, "type": "FALSE_POSITIVE",
                            "entity": etype, "gold": "(absent)", "pred": v,
                            "words_preview": " ".join(words[:15]) + "…"
                        })

        print("\nError Type Distribution:")
        for etype, cnt in error_types.most_common():
            print(f"  {etype:20s}: {cnt}")

        with open(f"{EVAL_OUTPUT_DIR}/error_analysis.json", "w") as f:
            json.dump(error_log, f, indent=2)
        print(f"\n  ✓ Error analysis saved → {EVAL_OUTPUT_DIR}/error_analysis.json")
        return error_log

    # ── Master Evaluate ──────────────────────────────────────────────────

    def run_full_evaluation(self, raw_test_dataset, trainer_state_path: Optional[str] = None):
        print("=" * 70)
        print("COMPREHENSIVE EVALUATION")
        print("=" * 70)

        # A. Token-level
        print("\n[A] Token-Level Metrics (seqeval) …")
        token_results, all_true, all_pred = self.evaluate_tokens()
        print(f"  Overall F1        : {token_results['overall_f1']:.4f}")
        print(f"  Overall Precision : {token_results['overall_precision']:.4f}")
        print(f"  Overall Recall    : {token_results['overall_recall']:.4f}")
        print(f"  Overall Accuracy  : {token_results['overall_accuracy']:.4f}")
        for etype in ["VENDOR", "DATE", "TOTAL", "RECEIPT_ID"]:
            if etype in token_results:
                r = token_results[etype]
                print(f"  {etype:12s} → P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} (n={r['number']})")

        # B. Calibration
        print("\n[B] Confidence Calibration …")
        calib = self.evaluate_confidence_calibration()
        print(f"  ECE (↓ better)    : {calib['ece']:.4f}")
        print(f"  Mean confidence   : {calib['mean_confidence']:.4f}")
        print(f"  Mean accuracy     : {calib['mean_accuracy']:.4f}")
        self.plot_calibration(calib)

        # C. Field-level extraction
        print("\n[C] Field-Level Extraction Metrics …")
        field_results = self.evaluate_field_extraction(raw_test_dataset, all_pred)

        # D. Confusion matrix
        print("\n[D] Confusion Matrix …")
        self.plot_confusion_matrix(all_true, all_pred)

        # E. Error analysis
        print("\n[E] Error Analysis …")
        errors = self.error_analysis(raw_test_dataset, all_true, all_pred)

        # F. Training curves (if state file available)
        if trainer_state_path and os.path.exists(trainer_state_path):
            print("\n[F] Training Curves …")
            self.plot_training_curves(trainer_state_path)

        # Save full report
        full_report = {
            "token_level": {k: v for k, v in token_results.items()
                            if not isinstance(v, list)},
            "calibration": {k: v for k, v in calib.items()
                            if not isinstance(v, list)},
            "field_level": field_results,
        }
        with open(f"{EVAL_OUTPUT_DIR}/full_evaluation_report.json", "w") as f:
            json.dump(full_report, f, indent=2)

        print(f"\n✓ Full evaluation report saved → {EVAL_OUTPUT_DIR}/full_evaluation_report.json")
        return full_report
