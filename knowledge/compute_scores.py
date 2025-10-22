#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_scores.py
-----------------
Compute accuracy of predicted answers against ground-truth labels.

Accuracy = (number of matching values) / (total number of rows)
"""

import argparse
import pandas as pd
from collections import defaultdict
import re

# Matches SOC codes of lengths 2,3,5,6,8 digits (with optional dashes and dot)
SOC_CODE_REGEX = re.compile(r"^\d{2}(-\d{1,4}(\.\d{2})?)?$")

def truncate_soc_code(val, num_digits):
    """
    Truncate an SOC code to the specified granularity.
    If val does not match SOC format, returns val unchanged.

    Example canonical format: '11-1100.00'
    num_digits: 2,3,5,6,8
    """
    if not isinstance(val, str):
        return val

    val = val.strip()
    if not SOC_CODE_REGEX.match(val):
        # Not a recognized SOC code, return as-is
        return val

    if num_digits >= 8:
        return val
    elif num_digits == 2:
        return val[:2]            # '11'
    elif num_digits == 3:
        return val[:4]            # '11-1'
    elif num_digits == 5:
        return val[:6]            # '11-110'
    elif num_digits == 6:
        return val[:7]            # '11-1100'
    else:
        return val

# -------------------------
# Accuracy / F1
# -------------------------

def compute_accuracy(df, ground_truth_col="ground-truth", answer_col="answer", num_code_digits=None):
    total = len(df)
    if num_code_digits:
        gt = df[ground_truth_col].apply(lambda x: truncate_soc_code(x, num_code_digits))
        pred = df[answer_col].apply(lambda x: truncate_soc_code(x, num_code_digits))
    else:
        gt = df[ground_truth_col]
        pred = df[answer_col]
    correct = (gt == pred).sum()
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total

def compute_per_label_f1(df, ground_truth_col="ground-truth", answer_col="answer", num_code_digits=None):
    label_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    if num_code_digits:
        gt_series = df[ground_truth_col].apply(lambda x: truncate_soc_code(x, num_code_digits))
        pred_series = df[answer_col].apply(lambda x: truncate_soc_code(x, num_code_digits))
    else:
        gt_series = df[ground_truth_col]
        pred_series = df[answer_col]

    all_labels = set(gt_series).union(set(pred_series))

    for label in all_labels:
        tp = ((pred_series == label) & (gt_series == label)).sum()
        fp = ((pred_series == label) & (gt_series != label)).sum()
        fn = ((pred_series != label) & (gt_series == label)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        label_counts[label] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    return label_counts

def compute_micro_macro_f1(per_label_metrics):
    total_tp = sum(v["tp"] for v in per_label_metrics.values())
    total_fp = sum(v["fp"] for v in per_label_metrics.values())
    total_fn = sum(v["fn"] for v in per_label_metrics.values())

    # Micro-average
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    # Macro-average
    macro_f1 = sum(v["f1"] for v in per_label_metrics.values()) / len(per_label_metrics) if per_label_metrics else 0.0

    return micro_f1, macro_f1


def analyze_none_responses(df: pd.DataFrame, answer_col="answer", gt_col="ground_truth"):
    # Ensure columns exist
    if answer_col not in df.columns or gt_col not in df.columns:
        raise ValueError(f"DataFrame must have columns '{answer_col}' and '{gt_col}'")

    # Filter 'None' responses
    none_mask = df[answer_col].astype(str).str.strip().eq("None")
    total_none = none_mask.sum()
    total_rows = len(df)

    # Breakdown per ground truth
    breakdown = df[none_mask].groupby(gt_col).size().sort_values(ascending=False)

    print(f"Total rows: {total_rows}")
    print(f"Total 'None' responses: {total_none} ({total_none/total_rows:.2%})\n")
    print("Breakdown by class label:")
    print(breakdown)

    return total_none, breakdown


def main():
    parser = argparse.ArgumentParser(description="Compute accuracy and per-label F1 scores from CSV")
    parser.add_argument("--input_csv", required=True, help="Path to input CSV file")
    parser.add_argument("--ground_truth_col", default="ground_truth", help="Name of ground_truth column")
    parser.add_argument("--answer_col", default="answer", help="Name of predicted answer column")
    parser.add_argument("--num_code_digits", type=int, default=8, help="If evaluating SOC codes, truncate to this many digits")
    parser.add_argument("--output_csv", default=None, help="Optional path to export per-label metrics as CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    # Overall accuracy
    accuracy, correct, total = compute_accuracy(df, args.ground_truth_col, args.answer_col, args.num_code_digits)
    print(f"Total rows: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f}\n")

    # Per-label metrics
    per_label_metrics = compute_per_label_f1(df, args.ground_truth_col, args.answer_col, args.num_code_digits)
    metrics_df = pd.DataFrame.from_dict(per_label_metrics, orient="index").reset_index().rename(columns={"index":"label"})
    print("Per-label metrics:")
    print(metrics_df.to_string(index=False))

    # Micro/Macro F1
    micro_f1, macro_f1 = compute_micro_macro_f1(per_label_metrics)
    print(f"\nMicro-average F1: {micro_f1:.4f}")
    print(f"Macro-average F1: {macro_f1:.4f}")

    # Optional CSV export
    if args.output_csv:
        metrics_df.to_csv(args.output_csv, index=False)
        print(f"\nPer-label metrics exported to: {args.output_csv}")

    total_none, breakdown = analyze_none_responses(df)

if __name__ == "__main__":
    main()
