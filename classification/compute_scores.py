#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_scores.py
-----------------
Compute instance-level scores for occupation (precision@1) or skill classification (RP@3,5,10)
and save instance-level results to CSV. Prints average score to the console.
"""

import argparse
import pandas as pd
import string
from typing import List

# -------------------------
# Utility functions
# -------------------------

def preprocess_label(text: str) -> str:
    """Lowercase, remove punctuation, and strip spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return text.strip()

def rp_at_k(gold: List[str], predicted: List[str], k: int) -> float:
    """Rank Precision@K as defined in IReRa."""
    gold_k = min(k, len(gold))
    top_k_predicted = predicted[:k]
    true_positives = sum(1 for item in top_k_predicted if item in gold)
    return true_positives / gold_k if gold_k > 0 else 0.0

# -------------------------
# Core metrics
# -------------------------

def compute_instance_precision_at_k(gt_labels: List[str], pred_labels: List[str], k=1) -> float:
    """Compute precision@k for a single instance."""
    top_k_preds = pred_labels[:k]
    return 1.0 if any(pred in gt_labels for pred in top_k_preds) else 0.0

def compute_instance_rp_at_k(gt_labels: List[str], pred_labels: List[str], k_list=[3,5,10]) -> dict:
    """Compute RP@k for a single instance."""
    scores = {}
    for k in k_list:
        scores[f"rp@{k}"] = rp_at_k(gt_labels, pred_labels, k)
    return scores

def evaluate_single_file(task, input_csv, input_col, prediction_col, ground_truth_csv, ground_truth_col):
    """Evaluate one prediction file and return instance-level scores as a DataFrame."""
    pred_df = pd.read_csv(input_csv, encoding="utf8").drop_duplicates(subset=[input_col])
    gt_df = pd.read_csv(ground_truth_csv, encoding="utf8").drop_duplicates(subset=[input_col])

    df = pd.merge(pred_df, gt_df, on=input_col, how="inner")

    if len(df) == 0:
        raise ValueError(f"No matching rows found between {input_csv} and {ground_truth_csv}.")

    if len(df) != len(gt_df):
        missing = len(gt_df) - len(df)
        print(
            f"Warning: Only {len(df)}/{len(gt_df)} ground-truth rows matched in {input_csv}. "
            f"{missing} instances could not be aligned.\n"
        )

    results = []

    for _, row in df.iterrows():
        gold = list({preprocess_label(x) for x in str(row[ground_truth_col]).split(";") if x.strip()})
        pred = [preprocess_label(x) for x in str(row[prediction_col]).split(";") if x.strip()]

        if task == "occupation":
            score = compute_instance_precision_at_k(gold, pred, k=1)
            results.append({
                "input": row[input_col],
                "ground_truth": row[ground_truth_col],
                "predicted": row[prediction_col],
                "score": score
            })
        elif task == "skill":
            rp_scores = compute_instance_rp_at_k(gold, pred, k_list=[3,5,10])
            results.append({
                "input": row[input_col],
                "ground_truth": row[ground_truth_col],
                "predicted": row[prediction_col],
                **rp_scores
            })
        else:
            raise ValueError(f"Unknown task type: {task}")

    instance_df = pd.DataFrame(results)
    return instance_df

# -------------------------
# CLI entry point
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute instance-level classification scores.")
    parser.add_argument("--task", choices=["occupation", "skill"], required=True, help="Type of classification task. 'occupation' uses precision@1, 'skill' uses RP@k metrics.")
    parser.add_argument("--input_csv", required=True, help="Path to the CSV file containing model predictions.")
    parser.add_argument("--input_col", required=True, default="sentence", help="Column name used to match predictions with ground-truth instances (e.g., 'sentence').")
    parser.add_argument("--prediction_col", required=True, help="Column name containing the predicted labels (semicolon-separated for multi-label cases).")
    parser.add_argument("--ground_truth_csv", required=True, help="Path to the ground-truth CSV file containing true labels for evaluation.")
    parser.add_argument("--ground_truth_col", required=True, help="Column name in the ground-truth CSV that contains the true labels (semicolon-separated).")
    parser.add_argument("--output_csv", required=True, help="Path to save the instance-level results with computed scores.")
    args = parser.parse_args()

    instance_df = evaluate_single_file(
        args.task, args.input_csv, args.input_col,
        args.prediction_col, args.ground_truth_csv, args.ground_truth_col
    )

    # Save CSV without the average row
    instance_df.to_csv(args.output_csv, index=False)
    print(f"Instance-level results saved to {args.output_csv}")

    # Compute and print average scores
    if args.task == "occupation":
        avg_score = instance_df["score"].mean()
        print(f"Average precision@1: {avg_score:.6f}")
    elif args.task == "skill":
        for k in [3,5,10]:
            avg_rp = instance_df[f"rp@{k}"].mean()
            print(f"Average rp@{k}: {avg_rp:.6f}")

if __name__ == "__main__":
    main()
