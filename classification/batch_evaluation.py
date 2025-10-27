#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
batch_evaluation.py
-----------------------
Batch compute metrics across multiple models and methods in the ICWSM paper 
using compute_scores.evaluate_single_file(), with output pivoted: 
rows = method/unit, columns = models following the same format in the paper.
"""

import argparse
import os
import pandas as pd
from itertools import product
from compute_scores import evaluate_single_file

# -------------------------
# Main batch runner
# -------------------------

def batch_evaluate(task, base_path, ground_truth_csv, input_col, prediction_col, ground_truth_col, output_csv):
    # Define base methods
    methods = ["tgre", "cot", "tgre-", "cot-", "tgre--", "cot--"]
    if task == "skill":
        # Only include tgre+ with sentence, cot+ with label
        methods.extend(["tgre+", "cot+"])
    
    models = ["gpt-3.5-turbo", "gemini-1.5-flash", "claude-instant-1.2", "llama-3-8b", "mistral-small"]
    
    results = []

    for method in methods:
        # Determine units to iterate
        if method.endswith("--"):
            units_to_iterate = ["None"]
        elif task == "skill" and method == "tgre+":
            units_to_iterate = ["sentence"]
        elif task == "skill" and method == "cot+":
            units_to_iterate = ["label"]
        else:
            units_to_iterate = ["sentence", "label"]

        for unit in units_to_iterate:
            row = {"method": method, "unit": unit}
            for model in models:
                # Decide filename
                if method.endswith("--"):
                    filename = f"{method}_{model}.csv"
                else:
                    filename = f"{method}_{model}_{unit}.csv"

                pred_path = os.path.join(base_path, task, filename)

                if not os.path.exists(pred_path):
                    print(f"Missing: {pred_path}")
                    if task == "occupation":
                        row[model] = None
                    else:
                        for k in [3, 5, 10]:
                            row[f"rp@{k}_{model}"] = None
                    continue

                try:
                    instance_df = evaluate_single_file(
                        task=task,
                        input_csv=pred_path,
                        input_col=input_col,
                        prediction_col=prediction_col,
                        ground_truth_csv=ground_truth_csv,
                        ground_truth_col=ground_truth_col
                    )

                    if task == "occupation":
                        avg_score = instance_df["score"].mean()
                        row[model] = round(avg_score, 4)
                        print(f"{filename}: prec@1={avg_score}")
                    else:
                        for k in [3, 5, 10]:
                            avg_score = instance_df[f"rp@{k}"].mean()
                            row[f"rp@{k}_{model}"] = round(avg_score, 4)
                            print(f"{filename}: rp@{k}={avg_score}")
                except Exception as e:
                    print(f"Error with {pred_path}: {e}")
                    if task == "occupation":
                        row[model] = None
                    else:
                        for k in [3, 5, 10]:
                            row[f"rp@{k}_{model}"] = None

            results.append(row)

    if results:
        df = pd.DataFrame(results)

        # Define custom order
        if task == "occupation":
            custom_order = [
                ("tgre", "sentence"), ("cot", "sentence"),
                ("tgre", "label"), ("cot", "label"),
                ("tgre-", "sentence"), ("cot-", "sentence"),
                ("tgre-", "label"), ("cot-", "label"),
                ("tgre--", "None"), ("cot--", "None")
            ]
        else:
            custom_order = [
                ("tgre+", "sentence"), ("tgre", "sentence"), ("cot", "sentence"),
                ("tgre", "label"), ("cot+", "label"), ("cot", "label"),
                ("tgre-", "sentence"), ("cot-", "sentence"),
                ("tgre-", "label"), ("cot-", "label"),
                ("tgre--", "None"), ("cot--", "None")
            ]

        order_dict = {item: idx for idx, item in enumerate(custom_order)}
        df["sort_key"] = df.apply(lambda r: order_dict.get((r["method"], r["unit"]), len(custom_order)), axis=1)
        df = df.sort_values("sort_key").drop(columns=["sort_key"])

        # Format method/unit labels
        method_map = {
            "tgre": "TGRE", "tgre-": "TGRE-", "tgre--": "TGRE--", "tgre+": "TGRE+",
            "cot": "CoT", "cot-": "CoT-", "cot--": "CoT--", "cot+": "CoT+"
        }
        df["method"] = df["method"].replace(method_map)
        df["unit"] = df["unit"].replace({"sentence": "Sentence", "label": "Label"})

        if task == "occupation":
            df.to_csv(output_csv, index=False)
            print(f"Pivoted summary saved to {output_csv}")
        else:
            for k in [3, 5, 10]:
                cols = ["method", "unit"] + [c for c in df.columns if c.startswith(f"rp@{k}_")]
                df_subset = df[cols]
                out_file = output_csv.replace(".csv", f"_rp_at_{k}.csv")
                df_subset.to_csv(out_file, index=False)
                print(f"RP@{k} pivoted summary saved to {out_file}")
    else:
        print("No results computed.")


# -------------------------
# CLI entry point
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch evaluatiton of all methods and models in the ICWSM'26 paper.")
    parser.add_argument("--task", choices=["occupation", "skill"], required=True, help="Type of classification task. 'occupation' uses precision@1, 'skill' uses RP@k metrics.")
    parser.add_argument("--base_path", required=True, help="Base path containing /occupation/ and /skill/ result subfolders")
    parser.add_argument("--ground_truth_csv", required=True, help="Path to ground-truth CSV file")
    parser.add_argument("--input_col", default="sentence", help="Column name used to join prediction and ground-truth files")
    parser.add_argument("--prediction_col", default="prediction", help="Predicted label column in prediction files")
    parser.add_argument("--ground_truth_col", default="ground_truth", help="Ground-truth label column in ground-truth CSV")
    parser.add_argument("--output_csv", required=True, help="Path to save pivoted summary CSV")
    args = parser.parse_args()

    batch_evaluate(
        task=args.task,
        base_path=args.base_path,
        ground_truth_csv=args.ground_truth_csv,
        input_col=args.input_col,
        prediction_col=args.prediction_col,
        ground_truth_col=args.ground_truth_col,
        output_csv=args.output_csv
    )

if __name__ == "__main__":
    main()
