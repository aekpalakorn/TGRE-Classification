#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rerank.py
----------------
Perform reranking of candidate labels using LLMs.
"""

import argparse
import json
import logging
import os
import random
import re
import time
from pathlib import Path

import pandas as pd

from llm_api import query_model, extract_response_text

# -------------------------------
# Logging Setup
# -------------------------------
def setup_logging(log_file: str, verbose: bool = False):
    log_file_path = Path(log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [logging.FileHandler(log_file, mode="a", encoding="utf-8")]
    if verbose:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers
    )

# -------------------------------
# Helper: XML tag parser
# -------------------------------
def extract_tag_content(text: str, tag: str):
    """Extract content inside a specific XML-like tag."""
    match = re.search(fr"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None

# -------------------------------
# Core Logic
# -------------------------------
def build_rerank_prompt(input_text, candidate_labels, prompt_template):
    """Replace placeholders in reranking prompt template."""
    options_text = "\n".join([f"- {c.strip()}" for c in candidate_labels])
    prompt = prompt_template.replace("${input}", input_text)
    prompt = prompt.replace("${options}", options_text)
    return prompt

def run_reranking(args, df, prompt_template):
    """Perform reranking over candidate labels (grouped per unique input)."""
    df = df.iloc[args.start_index:]
    csv_file, json_file = Path(args.output_csv), Path(args.raw_output_json)

    csv_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    if not args.append:
        for f in [csv_file, json_file]:
            if f.exists():
                f.unlink()

    random.seed(args.seed)
    logging.info(f"Candidate label shuffling enabled (seed={args.seed}).")

    unique_inputs = df[args.input_col].unique()
    total_instances = len(unique_inputs)
    logging.info(f"Found {total_instances} unique input instances.")

    for i, input_text in enumerate(unique_inputs, start=args.start_index):
        subset = df[df[args.input_col] == input_text]
        candidate_labels = subset[args.candidate_col].tolist()
        scores = subset["score"].tolist()

        # Shuffle candidates deterministically
        combined = list(zip(candidate_labels, scores))
        random.shuffle(combined)
        candidate_labels, scores = zip(*combined) if combined else ([], [])

        prompt = build_rerank_prompt(input_text, candidate_labels, prompt_template)

        try:
            resp_raw = query_model(
                args.model,
                args.system_prompt,
                prompt,
                temperature=args.temperature,
                project_id=args.vertex_project,
                region=args.vertex_location
            )
            resp_text = extract_response_text(resp_raw)

            reasoning = extract_tag_content(resp_text, "reasoning")
            prediction = extract_tag_content(resp_text, "prediction")

            result = {
                "input": input_text,
                "candidate_labels": ";".join(candidate_labels),
                "rerank_response": resp_text,
                "reasoning": reasoning,
                "prediction": prediction
            }

            pd.DataFrame([result]).to_csv(csv_file, mode="a", index=False, header=not csv_file.exists())
            with open(json_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"input": input_text, "response": resp_raw}, ensure_ascii=False) + "\n")

            logging.info(f"[Instance {i+1}/{total_instances}] SUCCESS")

        except Exception as e:
            logging.error(f"[Instance {i+1}/{total_instances}] FAILED: {e}")
            pd.DataFrame([{
                "input": input_text,
                "candidate_labels": ";".join(candidate_labels),
                "response": "None",
                "reasoning": None,
                "prediction": None,
                "error": str(e)
            }]).to_csv(csv_file, mode="a", index=False, header=not csv_file.exists())

        time.sleep(random.uniform(0.5, 1.5))

# -------------------------------
# Debug Mode
# -------------------------------
def debug_prompt(args, df, prompt_template):
    input_text = df.iloc[args.start_index][args.input_col]
    candidate_labels = df[df[args.input_col] == input_text][args.candidate_col].tolist()

    random.seed(args.seed)
    random.shuffle(candidate_labels)
    logging.info(f"Debug mode: showing shuffled candidates (seed={args.seed})")

    options_text = "\n".join(f"- {label}" for label in candidate_labels)
    prompt = prompt_template.replace("${input}", input_text).replace("${options}", options_text)

    logging.info("\n================ DEBUG PROMPT ================\n")
    logging.info(prompt)
    logging.info("\n=============================================\n")

# -------------------------------
# CLI Arguments
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Rerank top-K candidate labels for each input instance using LLMs.")
    parser.add_argument("--model", required=True, help="Official LLM model name to query (e.g., gpt-4, gpt-3.5-turbo).")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for model generation. Lower values produce more deterministic outputs.")
    parser.add_argument("--system_prompt", default="You are an expert classifier.", help="Optional system prompt to set model behavior or expertise context.")
    parser.add_argument("--prompt_file", required=True, help="Path to the text file containing the prompt template.")
    parser.add_argument("--log_file", required=True, help="Path to the log file for saving runtime information and error messages.")
    parser.add_argument("--input_csv", required=True, help="Path to the input CSV file containing the instances to classify.")
    parser.add_argument("--input_col", type=str, default="input", help="Column name in the input CSV containing the input instance.")
    parser.add_argument("--candidate_col", type=str, default="candidate_label", help="Column name in the input CSV containing candidate labels to be reranked.")
    parser.add_argument("--output_csv", required=True, help="Path to the output CSV file containing structured results.")
    parser.add_argument("--raw_output_json", required=True, help="Path to the JSON file where raw model responses will be stored (newline-delimited).")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed console logging.")
    parser.add_argument("--append", action="store_true", help="Append new results to existing CSV and JSON output files instead of overwriting them.")
    parser.add_argument("--start_index", type=int, default=0, help="Start index in the input CSV for processing, counting only unique input stances.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode: display constructed prompt(s) without calling the model API.")
    parser.add_argument("--seed", type=int, default=23, help="Random seed for shuffling candidate labels.")
    parser.add_argument("--vertex_project", type=str, default=os.environ.get("GCP_PROJECT_ID"), help="GCP Project ID for Vertex AI MAAS endpoints (optional, read from environment if not set).")
    parser.add_argument("--vertex_location", type=str, default=os.environ.get("GCP_REGION", "us-central1"), help="GCP Region for Vertex AI MAAS endpoints.")
    return parser.parse_args()

# -------------------------------
# Main
# -------------------------------
def main():
    args = parse_args()
    setup_logging(args.log_file, args.verbose)
    logging.info("Starting reranking task...")

    df = pd.read_csv(args.input_csv)
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    if args.debug:
        debug_prompt(args, df, prompt_template)
    else:
        run_reranking(args, df, prompt_template)

if __name__ == "__main__":
    main()
