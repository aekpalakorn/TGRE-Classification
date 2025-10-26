#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
infer.py
----------------
Perform LLM-based classification inference.
"""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
import re
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
# Utility Functions
# -------------------------------
def extract_tag(text: str, tag_name: str) -> str:
    """Extract content between <tag_name>...</tag_name> if present."""
    pattern = f"<{re.escape(tag_name)}>(.*?)</{re.escape(tag_name)}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def build_prompt(row, prompt_template, input_col):
    """Replace placeholders in prompt template."""
    instance_text = str(row[input_col]).strip()
    prompt = prompt_template.replace("${input}", instance_text)
    return prompt


def batch_iterator(df, batch_size):
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:i + batch_size], i // batch_size + 1

# -------------------------------
# Inference Logic
# -------------------------------
def run_inference(args, df, prompt_template):
    df = df.iloc[args.start_index:]
    total_batches = (len(df) + args.batch_size - 1) // args.batch_size
    csv_file, json_file = Path(args.output_csv), Path(args.raw_output_json)

    csv_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    if not args.append:
        for f in [csv_file, json_file]:
            if f.exists(): f.unlink()

    for batch_df, batch_num in batch_iterator(df, args.batch_size):
        logging.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_df)} items)...")
        batch_results, batch_raw = [], []

        for idx, (_, row) in enumerate(batch_df.iterrows(),
                                       start=batch_num * args.batch_size - args.batch_size + 1):
            prompt = build_prompt(row, prompt_template, args.input_col)
            try:
                resp_raw = query_model(args.model, args.system_prompt, prompt,
                                       temperature=args.temperature,
                                       project_id=args.vertex_project,
                                       region=args.vertex_location)
                resp_text = extract_response_text(resp_raw)
                reasoning = extract_tag(resp_text, "reasoning")
                prediction = extract_tag(resp_text, "prediction")

                batch_results.append({
                    "input": row[args.input_col],
                    "response": resp_text,
                    "reasoning": reasoning,
                    "prediction": prediction
                })
                batch_raw.append({"input": row[args.input_col], "response": resp_raw})
                logging.info(f"[Row {idx}] SUCCESS")
            except Exception as e:
                batch_results.append({
                    "input": row[args.input_col],
                    "response": "None",
                    "reasoning": "None",
                    "prediction": "None",
                    "error": str(e)
                })
                logging.error(f"[Row {idx}] FAILED: {e}")

        pd.DataFrame(batch_results).to_csv(csv_file, mode="a", index=False, header=not csv_file.exists())
        with open(json_file, "a", encoding="utf-8") as f:
            for item in batch_raw:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        time.sleep(random.uniform(0.5, 1.5))

# -------------------------------
# Debug Mode
# -------------------------------
def debug_prompt(args, df, prompt_template):
    row = df.iloc[args.start_index]
    prompt = build_prompt(row, prompt_template, args.input_col)
    logging.info("\n================ DEBUG PROMPT ================\n")
    logging.info(prompt)
    logging.info("\n=============================================\n")

# -------------------------------
# CLI Arguments
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Infer predicted labels for each input instance using LLMs.")
    parser.add_argument("--model", required=True, help="Official LLM model name to query (e.g., gpt-4, gpt-3.5-turbo).")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for model generation. Lower values produce more deterministic outputs.")
    parser.add_argument("--system_prompt", default="You are an expert classifier.", help="Optional system prompt to set model behavior or expertise context.")
    parser.add_argument("--prompt_file", required=True, help="Path to the text file containing the prompt template.")
    parser.add_argument("--log_file", required=True, help="Path to the log file for saving runtime information and error messages.")
    parser.add_argument("--input_csv", required=True, help="Path to the input CSV file containing the instances to classify.")
    parser.add_argument("--input_col", type=str, default="sentence", help="Column name in the input CSV containing the text to classify.")
    parser.add_argument("--output_csv", required=True, help="Path to the output CSV file containing structured results.")
    parser.add_argument("--raw_output_json", required=True, help="Path to the JSON file where raw model responses will be stored (newline-delimited).")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of input instances to process per batch.")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed console logging.")
    parser.add_argument("--append", action="store_true", help="Append new results to existing CSV and JSON output files instead of overwriting them.")
    parser.add_argument("--start_index", type=int, default=0, help="Start index in the input CSV for processing.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode: display constructed prompt(s) without calling the model API.")
    parser.add_argument("--vertex_project", type=str, default=os.environ.get("GCP_PROJECT_ID"), help="GCP Project ID for Vertex AI MAAS endpoints (optional, read from environment if not set).")
    parser.add_argument("--vertex_location", type=str, default=os.environ.get("GCP_REGION", "us-central1"), help="GCP Region for Vertex AI MAAS endpoints.")
    return parser.parse_args()

# -------------------------------
# Main
# -------------------------------
def main():
    args = parse_args()
    setup_logging(args.log_file, args.verbose)
    try:
        df = pd.read_csv(args.input_csv)
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        if args.debug:
            debug_prompt(args, df, prompt_template)
        else:
            run_inference(args, df, prompt_template)

    except KeyboardInterrupt:
        logging.warning("Execution interrupted by user.")
    except Exception as e:
        logging.error(f"[FATAL] {e}", exc_info=True)


if __name__ == "__main__":
    main()
