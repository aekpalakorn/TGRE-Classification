#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_knolwedge_task.py
-------------------
Probe LLMs for taxonomic recall and recognition of SOC titles and codes.

Features:
- Recall and recognition tests for codes at configurable SOC code granularities.
- Batch processing with append mode
- Retry-enabled API calls
- Optional partial answer hints
- Recognition distractor sampling with fixed seed
"""

import argparse
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import List, Dict

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
import requests
import os

from openai import OpenAI
from google import genai
from google.genai import types as genai_types

# --- For Vertex AI MAAS endpoint ---
from google.auth import default as google_auth_default
from google.auth.transport.requests import Request as GoogleAuthRequest

openai_client = None
gemini_client = None
GPT_MODEL_PREFIX = "gpt"
GEMINI_MODEL_PREFIX = "gemini"
VERTEX_LLAMA_MODEL_PREFIX = "llama"

# -------------------------------
# Logging Setup
# -------------------------------

def setup_logging(log_file: str, verbose: bool = False):
    """Configure logging to both file and console (if verbose)."""
    log_file_path = Path(log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    handlers = [logging.FileHandler(log_file, mode='a', encoding='utf-8')]
    if verbose:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers
    )


# -------------------------------
# Handle various LLM API requests
# -------------------------------

def get_gcp_access_token():
    """Fetches a fresh GCP access token using environment credentials (SA or ADC)."""
    
    SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
    
    try:
        credentials, project_id = google_auth_default(scopes=SCOPES)
        
        if not project_id:
              raise ValueError("Project ID could not be determined from credentials. Check file contents or IAM role.")
            
        auth_request = GoogleAuthRequest()
        credentials.refresh(auth_request)
        
        return credentials.token, project_id
        
    except Exception as e:
        import traceback
        error_info = "".join(traceback.format_exc())
        print(f"[FATAL_AUTH_ERROR] Authentication failed. Details:\n{error_info}", file=sys.stderr)
        
        raise RuntimeError(f"GCP Authentication failed: {e}")


def call_vertex_llama(
    model: str,
    system_prompt: str,
    prompt: str,
    temperature: float = 0.0,
    project_id: str = None,
    region: str = "us-central1",
):
    """
    Help function for calling Vertex AI Llama 3.1 REST API through the MAAS endpoint
    Fetches GCP access token if not provided.
    Returns standardized dict with 'choices' and 'usage'.
    """
    token, sa_project_id = get_gcp_access_token()
    final_project_id = project_id if project_id else sa_project_id

    if not final_project_id or not region:
        raise ValueError("Vertex AI Llama model requires a project_id and region.")

    MAAS_ENDPOINT_PATH = "openapi/chat/completions"
    endpoint_url = (
        f"https://{region}-aiplatform.googleapis.com/v1/projects/{final_project_id}/"
        f"locations/{region}/endpoints/{MAAS_ENDPOINT_PATH}"
    )

    full_prompt = f"{system_prompt}\n\n{prompt}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": temperature,
        "stream": False
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(endpoint_url, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()

        model_output = response_json["choices"][0]["message"]["content"]
        usage_data = response_json.get("usage", {})

        return {
            "choices": [{"message": {"content": model_output}}],
            "usage": {
                "prompt_tokens": usage_data.get("prompt_tokens"),
                "completion_tokens": usage_data.get("completion_tokens"),
                "total_tokens": usage_data.get("total_tokens")
            }
        }

    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            error_details = f"HTTP Status {e.response.status_code}: {e.response.text}"
        else:
            error_details = str(e)
        logging.error(f"[FATAL_VERTEX_ERROR] Request Failed: {error_details}")
        raise Exception(f"Vertex AI HTTP Call Error: {error_details}")


def call_api(
    model: str, 
    messages: List[Dict[str, str]] = None, 
    payload: dict = None, 
    temperature: float = 0.0, 
    system_prompt: str = None,
    project_id: str = None, 
    region: str = "us-central1"):
    """
    Low-level API submission. 
    Expects either `messages` (OpenAI/GPT) or `payload` (Vertex/Gemini).
    Returns standardized dict with choices.
    """
    if GPT_MODEL_PREFIX in model.lower():
        global openai_client
        if openai_client is None:
            openai_client = OpenAI()

        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return {
            "choices": [{"message": {"content": response.choices[0].message.content}}],
            "usage": getattr(response.usage, "to_dict", lambda: response.usage)()
        }

    elif GEMINI_MODEL_PREFIX in model.lower():
        global gemini_client
        if gemini_client is None:
            gemini_client = genai.Client(api_key=api_key)

        # payload must contain 'contents' and 'config'
        response = gemini_client.models.generate_content(
            model=model,
            contents=payload["contents"],
            config=payload["config"]
        )
        usage_metadata = response.usage_metadata
        return {
            "choices": [{"message": {"content": response.text}}],
            "usage": {
                "prompt_tokens": usage_metadata.prompt_token_count,
                "completion_tokens": usage_metadata.candidates_token_count,
                "total_tokens": usage_metadata.total_token_count
            }
        }

    elif VERTEX_LLAMA_MODEL_PREFIX in model.lower():
        return call_vertex_llama(
            model=model,
            system_prompt=system_prompt,
            prompt=payload["messages"][0]["content"],
            temperature=temperature,
            project_id=project_id,
            region=region
        )
    else:
        raise ValueError(f"Unsupported model: {model}")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
def query_model(model: str, system_prompt: str, prompt: str, temperature: float = 0.0, **vertex_kwargs) -> str:
    """
    Constructs message/payload for the API and handles retry logic.
    Returns text content only.
    """
    # GPT-style messages
    if GPT_MODEL_PREFIX in model.lower():
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        resp = call_api(model=model, messages=messages, temperature=temperature)

    # Gemini-style payload
    elif GEMINI_MODEL_PREFIX in model.lower():
        from google.genai import types as genai_types
        config = genai_types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_prompt
        )
        payload = {"contents": [prompt], "config": config}
        resp = call_api(model=model, payload=payload)

    # Vertex Llama
    elif VERTEX_LLAMA_MODEL_PREFIX in model.lower():
        payload = {
            "messages": [{"role": "user", "content": prompt}]
        }
        try:
            resp = call_api(
                model=model,
                payload=payload,
                temperature=temperature,
                system_prompt=system_prompt,
                project_id=vertex_kwargs["project_id"],
                region=vertex_kwargs["region"]
            )
        except Exception as e:
            print(e)

    else:
        raise ValueError(f"Unsupported model: {model}")

    return resp


def extract_response_text(resp: dict) -> str:
    """
    Extracts the main text content from an API response JSON.
    Returns 'None' if missing or malformed.
    """
    try:
        return resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return "None"

# -------------------------------
# Utility
# -------------------------------

def truncate_soc_code(val, answer_col, num_digits_answer):
    if answer_col == "code":
        if num_digits_answer >= 8:
            return val
        elif num_digits_answer == 2:
            return val[:num_digits_answer]
        else:
            return val[:num_digits_answer+1]
    else:
        return val
            
def extract_answer_tag(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else "None"

def batch_iterator(data: pd.DataFrame, batch_size: int):
    for i in range(0, len(data), batch_size):
        yield data.iloc[i:i+batch_size], i//batch_size + 1


# -------------------------------
# Prompt Builders
# -------------------------------

def build_taxonomy_recall_prompt(row, prompt_template, query_col, answer_col, num_digits_answer, partial_answer, taxonomy_df):
    query_val = str(row[query_col]).strip()
    answer_val = str(row[answer_col]).strip() if answer_col in row else ""

    # --------------------------
    # Example row (random, different from instance)
    # --------------------------
    if taxonomy_df is not None:
        example_row = taxonomy_df[taxonomy_df.index != row.name].sample(n=1, random_state=row.name).iloc[0]
        ex_query_val = str(example_row[query_col]).strip()
        ex_answer_val = str(example_row[answer_col]).strip()
    else:
        ex_query_val = query_val
        ex_answer_val = answer_val

    truncated_answer = truncate_soc_code(answer_val, answer_col, num_digits_answer)
    truncated_ex_answer = truncate_soc_code(ex_answer_val, answer_col, num_digits_answer)

    # --------------------------
    # Replace placeholders
    # --------------------------
    prompt = prompt_template
    # Example placeholders
    prompt = re.sub(r"\$\{example_query_field\}", query_col, prompt)
    prompt = re.sub(r"\$\{example_answer_field\}", f"{num_digits_answer}-digit {answer_col}" if answer_col=="code" else answer_col, prompt)
    prompt = re.sub(r"\$\{example_query\}", ex_query_val, prompt)
    prompt = re.sub(r"\$\{example_answer\}", truncated_ex_answer, prompt)

    # Real instance placeholders
    prompt = re.sub(r"\$\{query_field\}", query_col, prompt)
    prompt = re.sub(r"\$\{answer_field\}", f"{num_digits_answer}-digit {answer_col}" if answer_col=="code" else answer_col, prompt)
    prompt = re.sub(r"\$\{query\}", query_val, prompt)
    prompt = re.sub(r"\$\{answer\}", truncated_answer, prompt)

    # Partial hint
    if partial_answer and answer_val:
        hint_len = max(1, len(answer_val)//2)
        prompt += f"\nHint: It starts with '{answer_val[:hint_len]}'"

    return prompt


def build_taxonomy_recognition_prompt(row, prompt_template, query_col, answer_col, num_digits_answer, taxonomy_df, num_hard=2, num_easy=2):
    query_val = str(row[query_col]).strip()
    correct_answer = str(row[answer_col]).strip()
    truncated_correct = truncate_soc_code(correct_answer, answer_col, num_digits_answer)

    # Helper to sample unique distractors given a DataFrame of candidates
    def sample_unique_truncated(candidates_df, exclude_truncated_set, n, rng=None):
        """
        candidates_df: dataframe with answer_col column (full values)
        exclude_truncated_set: set of truncated strings to avoid
        returns: list of unique truncated distractor strings (len <= n)
        """
        if rng is None:
            rng = {}
        # compute truncated candidates and remove excludes
        cand_full = candidates_df[answer_col].astype(str).tolist()
        # compute truncated and preserve order
        truncated_list = [truncate_soc_code(x, answer_col, num_digits_answer) for x in cand_full]
        unique_truncated = []
        seen = set()
        # shuffle indices for randomness (but keep deterministic if DataFrame.sample used upstream)
        indices = list(range(len(truncated_list)))
        random.shuffle(indices)
        for i in indices:
            t = truncated_list[i]
            if t in exclude_truncated_set:  # skip excluded truncated ones
                continue
            if t in seen:  # ensure uniqueness in returned distractors
                continue
            seen.add(t)
            unique_truncated.append(t)
            if len(unique_truncated) >= n:
                break
        return unique_truncated

    # --------------------------
    # Real instance options
    # --------------------------
    # We will attempt to obtain num_hard from same-major candidates (by first 2 digits of full code if dealing with codes)
    options_truncated = [truncated_correct]
    exclude_set = {truncated_correct}

    if answer_col == "code":
        # use first 2 chars of the full code as the 'major' grouping (existing logic)
        major_prefix = correct_answer[:2]
        same_major = taxonomy_df[taxonomy_df[query_col].astype(str).str.startswith(major_prefix)]
        diff_major = taxonomy_df[~taxonomy_df[query_col].astype(str).str.startswith(major_prefix)]
    else:
        # for title->code mapping, same_major based on answer_col prefixes (as before)
        major_prefix = correct_answer[:2]
        same_major = taxonomy_df[taxonomy_df[answer_col].astype(str).str.startswith(major_prefix)]
        diff_major = taxonomy_df[~taxonomy_df[answer_col].astype(str).str.startswith(major_prefix)]

    # sample hard options from same_major (avoid the exact full correct answer)
    same_major = same_major[same_major[answer_col].astype(str) != correct_answer]
    hard_trunc = sample_unique_truncated(same_major, exclude_set, num_hard)
    exclude_set.update(hard_trunc)

    # sample easy options from diff_major
    easy_trunc = sample_unique_truncated(diff_major, exclude_set, num_easy)
    exclude_set.update(easy_trunc)

    # If we didn't get enough unique distractors, expand search to entire taxonomy
    needed = (num_hard + num_easy) - (len(hard_trunc) + len(easy_trunc))
    if needed > 0:
        extra_candidates = taxonomy_df[taxonomy_df[answer_col].astype(str) != correct_answer]
        extra_trunc = sample_unique_truncated(extra_candidates, exclude_set, needed)
        # distribute extras to hard/easy buckets until counts met (simple fill)
        remaining = extra_trunc
        while len(hard_trunc) < num_hard and remaining:
            hard_trunc.append(remaining.pop(0))
        while len(easy_trunc) < num_easy and remaining:
            easy_trunc.append(remaining.pop(0))

    # final options: truncated correct + truncated distractors (unique)
    options_truncated.extend(hard_trunc + easy_trunc)
    # ensure uniqueness and shuffle
    options_truncated = list(dict.fromkeys(options_truncated))  # preserve order then dedupe
    random.shuffle(options_truncated)
    real_options_str = "\n".join(options_truncated)

    # --------------------------
    # Example options (different row)
    # --------------------------
    # pick a different example row (deterministic per row)
    example_row = taxonomy_df[taxonomy_df.index != row.name].sample(n=1, random_state=row.name).iloc[0]
    ex_query_val = str(example_row[query_col]).strip()
    ex_correct_answer = str(example_row[answer_col]).strip()
    ex_truncated_correct = truncate_soc_code(ex_correct_answer, answer_col, num_digits_answer)

    # Build example distractors similarly, but use a fixed small random_state for internal sampling
    if answer_col == "code":
        ex_major_prefix = ex_correct_answer[:2]
        ex_same_major = taxonomy_df[taxonomy_df[query_col].astype(str).str.startswith(ex_major_prefix)]
        ex_diff_major = taxonomy_df[~taxonomy_df[query_col].astype(str).str.startswith(ex_major_prefix)]
    else:
        ex_major_prefix = ex_correct_answer[:2]
        ex_same_major = taxonomy_df[taxonomy_df[answer_col].astype(str).str.startswith(ex_major_prefix)]
        ex_diff_major = taxonomy_df[~taxonomy_df[answer_col].astype(str).str.startswith(ex_major_prefix)]

    ex_same_major = ex_same_major[ex_same_major[answer_col].astype(str) != ex_correct_answer]
    ex_hard_trunc = sample_unique_truncated(ex_same_major, {ex_truncated_correct}, num_hard)
    ex_exclude_set = {ex_truncated_correct} | set(ex_hard_trunc)
    ex_easy_trunc = sample_unique_truncated(ex_diff_major, ex_exclude_set, num_easy)

    # fill extras if needed
    ex_needed = (num_hard + num_easy) - (len(ex_hard_trunc) + len(ex_easy_trunc))
    if ex_needed > 0:
        ex_extra_candidates = taxonomy_df[taxonomy_df[answer_col].astype(str) != ex_correct_answer]
        ex_extra_trunc = sample_unique_truncated(ex_extra_candidates, ex_exclude_set, ex_needed)
        remaining = ex_extra_trunc
        while len(ex_hard_trunc) < num_hard and remaining:
            ex_hard_trunc.append(remaining.pop(0))
        while len(ex_easy_trunc) < num_easy and remaining:
            ex_easy_trunc.append(remaining.pop(0))

    ex_options_truncated = [ex_truncated_correct] + ex_hard_trunc + ex_easy_trunc
    ex_options_truncated = list(dict.fromkeys(ex_options_truncated))
    random.shuffle(ex_options_truncated)
    ex_options_str = "\n".join(ex_options_truncated)

    # --------------------------
    # Replace placeholders in template
    # --------------------------
    prompt = prompt_template
    # example placeholders
    prompt = re.sub(r"\$\{example_query_field\}", query_col, prompt)
    prompt = re.sub(
        r"\$\{example_answer_field\}",
        f"{num_digits_answer}-digit {answer_col}" if answer_col == "code" else answer_col,
        prompt,
    )
    prompt = re.sub(r"\$\{example_query\}", ex_query_val, prompt)
    prompt = re.sub(r"\$\{example_answer\}", ex_truncated_correct, prompt)
    prompt = re.sub(r"\$\{ex_options\}", ex_options_str, prompt)

    # real instance placeholders
    prompt = re.sub(r"\$\{query_field\}", query_col, prompt)
    prompt = re.sub(
        r"\$\{answer_field\}",
        f"{num_digits_answer}-digit {answer_col}" if answer_col == "code" else answer_col,
        prompt,
    )
    prompt = re.sub(r"\$\{query\}", query_val, prompt)
    prompt = re.sub(r"\$\{options\}", real_options_str, prompt)

    return prompt, options_truncated


# -------------------------------
# Tasks
# -------------------------------

def run_taxonomy_recall(args, taxonomy_df, prompt_template):
    taxonomy_df = taxonomy_df.iloc[args.start_index:]
    total_batches = (len(taxonomy_df) + args.batch_size - 1)//args.batch_size
    csv_file = Path(args.output_csv)
    json_file = Path(args.raw_output_json)

    # Ensure directories exist
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing files if not appending
    if not args.append:
        if csv_file.exists(): csv_file.unlink()
        if json_file.exists(): json_file.unlink()

    for batch_df, batch_num in batch_iterator(taxonomy_df, args.batch_size):
        logging.info(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch_df)} items)...")

        batch_results, batch_raw = [], []

        for idx, (_, row) in enumerate(batch_df.iterrows(), start=batch_num*args.batch_size - args.batch_size + 1):
            prompt = build_taxonomy_recall_prompt(
                row, prompt_template, args.query_col, args.answer_col,
                args.num_digits_answer, args.partial_answer, taxonomy_df
            )
            try:
                response_raw = query_model(
                    args.model,
                    args.system_prompt,
                    prompt,
                    temperature=args.temperature,
                    project_id=args.vertex_project,
                    region=args.vertex_location
                )
                response_text = extract_response_text(response_raw)
                response_answer = extract_answer_tag(response_text)
                batch_raw.append({"query": row[args.query_col], "response": response_raw})
                batch_results.append({
                    "query": row[args.query_col], 
                    "ground_truth": row[args.answer_col], 
                    "response": response_text, 
                    "answer": response_answer})
                logging.info(f"[Row {idx}] SUCCESS: {row[args.query_col]} -> {response_answer}")
            except Exception as e:
                batch_results.append({
                    "query": row[args.query_col], 
                    "ground_truth": row[args.answer_col], 
                    "response": response_text, 
                    "answer":"None",
                    "error":str(e)})
                batch_raw.append({"query": row[args.query_col], "response": response_raw,"error":str(e)})
                logging.error(f"[Row {idx}] FAILED: {e}")

        # Append batch to CSV
        pd.DataFrame(batch_results).to_csv(csv_file, mode="a", index=False, header=not csv_file.exists())
        # Append batch to JSON (newline-delimited)
        with open(json_file, "a", encoding="utf-8") as f:
            for item in batch_raw:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        time.sleep(random.uniform(0.5,1.5))


def run_taxonomy_recognition(args, taxonomy_df, prompt_template):
    taxonomy_df = taxonomy_df.iloc[args.start_index:]
    total_batches = (len(taxonomy_df) + args.batch_size - 1)//args.batch_size
    csv_file = Path(args.output_csv)
    json_file = Path(args.raw_output_json)

    # Ensure directories exist
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing files if not appending
    if not args.append:
        if csv_file.exists(): csv_file.unlink()
        if json_file.exists(): json_file.unlink()

    random.seed(args.seed)
    num_hard, num_easy = 2, 2

    for batch_df, batch_num in batch_iterator(taxonomy_df, args.batch_size):
        logging.info(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch_df)} items)...")

        batch_results, batch_raw = [], []

        for idx, (_, row) in enumerate(batch_df.iterrows(), start=batch_num*args.batch_size - args.batch_size + 1):
            prompt, options = build_taxonomy_recognition_prompt(
                row, prompt_template, args.query_col, args.answer_col, args.num_digits_answer,
                taxonomy_df, num_hard, num_easy
            )
            try:
                response_raw = query_model(
                    args.model,
                    args.system_prompt,
                    prompt,
                    temperature=args.temperature,
                    project_id=args.vertex_project,
                    region=args.vertex_location
                )
                response_text = extract_response_text(response_raw)
                response_answer = extract_answer_tag(response_text)
                batch_raw.append({"query": row[args.query_col], "response": response_raw})
                batch_results.append({
                    "query": row[args.query_col],
                    "options": options,
                    "ground_truth": row[args.answer_col],
                    "response": response_text,
                    "answer": response_answer
                })
                logging.info(f"[Row {idx}] SUCCESS: {row[args.query_col]} -> {response_answer}")
            except Exception as e:
                batch_results.append({
                    "query": row[args.query_col],
                    "options": options,
                    "ground_truth": row[args.answer_col],
                    "response": response_text,
                    "answer": "None",
                    "error": str(e)
                })
                batch_raw.append({"query": row[args.query_col], "response":response_raw,"error": str(e)})
                logging.error(f"[Row {idx}] FAILED: {e}")

        # Append batch to CSV
        pd.DataFrame(batch_results).to_csv(csv_file, mode="a", index=False, header=not csv_file.exists())
        # Append batch to JSON (newline-delimited)
        with open(json_file, "a", encoding="utf-8") as f:
            for item in batch_raw:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        time.sleep(random.uniform(0.5,1.5))


# -------------------------------
# Debug Wrappers
# -------------------------------

def debug_taxonomy_recall(args, taxonomy_df, prompt_template):
    row = taxonomy_df.iloc[args.start_index]
    prompt = build_taxonomy_recall_prompt(row, prompt_template, args.query_col, args.answer_col, args.num_digits_answer, args.partial_answer, taxonomy_df)
    logging.info("\n================ DEBUG PROMPT (Recall Task) ================\n")
    logging.info(prompt)
    logging.info("\n============================================================\n")

def debug_taxonomy_recognition(args, taxonomy_df, prompt_template):
    row = taxonomy_df.iloc[args.start_index]
    prompt,_ = build_taxonomy_recognition_prompt(row, prompt_template, args.query_col, args.answer_col, args.num_digits_answer, taxonomy_df)
    logging.info("\n================ DEBUG PROMPT (Recognition Task) ================\n")
    logging.info(prompt)
    logging.info("\n============================================================\n")

# -------------------------------
# Arguments
# -------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Probe LLM taxonomy knowledge.")
    parser.add_argument("--task", choices=["taxonomy_recall","taxonomy_recognition"], required=True, help="Task type to run. Either taxonomy_recall or taxonomy_recognition.")    
    parser.add_argument("--vertex_project", type=str, default=os.environ.get("GCP_PROJECT_ID"),
                    help="GCP Project ID for Vertex AI MAAS endpoints (optional, read from SA if not set).")
    parser.add_argument("--vertex_location", type=str, default=os.environ.get("GCP_REGION", "us-central1"),
                    help="GCP Region for Vertex AI MAAS endpoints.")    
    parser.add_argument("--model", required=True, help="Official model name to query.")
    parser.add_argument("--taxonomy_file", required=True, help="Path to the taxonomy file (CSV) containing SOC titles and codes.")
    parser.add_argument("--system_prompt", default="You are an expert O*NET-SOC 2019 coder.", help="Optional system prompt to set model behavior or expertise context.")
    parser.add_argument("--prompt_file", required=True, help="Path to the text file containing the prompt template.")
    parser.add_argument("--log_file", required=True, help="Path to the log file for saving runtime information and error messages.")
    parser.add_argument("--output_csv", required=True, help="Path to the output CSV file containing structured results.")
    parser.add_argument("--raw_output_json", required=True, help="Path to the JSON file where raw model responses will be stored (newline-delimited).")
    parser.add_argument("--query_col", required=True, help="Name of the column in the taxonomy file to use as the query field.")
    parser.add_argument("--answer_col", required=True, help="Name of the column in the taxonomy file to use as the answer field.")
    parser.add_argument("--num_digits_answer", type=int, default=8, help="Number of digits of the SOC code used for the test.")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of taxonomy instances to process per batch.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for model generation.")
    parser.add_argument("--seed", type=int, default=100, help="Random seed for shuffling candidate labels")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed console logging.")
    parser.add_argument("--append", action="store_true", help="Append new results to existing CSV and JSON output files instead of overwriting them.")
    parser.add_argument("--start_index", type=int, default=0, help="Start index in the taxonomy file for processing.")
    parser.add_argument("--partial_answer", action="store_true", help="If set, includes a partial hint of the correct answer (e.g., first few digits or characters) in the prompt.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode: only display the constructed prompt for inspection without making API calls.")

    return parser.parse_args()

# -------------------------------
# Main
# -------------------------------

def main():
    args = parse_args()
    setup_logging(args.log_file, args.verbose)

    client_to_close = None

    try:
        # --- Conditional Client Initialization ---
        if args.model.startswith("gpt"):
            # OpenAI Setup
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logging.warning("[WARNING] OPENAI_API_KEY not set")
            global openai_client
            openai_client = OpenAI()
            client_to_close = None  # no explicit close needed

        elif GEMINI_MODEL_PREFIX in args.model.lower():
            # Gemini Setup
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                logging.error("[FATAL] GEMINI_API_KEY not set. Cannot use Gemini models.")
                return

            global gemini_client
            try:
                gemini_client = genai.Client(api_key=api_key)
                client_to_close = gemini_client
                logging.info(f"[INFO] Configuring Gemini API client for model: {args.model}")
            except Exception as e:
                logging.error(f"[FATAL] Failed to initialize Gemini Client. Error: {e}")
                return

        elif VERTEX_LLAMA_MODEL_PREFIX in args.model.lower():
            # Vertex AI Setup (using requests/google-auth)
            logging.info(f"[INFO] Configuring Vertex AI Llama REST client in region: {args.vertex_location}")
            client_to_close = None  # no explicit close needed

        else:
            logging.error(f"[ERROR] Unsupported model type: {args.model}")
            return

        logging.info("Starting SOC taxonomy knowledge assessment.")
        logging.info(f"Task: {args.task}, Model: {args.model}, Temperature: {args.temperature}, Batch size: {args.batch_size}")

        Path(args.raw_output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

        taxonomy_df = pd.read_csv(args.taxonomy_file)
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        # --- Main Task Execution ---
        if args.task == "taxonomy_recall":
            if args.debug:
                debug_taxonomy_recall(args, taxonomy_df, prompt_template)
            else:
                run_taxonomy_recall(args, taxonomy_df, prompt_template)
        else:
            if args.debug:
                debug_taxonomy_recognition(args, taxonomy_df, prompt_template)
            else:
                run_taxonomy_recognition(args, taxonomy_df, prompt_template)

    except KeyboardInterrupt:
        logging.warning("Execution interrupted by user.")

    except Exception as e:
        logging.error(f"[FATAL] Unexpected error: {e}", exc_info=True)

    finally:
        if client_to_close:
            try:
                client_to_close.close()
                logging.info("Client closed successfully.")
            except Exception as e:
                logging.warning(f"Error closing client: {e}")


if __name__=="__main__":
    main()
