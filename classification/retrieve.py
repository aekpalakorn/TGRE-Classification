#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
retrieve.py
------------------------
Retrieve top-K taxonomy entries for queries in an input CSV and save results to output CSV.
"""

import argparse
import json
import logging
from pathlib import Path
import re
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


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

# ---------------------------
# Utility functions
# ---------------------------

def embed_queries(model_name: str, queries: list[str]) -> np.ndarray:
    """Embed a list of queries using SentenceTransformer."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings

def search_faiss(index_file: str, meta_file: str, query_emb: np.ndarray, top_k: int = 5):
    """Retrieve top-k documents from a FAISS index."""
    index = faiss.read_index(index_file)
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    texts = metadata["texts"]

    query_emb = query_emb.astype(np.float32)
    D, I = index.search(query_emb, top_k)

    results = []
    for i, indices in enumerate(I):
        top_texts = [texts[idx] for idx in indices]
        top_scores = D[i].tolist()
        results.append(list(zip(top_texts, top_scores)))
    return results

def extract_label_from_doc(doc_text: str, taxonomy_type: str):
    """Extract canonical label from tagged document text."""
    if taxonomy_type == "ONET":
        title_match = re.search(r"<title>(.*?)</title>", doc_text, re.DOTALL)
        code_match = re.search(r"<code>(.*?)</code>", doc_text, re.DOTALL)
        title = title_match.group(1).strip() if title_match else ""
        code = code_match.group(1).strip() if code_match else ""
        return f"{title} ({code})" if title and code else title or code
    else:  # ESCO
        pref_match = re.search(r"<preferredLabel>(.*?)</preferredLabel>", doc_text, re.DOTALL)
        return pref_match.group(1).strip() if pref_match else ""


def extract_queries(row, reasoning_col: str, prediction_col: str, taxonomy_type: str, unit: str):
    """
    Construct query list from a CSV row according to taxonomy type and retrieval unit.

    For ONET:
        - sentence: use reasoning text
        - label: use prediction text

    For ESCO:
        - sentence: extract <skill> blocks, keep label and description text, remove tags
        - label: extract <label> values directly
    """
    if taxonomy_type == "ONET":
        if unit == "sentence":
            return [str(row.get(reasoning_col, ""))]
        else:
            return [str(row.get(prediction_col, ""))]

    elif taxonomy_type == "ESCO":
        prediction_text = str(row.get(prediction_col, ""))

        if unit == "sentence":
            # Extract <skill> blocks
            skill_blocks = re.findall(r"<skill>(.*?)</skill>", prediction_text, re.DOTALL)
            cleaned_skills = []
            for s in skill_blocks:
                # Replace <label>...</label> and <description>...</description> with inner text
                s_clean = re.sub(r"</?label>", "", s)
                s_clean = re.sub(r"</?description>", "", s_clean)
                s_clean = s_clean.strip()
                if s_clean:
                    cleaned_skills.append(s_clean)
            return cleaned_skills

        else:  # unit == label
            labels = re.findall(r"<label>(.*?)</label>", prediction_text, re.DOTALL)
            labels = [l.strip() for l in labels if l.strip()]
            return labels

# -------------------------------
# CLI Arguments
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Batch retrieve top-K candidate taxonomy labels for each query, based on reasoning or prediction text.")
    parser.add_argument("--taxonomy_type", choices=["ONET", "ESCO"], required=True, help="Type of taxonomy to use for retrieval: 'ONET' for O*NET-SOC, 'ESCO' for ESCO skills taxonomy.")
    parser.add_argument("--unit", choices=["sentence", "label"], default="sentence", help="Retrieval unit. 'sentence' retrieves based on reasoning text; 'label' retrieves based on prediction text.")
    parser.add_argument("--index_file", required=True, help="Path to the FAISS index file for similarity search.")
    parser.add_argument("--meta_file", required=True, help="Path to the metadata JSON file corresponding to the FAISS index.")
    parser.add_argument("--log_file", required=True, help="Path to the log file for saving runtime information and error messages.")
    parser.add_argument("--model", required=True, help="Name of the sentence embedding model (SentenceTransformer) to use for encoding queries.")
    parser.add_argument("--input_csv", required=True, help="Path to the input CSV containing rows with 'prediction' and 'reasoning' columns for retrieval.")
    parser.add_argument("--input_col", type=str, default="input", help="Column name in the input CSV containing the input instance.")
    parser.add_argument("--reasoning_col", type=str, default="reasoning", help="Column name in the input CSV containing reasoning text.")
    parser.add_argument("--prediction_col", type=str, default="prediction", help="Column name in the input CSV containing prediction text.")
    parser.add_argument("--output_csv", required=True, help="Path to the output CSV where top-K retrieved candidates will be stored.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top documents to retrieve per query. For ESCO multi-label queries, top-K is distributed evenly across each label.")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed console logging.")
    return parser.parse_args()

# ---------------------------
# Main
# ---------------------------

def main():
    args = parse_args()
    setup_logging(args.log_file, args.verbose)

    df_input = pd.read_csv(args.input_csv)
    if args.prediction_col not in df_input.columns or args.reasoning_col not in df_input.columns:
        raise ValueError(f"Input CSV must have '{args.prediction_col}' and '{args.reasoning_col}' columns.")

    all_rows = []
    
    total_rows = len(df_input)
    logging.info(f"Starting retrieval for {total_rows} rows...")

    # Process each row individually
    for idx, row in df_input.iterrows():
        queries = extract_queries(row, args.reasoning_col, args.prediction_col, args.taxonomy_type, args.unit)
        N = len(queries)
        if N == 0:
            logging.warning(f"[Row {idx+1}/{total_rows}] No queries extracted, skipping.")
            continue
        
        logging.info(f"[Row {idx+1}/{total_rows}] Processing {N} queries.")

        # Adjust top_k per query for ESCO multi-label
        effective_top_k = args.top_k // N if args.taxonomy_type == "ESCO" else args.top_k

        # Embed queries
        query_emb = embed_queries(args.model, queries)

        # Search FAISS
        retrieval_results = search_faiss(args.index_file, args.meta_file, query_emb, effective_top_k)

        # Flatten results
        for q_text, candidates in zip(queries, retrieval_results):
            for doc_text, score in candidates:
                candidate_label = extract_label_from_doc(doc_text, args.taxonomy_type)
                all_rows.append({
                    "input": row.get(args.input_col, ""),
                    "query": q_text,
                    "candidate_label": candidate_label,
                    "score": score
                })

        logging.info(f"[Row {idx+1}/{total_rows}] Retrieved {sum(len(r) for r in retrieval_results)} documents.")

    # Save output CSV
    df_output = pd.DataFrame(all_rows)
    df_output.to_csv(args.output_csv, index=False)
    logging.info(f"Saved retrieval results to {args.output_csv}")

if __name__ == "__main__":
    main()
