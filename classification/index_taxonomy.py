#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
index_taxonomy.py
----------------------
Generate sentence and label embeddings for taxonomic classes (O*NET-SOC or ESCO skills),
store vectors in FAISS index and metadata in JSON for reproducible retrieval.
"""

import argparse
import logging
import time
import json
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
import warnings

warnings.filterwarnings(
    "ignore",
    message="You try to use a model that was created with version"
)

# ---------------------------
# Utility Functions
# ---------------------------

def set_seed(seed=42):
    """Ensure reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_logging(verbose: bool):
    """Configure console logging."""
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

def load_onet_taxonomy(file_path: str, fields: list[str], for_label: bool = False):
    """
    Load O*NET-SOC taxonomy with XML-like tags.

    Example sentence embedding:
    <title>Chief Executive Officer</title>
    <code>11-1011.00</code>
    <description>Plan, direct, or coordinate operational activities...</description>
    <alternate_titles>CEO; President</alternate_titles>
    """
    df = pd.read_csv(file_path)
    for f in fields:
        if f not in df.columns:
            df[f] = ""
        df.loc[df[f].isna(), f] = ""
        df[f] = df[f].astype(str)

    docs = []
    for _, row in df.iterrows():
        if for_label:
            # Label embedding: include title and code
            parts = []
            if "title" in row: parts.append(f"<title>{row['title']}</title>")
            if "code" in row: parts.append(f"<code>{row['code']}</code>")
            text = "\n".join(parts)
        else:
            # Sentence embedding: include title, code, description, and alternate_titles
            parts = []
            if "title" in row: parts.append(f"<title>{row['title']}</title>")
            if "code" in row: parts.append(f"<code>{row['code']}</code>")
            if "description" in row: parts.append(f"<description>{row['description']}</description>")
            if "alternate_titles" in row: parts.append(f"<alternate_titles>{row['alternate_titles']}</alternate_titles>")
            text = "\n".join(parts)
        docs.append(text.strip())
    return docs


def load_esco_taxonomy(file_path: str, fields: list[str], for_label: bool = False):
    """
    Load ESCO skills taxonomy with XML-like tags.

    Example sentence embedding:
    <preferredLabel>Machine Learning</preferredLabel>
    <altLabels>ML</altLabels>
    <description>Knowledge of algorithms, models, and data handling...</description>
    """
    df = pd.read_csv(file_path)
    for f in fields:
        if f not in df.columns:
            df[f] = ""
        df.loc[df[f].isna(), f] = ""
        df[f] = df[f].astype(str).apply(lambda x: x.replace("\n", " "))

    docs = []
    for _, row in df.iterrows():
        if for_label:
            # Label embedding: include preferredLabel
            parts = []
            if "preferredLabel" in row: parts.append(f"<preferredLabel>{row['preferredLabel']}</preferredLabel>")
            text = "\n".join(parts)
        else:
            # Sentence embedding: include preferredLabel, altLabels, description
            parts = []
            if "preferredLabel" in row: parts.append(f"<preferredLabel>{row['preferredLabel']}</preferredLabel>")
            if "altLabels" in row: parts.append(f"<altLabels>{row['altLabels']}</altLabels>")
            if "description" in row: parts.append(f"<description>{row['description']}</description>")
            text = "\n".join(parts)
        docs.append(text.strip())
    return docs


def embed_texts(model_name: str, docs: list[str]) -> np.ndarray:
    """Generate embeddings using a SentenceTransformer model."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        docs,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # cosine similarity == dot product
    )
    return embeddings

def save_faiss_index(embeddings: np.ndarray, texts: list[str], output_prefix: str):
    """Save normalized embeddings and texts as FAISS index and JSON metadata."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity equivalent
    index.add(embeddings.astype(np.float32))
    
    faiss_file = f"{output_prefix}.faiss"
    json_file = f"{output_prefix}_meta.json"

    faiss.write_index(index, faiss_file)

    metadata = {
        "n_embeddings": len(texts),
        "dim": dim,
        "faiss_index_type": "IndexFlatIP",
        "normalized": True,
        "texts": texts,
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logging.info(f"FAISS index saved to: {faiss_file}")
    logging.info(f"Metadata saved to: {json_file}")

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate taxonomy embeddings as FAISS index.")
    parser.add_argument("--taxonomy_file", required=True, help="Path to taxonomy CSV file.")
    parser.add_argument("--unit", choices=["sentence", "label"], default="sentence", help="Retrieval unit.")
    parser.add_argument("--model", required=True, help="SentenceTransformer model name.")
    parser.add_argument("--output_prefix", required=True, help="Output prefix for FAISS index and metadata.")
    parser.add_argument("--verbose", action="store_true", help="Enable console logging.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    setup_logging(args.verbose)
    set_seed(args.seed)
    start_time = time.time()

    logging.info(f"Loading taxonomy from: {args.taxonomy_file}")
    logging.info(f"Retrieval unit: {args.unit}")
    logging.info(f"Using model: {args.model}")

    try:
        # Detect retrieval unit for embeddings
        if "onet" in args.taxonomy_file.lower():
            if args.unit == "sentence":
                fields = ["title", "code", "description", "alternate_titles"]
                docs = load_onet_taxonomy(args.taxonomy_file, fields, for_label=False)
            else:
                docs = load_onet_taxonomy(args.taxonomy_file, ["title", "code"], for_label=True)
            taxonomy_type = "O*NET-SOC"
        elif "esco" in args.taxonomy_file.lower() or "skill" in args.taxonomy_file.lower():
            if args.unit == "sentence":
                fields = ["preferredLabel", "altLabels", "description"]
                docs = load_esco_taxonomy(args.taxonomy_file, fields, for_label=False)
            else:
                docs = load_esco_taxonomy(args.taxonomy_file, ["preferredLabel"], for_label=True)
            taxonomy_type = "ESCO"
        else:
            raise ValueError("Unable to detect taxonomy type. Filename must include 'onet' or 'esco'.")

        logging.info(f"Detected taxonomy type: {taxonomy_type}")
        logging.info(f"Number of documents to embed: {len(docs):,}")

        embeddings = embed_texts(args.model, docs)

        save_faiss_index(embeddings, docs, args.output_prefix)

        elapsed = (time.time() - start_time) / 60
        logging.info(f"Total runtime: {elapsed:.2f} minutes")

    except KeyboardInterrupt:
        logging.warning("Execution interrupted by user.")
    except Exception as e:
        logging.error(f"[FATAL] {e}", exc_info=True)

if __name__ == "__main__":
    main()
