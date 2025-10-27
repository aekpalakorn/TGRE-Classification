#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_tgre.py
----------------
Generate taxonomy-guided reasoning examples (TGRE) for occupation classification.
"""

import argparse
import json
import pandas as pd
import re

from llm_api import query_model, extract_response_text 

# -------------------------------
# Core Functions
# -------------------------------
def parse_label(label: str):
    """
    Extract SOC title and code from label string.
    Example: 'Art Drama and Music Teachers Postsecondary (25-1121.00)'
    -> ('Art Drama and Music Teachers Postsecondary', '25-1121.00')
    """
    match = re.match(r"^(.*)\s+\(([\d\-\.]+)\)$", label.strip())
    if match:
        title, code = match.groups()
        return title.strip(), code.strip()
    else:
        # fallback if pattern doesn't match
        return label.strip(), ""


def construct_rationale_text(sentence: str, soc_code: str, taxonomy: dict) -> str:
    """
    Construct a grounded rationale from taxonomy information without calling the LLM.
    """
    info = taxonomy.get(soc_code, {})
    description = info.get("description", "[description missing]")
    rationale = (
        f"Given the job title '{sentence}' the individual is likely expected to {description.lower()}"
    )
    return rationale

def construct_prompt(sentence: str, soc_code: str, taxonomy: dict) -> str:
    """
    Construct an LLM prompt that continues the rationale after the fixed prefix.
    """
    info = taxonomy.get(soc_code, {})
    description = info.get("description", "[description missing]")

    prompt = (
        f"the SOC occupational description. Focus only on tasks and responsibilities.\n\n"
        f"Prefix: Given the job title '{sentence}' the individual is likely expected to...\n"
        f"SOC Description: {description}\n\n"
        f"Requirements:\n"
        f"- Always start from the prefix.\n"
        f"- Use natural, human-readable language.\n"
        f"- Align with the given SOC Description.\n"
        f"- Avoid meta commentary.\n"
        f"- Keep it concise (2-3 sentences).\n\n"
        f"Rationale:"
    )
    return prompt


def generate_grounded_rationale(sentence: str, soc_code: str, taxonomy: dict, use_llm: bool = False,
                                 model: str = None, temperature: float = 0.0,
                                 system_prompt: str = "You are an expert classifier.") -> str:
    """
    Generate taxonomy-grounded rationale using an LLM.
    """
    prompt = construct_prompt(sentence, soc_code, taxonomy)
    response_raw = query_model(model, system_prompt, prompt, temperature=temperature)
    response_text = extract_response_text(response_raw)
    return response_text


def build_tgre(df: pd.DataFrame, taxonomy: dict, args):
    """
    Generate TGRE for a random subset of input rows. Outputs JSON to console.
    """
    input_col = args.input_col
    system_prompt = args.system_prompt
    model = args.model
    temperature = args.temperature
    use_llm = args.use_llm

    # Sample n random rows
    n_samples = min(args.n_samples, len(df))
    sampled_rows = df.sample(n=n_samples, random_state=args.seed)
    grounded_rationales = []

    for _, row in sampled_rows.iterrows():
        sentence = str(row[input_col]).strip()
        soc_label_raw = str(row.get("label", "")).strip()
        soc_title, soc_code = parse_label(soc_label_raw)

        if not use_llm:
            rationale_text = construct_rationale_text(sentence, soc_code, taxonomy)
        else:
            if args.debug:
                prompt = construct_prompt(sentence, soc_code, taxonomy)
                print("\n================ DEBUG PROMPT ================\n")
                print(f"Input: {sentence}")
                print(f"SOC Title: {soc_title}")
                print(f"SOC Code: {soc_code}")
                print("\n----------------------------------------------\n")
                print(f"Constructed Prompt:\n{prompt}")
                print("\n=============================================\n")
                continue  # move to next row

            rationale_text = generate_grounded_rationale(
                sentence, soc_code, taxonomy, 
                model=model, temperature=temperature,
                system_prompt=system_prompt
            )

        grounded_rationales.append({
            "Input": sentence,
            "Reasoning": rationale_text,
            "Prediction": f"{soc_title} ({soc_code})",
        })

    return grounded_rationales

# -------------------------------
# CLI Arguments
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Generate Taxonomy-Guided Reasoning Examples (TGRE).")
    parser.add_argument("--model", required=True, help="LLM model name to use (e.g., gpt-4, gpt-3.5-turbo).")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for LLM generation.")
    parser.add_argument("--input_csv", required=True, help="CSV file containing input instances and SOC labels.")
    parser.add_argument("--input_col", type=str, default="sentence", help="Column name containing the input text.")
    parser.add_argument("--taxonomy_csv", required=True, help="CSV file containing SOC taxonomy information.")
    parser.add_argument("--system_prompt", default="You are an expert classifier.", help="Optional system prompt for the LLM.")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of random rows to process.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--use_llm", action="store_true", help="If set, call LLM to rewrite/enrich the rationale. Otherwise, use verbatim.")
    parser.add_argument("--debug", action="store_true", help="Debug mode: print constructed prompt without calling the LLM.")
    return parser.parse_args()

# -------------------------------
# Main
# -------------------------------
def main():
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    taxonomy_df = pd.read_csv(args.taxonomy_csv)
    taxonomy = {
        str(row["code"]): {"description": str(row["description"])}
        for _, row in taxonomy_df.iterrows()
    }
    rationales = build_tgre(df, taxonomy, args)
    if rationales:
        for tgre in rationales:
            for k, v in tgre.items():
                print(f"{k}: {v}")

if __name__ == "__main__":
    main()
