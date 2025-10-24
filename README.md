# Assessing Large Language Models' Knowledge of Occupational Taxonomy

## Overview

This repository contains the source code and resources for the taxonomy knowledge assessment framework described in the paper:

**"A Multi-Stage Framework with Taxonomy-Guided Reasoning for Occupation Classification Using Large Language Models."**

This code base enables reproduction of the Occupational Knowledge Assessment experiments in the TGRE-LLM framework. It evaluates LLM’s internalized understanding of the O*NET-SOC 2019 occupational taxonomy through recall and recognition tasks.

The repository focuses on the following core capabilities:

* Taxonomy Recall Test: Evaluates a model’s ability to generate the correct SOC code or title from memory.
* Taxonomy Recognition Test: Evaluates the model’s ability to select the correct SOC code or title among distractors.
* Digit-Level Control: Allows configurable code granularity (2, 3, 5, 6, or 8 digits) to analyze hierarchical taxonomy reasoning.


## Important Note for Reproducibility

* This codebase reproduces the original experiment scripts as closely as possible.
* File paths used in example commands are placeholders. Replace them with your own local paths as needed.
* Model outputs and evaluation results may vary slightly depending on API versions, model updates, or temperature settings.


## Repository Structure

| **Directory** | **Content**            | **Description**                                                                                                            |
| ------------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `/knowledge/` | Python modules (`.py`) | Contains runnable scripts for taxonomic knowledge assessment and evaluation.                      |
| `/data/`      | `.csv`  files  | Includes taxonomy files (O*NET-SOC 2019) and prompt templates for recall and recognition tests. |
| `/prompts/`      | `.txt`  files  | Contains prompt templates for recall and recognition tests, and occupation and classification tasks. |


## Taxonomy Knowledge Assessment Framework

The framework is implemented through the `run_knowledge_task.py` module. It uses `query_model()` for constructing standardized prompts and `call_api()` for transport across OpenAI GPT, Gemini, and Vertex AI Llama endpoints.

### 1. Run the Recall or Recognition Test

The framework supports two types of tasks:

* Taxonomy Recall -- tests the model's ability to recall the correct concept (e.g., SOC title given a code, or vice versa).
* Taxonomy Recognition -- tests the model's ability to select or identify the correct answer from a given list of options.

**Parameters**

| **Argument**        | **Description**                                                                          |
| ------------------- | ---------------------------------------------------------------------------------------- |
| `task`            | Task type to run. Either `taxonomy_recall` or `taxonomy_recognition`.                      |
| `taxonomy_file`   | Path to the O*NET-SOC 2019 taxonomy file.                                                  |
| `prompt_file`     | Path to the text file containing the prompt template.                                      |
| `system_prompt`   | Optional system prompt to set model behavior or expertise context.                         |
| `query_col`       | Name of the column in the taxonomy file to use as the query field (e.g., `code`).          |
| `answer_col`      | Name of the column in the taxonomy file to use as the answer field (e.g., `title`).        |
| `num_digits_answer` | Number of SOC code digits to use for evaluation. Supports 2, 3, 5, 6, or 8 digits.       |
| `output_csv`      | Path to the output CSV file containing structured results.                                 |
| `raw_output_json` | Path to the JSON file where raw model responses will be stored (newline-delimited).        |
| `log_file`        | Path to the log file for saving runtime information and error messages.                    |
| `vertex_project`  | GCP Project ID for Vertex AI MAAS endpoints (optional, read from SA if not set).           |
| `vertex_location` | GCP Region for Vertex AI MAAS endpoints.                                                   |
| `model`           | Official model name to query. (e.g., `gpt-4o-mini`, `gemini-2.5-flash`, `meta/llama-3.1-8b-instruct-maas`). |
| `temperature`     | Sampling temperature for model generation (default = 0.0).                                 |
| `partial_answer`  | If specified, provides partial answers as hints during recall tests.                       |
| `batch_size`      | Number of input instances processed per batch.                                             |
| `start_index `    | Start index in the taxonomy file for processing.                                           |
| `append `         | If set, appends new results to existing output files.                                      |
| `verbose `        | Enables detailed logging to the console.                                                   |
| `debug `          | Run in debug mode to display the constructed prompt without making API calls.              |


**Example Command:**

**Code-to-Title Recall Test:**

```bash
python run_knowledge_task.py \
    --task taxonomy_recall \
    --model gpt-3.5-turbo \
    --batch_size 1 \
    --taxonomy_file ../data/onet-soc_2019.csv \
    --prompt_file ../prompts/onet_soc_recall_prompt.txt \
    --log_file ../logs/run.log \
    --output_csv ../results/results.csv \
    --raw_output_json ../results/api_responses/responses.json \
    --query_col code \
    --answer_col title \
    --temperature 0.0 \
    --verbose
```

**Title-to-Code Recall Test (2-Digit Level):**

```bash
python run_knowledge_task.py \
    --task taxonomy_recall \
    --model gpt-3.5-turbo \
    --batch_size 1 \
    --taxonomy_file ../data/onet-soc_2019.csv \
    --prompt_file ../prompts/onet_soc_recall_prompt.txt \
    --log_file ../logs/run.log \
    --output_csv ../results/results.csv \
    --raw_output_json ../results/api_responses/responses.json \
    --query_col title \
    --answer_col code \
    --num_digits_answer 2 \
    --temperature 0.0 \
    --verbose
```

**Code-to-Title Recognition Test:**

```bash
python run_knowledge_task.py \
    --task taxonomy_recognition \
    --model gpt-3.5-turbo \
    --batch_size 1 \
    --taxonomy_file ../data/onet-soc_2019.csv \
    --prompt_file ../prompts/onet_soc_recognition_prompt.txt \
    --log_file ../logs/run.log \
    --output_csv ../results/results.csv \
    --raw_output_json ../results/api_responses/responses.json \
    --query_col code \
    --answer_col title \
    --temperature 0.0 \
    --verbose
```


### 2. Evaluate Model Outputs

After running the recall or recognition test, you can compute accuracy and F1 scores using `compute_scores.py`.

**Parameters**

| **Argument**         | **Description**                                                                               |
| -------------------- | --------------------------------------------------------------------------------------------- |
| `input_csv`        | Path to the CSV file containing model predictions and ground truth.                           |
| `answer_col`       | Column name in the CSV containing the model’s predicted answers (default: `answer`).          |
| `ground_truth_col` | Column name in the CSV containing the ground truth labels (default: `ground_truth`).          |
| `num_code_digits`  | Number of SOC code digits to use for evaluation. Supports 2, 3, 5, 6, or 8 digits. |
| `output_csv`       | Path to save the per-label F1 score table as CSV (optional).                                  |

**Example Command:**

```bash
python compute_scores.py \
    --input_csv ../results/results.csv \
    --answer_col answer \
    --ground_truth_col ground_truth \
    --num_code_digits 8 \
    --output_csv ../results/evaluation_f1.csv
```

## Supported LLM APIs

The repository currently supports three major LLM API families:

* OpenAI API -- e.g., `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`
* Google Gemini API --  e.g., `gemini-2.5-pro`, `gemini-2.5-flash`
* Open-weight models via Vertex AI's Model-as-a-Service (MAAS) -- e.g., `meta/llama-3.1-8b-instruct-maas`

Each model is specified using its official model identifier or model endpoint in API documentation.

To add support for other LLMs, modify the following functions in `run_knowledge_task.py`:

* `query_model()` -- constructs the message or payload format expected by the API.
* `call_api()` -- sends the request and standardizes the response format (i.e., ensures it returns a dictionary containing `"choices"` and `"usage"` fields).


## Files Description

| **File**                               | **Description**                                                        |
| -------------------------------------- | ---------------------------------------------------------------------- |
| `onet-soc_2019.csv`                    | O*NET-SOC 2019 taxonomy (8-digit codes and titles). |
| `onet_soc_recall_prompt.txt`      | Prompt template for taxonomy recall evaluation.                        |
| `onet_soc_recognition_prompt.txt` | Prompt template for taxonomy recognition evaluation.                   |


## Setup and Installation

### Prerequisites

* Python 3.11 or higher
* Access credentials for the model API(s) you intend to use (OpenAI, Vertex AI, etc.)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Requirements File

```
pandas==2.3.3
requests==2.31.0
tenacity==9.1.2
openai==1.66.3
google-auth==2.35.0
google-genai==1.41.0
protobuf==5.28.3
```

## Citation

If you use this repository in your academic work, please cite:

```bibtex
@article{achananuparp2025multi,
  title={A Multi-Stage Framework with Taxonomy-Guided Reasoning for Occupation Classification Using Large Language Models},
  author={Achananuparp, Palakorn and Lim, Ee-Peng and Lu, Yao},
  journal={arXiv preprint arXiv:2503.12989},
  year={2025}
}
```
