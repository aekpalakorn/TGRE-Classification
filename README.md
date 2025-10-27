# Assessing LLM Knowledge & Occupation/Skill Classification

## 1. Overview

This repository contains the source code and resources for the taxonomy knowledge assessment framework and occupation/skill classification described in the ICWSM [paper](https://arxiv.org/abs/2503.12989):

**"A Multi-Stage Framework with Taxonomy-Guided Reasoning for Occupation Classification Using Large Language Models."**

The codebase enables two main tasks:

1. **Assessing Large Language Models' Knowledge of Occupational Taxonomy**
   Evaluates LLMs’ internalized understanding of the O*NET-SOC 2019 taxonomy through recall and recognition tasks.

2. **Occupation and Skill Classifications using LLMs**
   Implements an LLM-based pipeline for classifying job titles or skills using a multi-stage framework consisting of inference, retrieval, and reranking with taxonomy-guided in-context examples.

### Important Note for Reproducibility

* This repository reproduces the original codebase used in conducting the experiments described in the paper. Some minor changes have been made to refactor the code, standardize prompts, and improve modularity, but the core functionality and experimental procedures remain consistent with the published work.
* All file names used in the code are placeholders. Users must substitute them with their own files.


### Repository Structure

| **Directory**             | **Content**            | **Description**                                                                                                                                 |
| ------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `/knowledge/`             | Python modules (`.py`) | Contains runnable scripts for taxonomic knowledge assessment and evaluation.                                                                    |
| `/classification/`        | Python modules (`.py`) | Contains scripts for occupation and skill classification using LLMs. |
| `/data/`                  | `.csv`  files          | Includes taxonomy files (O*NET-SOC 2019, ESCO skills), and test/validation datasets.                                                            |
| `/prompts/`               | `.txt`  files          | Contains prompt templates for taxonomy knowledge tests and occupation & skill classification tasks.                                             |


## 2. Taxonomy Knowledge Assessment Framework

The framework is implemented through the `run_knowledge_task.py` module. It uses `query_model()` for constructing standardized prompts and `call_api()` for transport across OpenAI GPT, Gemini, and Vertex AI Llama endpoints.

### Run the Recall or Recognition Test

The framework supports two types of tasks:

* **Taxonomy Recall** – tests the model's ability to recall the correct concept (e.g., SOC title given a code, or vice versa).
* **Taxonomy Recognition** – tests the model's ability to select or identify the correct answer from a given list of options.

**Parameters**

| **Argument**        | **Description**                                                                                             |
| ------------------- | ----------------------------------------------------------------------------------------------------------- |
| `task`              | Task type to run. Either `taxonomy_recall` or `taxonomy_recognition`.                                       |
| `taxonomy_file`     | Path to the O*NET-SOC 2019 taxonomy file.                                                                   |
| `prompt_file`       | Path to the text file containing the prompt template.                                                       |
| `system_prompt`     | Optional system prompt to set model behavior or expertise context.                                          |
| `query_col`         | Name of the column in the taxonomy file to use as the query field (e.g., `code`).                           |
| `answer_col`        | Name of the column in the taxonomy file to use as the answer field (e.g., `title`).                         |
| `num_digits_answer` | Number of SOC code digits to use for evaluation. Supports 2, 3, 5, 6, or 8 digits.                          |
| `output_csv`        | Path to the output CSV file containing structured results.                                                  |
| `raw_output_json`   | Path to the JSON file where raw model responses will be stored (newline-delimited).                         |
| `log_file`          | Path to the log file for saving runtime information and error messages.                                     |
| `vertex_project`    | GCP Project ID for Vertex AI MAAS endpoints (optional, read from SA if not set).                            |
| `vertex_location`   | GCP Region for Vertex AI MAAS endpoints.                                                                    |
| `model`             | Official model name to query. (e.g., `gpt-4o-mini`, `gemini-2.5-flash`, `meta/llama-3.1-8b-instruct-maas`). |
| `temperature`       | Sampling temperature for model generation (default = 0.0).                                                  |
| `partial_answer`    | If specified, provides partial answers as hints during recall tests.                                        |
| `batch_size`        | Number of input instances processed per batch.                                                              |
| `start_index`       | Start index in the taxonomy file for processing.                                                            |
| `append`            | If set, appends new results to existing output files.                                                       |
| `verbose`           | Enables detailed logging to the console.                                                                    |
| `debug `            | Run in debug mode to display the constructed prompt without making API calls.                               |

**Example Commands**

**Code-to-Title Recall Test**

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

**Title-to-Code Recall Test (2-Digit Level)**

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

**Code-to-Title Recognition Test**

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

### Evaluate Model Outputs

```bash
python compute_scores.py \
    --input_csv ../results/results.csv \
    --answer_col answer \
    --ground_truth_col ground_truth \
    --num_code_digits 8 \
    --output_csv ../results/evaluation_f1.csv
```

## 3. Occupation & Skill Classification Pipeline

The occupation and skill classification pipeline implements the multi-stage framework with taxonomy-guided reasoning examples (TGRE), described in the [paper](https://arxiv.org/abs/2503.12989). The two tasks aim to:

* **Occupation classification**: Map job titles and company name from the Jobs12K dataset, described in the paper, to standardized occupational titles and codes from the O*NET-SOC 2019 taxonomy.
* **Skill classification**: Map skills mentioned in job posts from the annotated tech skill spans dataset (Decorteetal.2022) to standardized skills from the ESCO taxonomy.

### Steps Overview

* **Step 0**: Generate sentence embeddings for O*NET-SOC and ESCO taxonomies and index them using FAISS for later retrieval.
* **Step 1**: Run inference using LLMs to predict candidate SOC or skill labels.
* **Step 2**: Retrieve top-K candidate taxonomy entries based on embedding similarity.
* **Step 3**: Rerank candidates using an LLM to produce the final prediction.

### Indexing Taxonomy

Run `index_taxonomy.py` with the following parameters:

**Parameters**

| **Argument**    | **Description**                                                                                              |
| --------------- | ------------------------------------------------------------------------------------------------------------ |
| `taxonomy_file` | Path to the taxonomy CSV file (e.g., `onet-soc-alt-titles_2019.csv` or `skills_en.csv`).                     |
| `unit`          | Retrieval unit for embedding generation. Choices: `sentence` (context-rich) or `label` (compact title/code). |
| `model`         | Name of the SentenceTransformer model used to generate embeddings (e.g., `multi-qa-mpnet-base-dot-v1`).      |
| `output_prefix` | Output prefix for saving FAISS index (`.faiss`) and metadata JSON (`_meta.json`).                            |
| `verbose`       | Enables console logging for progress and runtime information.                                                |
| `seed`          | Random seed for reproducibility (default = 42).                                                              |


**Example: Indexing O*NET-SOC 2019 (Sentence-level Retrieval)**

```bash
python index_taxonomy.py \
    --taxonomy_file ../data/onet-soc-alt-titles_2019.csv \
    --unit sentence \
    --model multi-qa-mpnet-base-dot-v1 \
    --output_prefix ../data/onet-soc_sentence_embeddings \
    --verbose
```

**Example: Indexing O*NET-SOC 2019 (Label-level Retrieval)**

```bash
python index_taxonomy.py \
    --taxonomy_file ../data/onet-soc-alt-titles_2019.csv \
    --unit label \
    --model all-mpnet-base-v2 \
    --output_prefix ../data/onet-soc_label_embeddings \
    --verbose
```

**Example: Indexing ESCO Skills (Sentence-level Retrieval)**

```bash
python index_taxonomy.py \
    --taxonomy_file ../data/skills_en.csv \
    --unit sentence \
    --model multi-qa-mpnet-base-dot-v1 \
    --output_prefix ../data/skills_sentence_embeddings \
    --verbose
```

**Example: Indexing ESCO Skills (Label-level Retrieval)**

```bash
python index_taxonomy.py \
    --taxonomy_file ../data/skils_en.csv \
    --unit label \
    --model all-mpnet-base-v2 \
    --output_prefix ../data/skills_label_embeddings \
    --verbose
```

### Inference

Run `infer.py` with the following parameters:

**Parameters**

| **Argument**      | **Description**                                                                                                      |
| ----------------- | -------------------------------------------------------------------------------------------------------------------- |
| `model`           | Official LLM model name to query (e.g., `gpt-4o-mini`, `gemini-2.5-flash`, `meta/llama-3.1-8b-instruct-maas`).       |
| `temperature`     | Sampling temperature for model generation. Lower values produce more deterministic outputs (default = 0.0).          |
| `system_prompt`   | Optional system prompt to specify model behavior or expertise context (default = `"You are an expert classifier."`). |
| `prompt_file`     | Path to the text file containing the prompt template.                                                                |
| `log_file`        | Path to the log file for saving runtime information and error messages.                                              |
| `input_csv`       | Path to the input CSV file containing instances to classify.                                                         |
| `input_col`       | Column name in the input CSV containing the text to classify (default = `sentence`).                                 |
| `output_csv`      | Path to the output CSV file containing structured inference results.                                                 |
| `raw_output_json` | Path to the JSON file for storing raw model responses (newline-delimited).                                           |
| `batch_size`      | Number of input instances processed per batch (default = 1).                                                         |
| `append`          | If set, appends new results to existing output files instead of overwriting.                                         |
| `start_index`     | Start index in the input CSV for processing (default = 0).                                                           |
| `verbose`         | Enables detailed logging to the console.                                                                             |
| `debug`           | Runs in debug mode to display constructed prompts without calling the API.                                           |
| `vertex_project`  | GCP Project ID for Vertex AI MAAS endpoints (optional, read from environment if not set).                            |
| `vertex_location` | GCP Region for Vertex AI MAAS endpoints (default = `us-central1`).                                                   |

**Example Inferring O*NET-SOC Title and Code using the TGRE-based Prompt**

```bash
python infer.py \
    --model gpt-3.5-turbo \
    --prompt_file ../prompts/tgre_occ_infer_prompt.txt \
    --log_file ../logs/run.log \
    --input_csv ../data/job_titles_test.csv \
    --output_csv ../results/occ_tgre_infer.csv \
    --raw_output_json ../results/api_responses/responses.json \
    --input_col sentence \
    --verbose
```

### Retrieval

Run `retrieve.py` with the following parameters:

**Parameters**

| **Argument**     | **Description**                                                                                                                             |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `taxonomy_type`  | Type of taxonomy used for retrieval. Choices: `ONET` (O*NET-SOC) or `ESCO` (ESCO skills taxonomy).                                          |
| `unit`           | Retrieval unit. Use `sentence` to retrieve based on reasoning text, or `label` to retrieve based on prediction text (default = `sentence`). |
| `index_file`     | Path to the FAISS index file for similarity search.                                                                                         |
| `meta_file`      | Path to the metadata JSON file corresponding to the FAISS index.                                                                            |
| `log_file`       | Path to the log file for saving runtime information and error messages.                                                                     |
| `model`          | Name of the SentenceTransformer model to use for embedding query texts.                                                                     |
| `input_csv`      | Path to the input CSV file containing the rows with reasoning and prediction fields for retrieval.                                          |
| `input_col`      | Column name in the input CSV containing the original input instance (default = `input`).                                                    |
| `reasoning_col`  | Column name in the input CSV containing the reasoning text (default = `reasoning`).                                                         |
| `prediction_col` | Column name in the input CSV containing the predicted label or description text (default = `prediction`).                                   |
| `output_csv`     | Path to the output CSV file where top-K retrieved candidates will be stored.                                                                |
| `top_k`          | Number of top documents to retrieve per query. For ESCO multi-label queries, top-K is evenly distributed across labels (default = 10).      |
| `verbose`        | Enables detailed console logging.                                                                                                           |


**Example: Retriving Top-K Candidate SOC Labels for the TGRE-based Inferences**

```bash
python retrieve.py \
    --taxonomy_type ONET \
    --unit sentence \
    --index_file ../data/onet-soc_sentence_embeddings.faiss \
    --meta_file ../data/onet-soc_sentence_embeddings_meta.json \
    --log_file ../logs/run.log \
    --model multi-qa-mpnet-base-dot-v1 \
    --top_k 10 \
    --input_csv ../results/occ_tgre_infer.csv \
    --output_csv ../results/occ_tgre_retrieve.csv \
    --verbose
```

### Reranking

Run `rerank.py` with the following parameters:

| **Argument**      | **Description**                                                                                                             |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `model`           | Official LLM model name to query (e.g., `gpt-4`, `gpt-3.5-turbo`).                                                          |
| `temperature`     | Sampling temperature for model generation. Lower values yield more deterministic outputs.                                   |
| `system_prompt`   | Optional system prompt defining model behavior or expertise context.                                                        |
| `prompt_file`     | Path to the text file containing the reranking prompt template.                                                             |
| `log_file`        | Path to the log file for saving runtime information and error messages.                                                     |
| `input_csv`       | Path to the input CSV file containing instances to be reranked.                                                             |
| `input_col`       | Column name in the input CSV containing the original input instance.                                                        |
| `candidate_col`   | Column name in the input CSV containing candidate labels to be reranked.                                                    |
| `output_csv`      | Path to the output CSV file where reranked results will be stored.                                                          |
| `raw_output_json` | Path to the newline-delimited JSON file where raw model responses will be stored.                                           |
| `verbose`         | Enables detailed console logging.                                                                                           |
| `append`          | Appends new results to existing CSV/JSON files instead of overwriting them.                                                 |
| `start_index`     | Start index in the input CSV for processing unique input instances.                                                         |
| `debug`           | Runs in debug mode: displays constructed reranking prompts without calling the model API.                                   |
| `seed`            | Random seed for deterministic shuffling of candidate labels.                                                                |
| `vertex_project`  | GCP Project ID for Vertex AI MAAS endpoints (optional, read from environment if not set).                                   |
| `vertex_location` | GCP Region for Vertex AI MAAS endpoints (default = `us-central1`).                                                          |


**Example: Reranking Candidate SOC Labels using the TGRE-based Prompt**

```bash
python rerank.py \
    --model gpt-3.5-turbo \
    --prompt_file ../prompts/tgre_occ_rank_prompt.txt \
    --log_file ../logs/run.log \
    --input_csv ../results/occ_tgre_retrieve.csv \
    --output_csv ../results/occ_tgre_rank.csv \
    --raw_output_json ../results/api_responses/responses.json \
    --input_col input \
    --candidate_col candidate_label \
    --verbose
```

### Optional: Generate Taxonomy-Guided Reasoning Examples (TGRE)

All TGRE-based prompt templates (e.g., `tgre*prompt.txt`) in this repository already include taxonomy-guided reasoning examples -- also called grounded examples -- as in-context demonstrations. You can generate additional TGREs from random validation samples by running `generate_tgre.py`.

**Example: Using SOC Description Verbatim**

Generates grounded rationales by directly inserting the official SOC occupation descriptions into a rationale template, without invoking an LLM. This corresponds to the verbatim grounding method described in the paper.

```bash
python .\generate_tgre.py \
    --input_csv ../data/job_titles_validation.csv \
    --input_col sentence \
    --taxonomy_csv ../data/onet-soc_2019.csv \
    --model gpt-3.5-turbo \
    --n_samples 2
```

**Example: Using LLM-Generated Rationales**

Alternatively, generates grounded rationales by prompting an LLM to paraphrase and enrich the SOC description, producing more natural and varied text. This method is exploratory and was not investigated in the paper.

```bash
python .\generate_tgre.py \
    --input_csv ../data/job_titles_validation.csv \
    --input_col sentence \
    --taxonomy_csv ../data/onet-soc_2019.csv \
    --model gpt-3.5-turbo \
    --n_samples 2
    --use_llm
```

## 4. Evaluation

### Computing Performance Scores

The evaluation script `compute_scores.py` computes instance-level and aggregated performance metrics for both occupation and skill classification tasks. Metrics are computed using precision@1 (for occupation classification) and ranked precision@K (for skill classification) following the definitions in the [paper](https://arxiv.org/abs/2503.12989).


**Example Command**

```bash
python compute_scores.py \
    --task occupation \
    --ground_truth_csv ../results/ground_truth.csv \
    --input_csv ../results/predictions.csv \
    --input_col sentence \
    --prediction_col prediction \
    --ground_truth_col answer \
    --output_csv ../results/scores.csv
```

### Reproducing Experimental Results

All result files used in the [paper](https://arxiv.org/abs/2503.12989) can be downloaded from the following [link](https://drive.google.com/file/d/1N-1qy8FAJHUa_-VDwyWcm4HMwnr5Z-hL/view?usp=drive_link).

After downloading, extract the archive under the `results/icwsm26/` directory:

```
results/
└── icwsm26/
    ├── occupation/
    │   ├── *.csv
    │   └── ground_truth.csv
    └── skill/
        ├── *.csv
        └── ground_truth.csv
```

Each subfolder contains raw model output files (per-method CSVs) and ground-truth tables.

Use `batch_evaluation.py` to compute evaluation metrics across all methods and models in `/results/icwsm26/`. The script automatically iterates over all relevant CSVs in the specified directory and aggregates their performance.


**Example: Occupation Classification Evaluation**

```bash
python batch_evaluation.py \
    --task occupation \
    --base_path ../results/icwsm26 \
    --ground_truth_csv ../results/icwsm26/occupation/ground_truth.csv \
    --input_col sentence \
    --prediction_col prediction \
    --ground_truth_col answer \
    --output_csv ../results/icwsm26/occupation_batch_scores.csv
```

**Example: Skill Classification Evaluation**

```bash
python batch_evaluation.py \
    --task skill \
    --base_path ../results/icwsm26 \
    --ground_truth_csv ../results/icwsm26/skill/ground_truth.csv \
    --input_col sentence \
    --prediction_col prediction \
    --ground_truth_col label \
    --output_csv ../results/icwsm26/skill_batch_scores.csv
```

**Reproducibility Note**

The reproduced scores may differ slightly from those reported in the paper (by less than 0.0001-0.001) due to floating-point precision and minor data-cleaning variations. These discrepancies do not affect any relative comparisons or conclusions presented in the paper.

## 5. Files Description

### Taxonomic Knowledge Assessment Related Files

| **File**                               | **Description**                                                        |
| -------------------------------------- | ---------------------------------------------------------------------- |
| `onet-soc_2019.csv`                    | O*NET-SOC 2019 taxonomy (8-digit codes and titles). |
| `onet_soc_recall_prompt.txt`      | Prompt template for taxonomy recall evaluation.                        |
| `onet_soc_recognition_prompt.txt` | Prompt template for taxonomy recognition evaluation.                   |

### Occupation/Skill Classification Related Files

| **File**                          | **Description**                                       |
| --------------------------------- | ----------------------------------------------------- |
| `onet-soc-alt-titles_2019.csv`    | Extended O*NET SOC titles for indexing and retrieval. |
| `skills_en.csv`                   | Skills taxonomy for classification tasks.             |
| `job_titles_test.csv`             | Test dataset for job title classification.            |
| `job_titles_validation.csv`       | Validation dataset for job titles.                    |
| `skills_test.csv`                 | Test dataset for skills classification.               |
| `skills_validation.csv`           | Validation dataset for skills.                        |
| `tgre/cot*prompt.txt`             | Various TGRE-based and CoT-based prompt templates for classification tasks.    |


### 6. Supported LLM APIs

Three major LLM API families are currently supported:

* OpenAI API -- e.g., `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`
* Google Gemini API --  e.g., `gemini-2.5-pro`, `gemini-2.5-flash`
* Open-weight models via Vertex AI's Model-as-a-Service (MAAS) -- e.g., `meta/llama-3.1-8b-instruct-maas`

Each model is specified using its official model identifier or model endpoint in API documentation.

To add support for other LLMs, modify the following functions in `knowledge/run_knowledge_task.py` or `classification/llm_api.py`:

* `query_model()` -- constructs the message or payload format expected by the API.
* `call_api()` -- sends the request and standardizes the response format (i.e., ensures it returns a dictionary containing `"choices"` and `"usage"` fields).


## 7. Setup and Installation

### Prerequisites

* Python 3.11 or higher
* Access credentials for the model API(s) you intend to use (OpenAI, Vertex AI, etc.)

### Install Dependencies

Install requirements.txt in `/knowledge/` and `/classification/`.

```bash
pip install -r requirements.txt
```

## 8. Citation

If you use this repository in your academic research, please cite:

```bibtex
@article{achananuparp2025multi,
  title={A Multi-Stage Framework with Taxonomy-Guided Reasoning for Occupation Classification Using Large Language Models},
  author={Achananuparp, Palakorn and Lim, Ee-Peng and Lu, Yao},
  journal={arXiv preprint arXiv:2503.12989},
  year={2025}
}
```