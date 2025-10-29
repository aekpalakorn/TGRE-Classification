# Prompt Templates

This directory contains all prompt templates used in the Taxonomic Knowledge Assessment and Occupation/Skill Classification experiments described in the [paper](https://arxiv.org/abs/2503.12989). 

## 1. Taxonomic Knowledge Assessment Prompts

Used by `knowledge/run_knowledge_task.py` to evaluate LLM taxonomic knowledge of occupations without fine-tuning or external context.  
Each prompt assesses one of two capabilities:

| **File** | **Task Type** | **Description** |
|-----------|----------------|----------------|
| `onet_soc_recall_prompt.txt` | *Recall* | Given an SOC title (or code), the model must recall the corresponding code (or title). |
| `onet_soc_recognition_prompt.txt` | *Recognition* | The model identifies the correct SOC entry from multiple plausible options. |


## 2. Occupation and Skill Classification Prompts

Used in the multi-stage classification framework for inferring and refining O*NET-SOC codes and ESCO skills from natural-language descriptions. These prompts integrate reasoning strategies such as Taxonomy-Guided Reasoning Examples (TGRE) and Chain-of-Thought (CoT) reasoning in the in-context demonstrations.

### Naming Convention

All prompt files follow the format: `{reasoning_method}_{task_type}_{stage}[_llm-name]_prompt.txt`

| **Component** |**Description** | **Example** |
|----------------|--------------|--------------|
| `reasoning_method` |Reasoning approach. <br> `tgre`: Taxonomy-Guided Reasoning Example (TGRE) <br> `cot`: Chain-of-Thought (teacher-generated) | `tgre_occ_infer_prompt.txt` |
| `task_type` | Classification task. <br>`occ`: Occupation <br>`skill`: Skill | `cot_skill_rank_gpt-4o_prompt.txt` |
| `stage` | Stage in multi-stage pipeline. <br>`infer`: First-pass classification <br>`rank`: Candidate reranking | `tgre_occ_rank_prompt.txt` |
| `llm-name` *(optional)* | LLM used for generating reasoning chains (CoT prompts only). | `cot_occ_infer_gpt-3.5-turbo_prompt.txt` |

### Usage Context

| **Stage** | **Script** | **Description** |
|------------|-------------|-------------|
| **Inference** | `classification/infer.py` | Generate initial occupation or skill predictions using TGRE or CoT reasoning. |
| **Reranking** | `classification/rerank.py` | Re-rank top-k retrieved candidates. |

### Prompt Variables
Each prompt includes placeholders (e.g., `${input}` and `${options}`) automatically filled by the pipeline functions.


## 3. LLM-as-a-Judge Evaluation Prompts

These templates define annotation tasks for using an LLM as an annotator or both occupation and skill classification.  

| **File** | **Description** | 
|-----------|--------------|
| `occupations_annotate_prompt.txt` | Evaluate the correctness, plausibility, or reasoning quality of predicted occupation labels. 
| `skills_annotate_prompt.txt` | Evaluate predicted or extracted skills for relevance and taxonomic validity.

## Reproducibility

* Prompt formatting, placeholder names, and file names have been standardized for consistency compared to the original experiments. 
* If exact prompt replication is required for reproduction, the original artifacts are preserved in the `/prompts/original_prompts/` directory.
