#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# llm_api.py
# ---------
# LLM API helper functions for GPT, Gemini, and Vertex Llama

import os
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
from google import genai
from google.genai import types as genai_types
from google.auth import default as google_auth_default
from google.auth.transport.requests import Request as GoogleAuthRequest

GPT_MODEL_PREFIX = "gpt"
GEMINI_MODEL_PREFIX = "gemini"
VERTEX_LLAMA_MODEL_PREFIX = "llama"

openai_client = None
gemini_client = None


def get_gcp_access_token():
    """Fetch a GCP access token."""
    SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
    credentials, project_id = google_auth_default(scopes=SCOPES)
    if not project_id:
        raise ValueError("Project ID could not be determined from credentials.")
    credentials.refresh(GoogleAuthRequest())
    return credentials.token, project_id


def call_vertex_llama(model, system_prompt, prompt, temperature=0.0, project_id=None, region="us-central1"):
    """Call Vertex AI Llama via MAAS REST endpoint."""
    token, sa_project_id = get_gcp_access_token()
    final_project_id = project_id if project_id else sa_project_id
    endpoint_url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{final_project_id}/locations/{region}/endpoints/openapi/chat/completions"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"{system_prompt}\n\n{prompt}"}],
        "temperature": temperature,
        "stream": False
    }

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    response = requests.post(endpoint_url, headers=headers, json=payload)
    response.raise_for_status()
    response_json = response.json()

    return {
        "choices": [{"message": {"content": response_json["choices"][0]["message"]["content"]}}],
        "usage": response_json.get("usage", {})
    }


def call_api(model, messages=None, payload=None, temperature=0.0, system_prompt=None, project_id=None, region="us-central1"):
    """Dispatch API call based on model type."""
    global openai_client, gemini_client

    if GPT_MODEL_PREFIX in model.lower():
        if openai_client is None:
            openai_client = OpenAI()
        response = openai_client.chat.completions.create(model=model, messages=messages, temperature=temperature)
        return {
            "choices": [{"message": {"content": response.choices[0].message.content}}],
            "usage": getattr(response.usage, "to_dict", lambda: response.usage)()
        }

    elif GEMINI_MODEL_PREFIX in model.lower():
        if gemini_client is None:
            gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        response = gemini_client.models.generate_content(
            model=model, contents=payload["contents"], config=payload["config"]
        )
        usage = response.usage_metadata
        return {
            "choices": [{"message": {"content": response.text}}],
            "usage": {
                "prompt_tokens": usage.prompt_token_count,
                "completion_tokens": usage.candidates_token_count,
                "total_tokens": usage.total_token_count
            }
        }

    elif VERTEX_LLAMA_MODEL_PREFIX in model.lower():
        return call_vertex_llama(model, system_prompt, payload["messages"][0]["content"], temperature, project_id, region)

    else:
        raise ValueError(f"Unsupported model type: {model}")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=10))
def query_model(model, system_prompt, prompt, temperature=0.0, **vertex_kwargs):
    """Unified model query interface with retries."""
    if GPT_MODEL_PREFIX in model.lower():
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        resp = call_api(model=model, messages=messages, temperature=temperature)

    elif GEMINI_MODEL_PREFIX in model.lower():
        config = genai_types.GenerateContentConfig(temperature=temperature, system_instruction=system_prompt)
        payload = {"contents": [prompt], "config": config}
        resp = call_api(model=model, payload=payload)

    elif VERTEX_LLAMA_MODEL_PREFIX in model.lower():
        payload = {"messages": [{"role": "user", "content": prompt}]}
        resp = call_api(model=model, payload=payload, temperature=temperature, system_prompt=system_prompt,
                        project_id=vertex_kwargs.get("project_id"), region=vertex_kwargs.get("region"))
    else:
        raise ValueError(f"Unsupported model type: {model}")

    return resp


def extract_response_text(resp: dict) -> str:
    """Extract text content from API response."""
    try:
        return resp["choices"][0]["message"]["content"]
    except Exception:
        return "None"
