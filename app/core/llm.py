# app/core/llm.py
import os
from dotenv import load_dotenv, find_dotenv

# Always load .env for local development
load_dotenv(find_dotenv(), override=False)

# --- Streamlit secrets fallback ---
# If running in Streamlit Cloud and secrets exist, use them; else use .env
try:
    import streamlit as st
    # Only set env vars from secrets if not already set
    for key in ["OPENAI_API_KEY", "OPENAI_RESPONSES_MODEL", "OPENAI_EMBEDDINGS_MODEL"]:
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = st.secrets[key]
except Exception:
    # Not running in Streamlit or no secrets available
    pass

import numpy as np
from openai import OpenAI


class LLMClient:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found. "
                "On Streamlit Cloud add it in Secrets; locally create a .env or export it."
            )
        self.client = OpenAI(api_key=api_key)
        self.response_model = os.getenv("OPENAI_RESPONSES_MODEL", "gpt-4o-mini")
        self.embedding_model = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")
    
    def respond(self, prompt: str, model: str = None, temperature: float = 0.2, response_format=None) -> str:
        """
        Generate text response using OpenAI API.
        If response_format is provided (e.g., {"type": "json_object"}), the API will try to return strict JSON.
        """
        try:
            kwargs = {
                "model": model or self.response_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature
            }
            if response_format:
                kwargs["response_format"] = response_format  # JSON mode for supported models

            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def embed_texts(self, texts: list[str], model: str = None) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        try:
            response = self.client.embeddings.create(
                model=model or self.embedding_model,
                input=texts
            )
            embeddings = np.array([item.embedding for item in response.data], dtype=np.float32)
            return embeddings
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")


# Global client instance
llm_client = LLMClient()


def respond(prompt: str, model: str = None, temperature: float = 0.2, response_format=None) -> str:
    return llm_client.respond(prompt, model, temperature, response_format)


def embed_texts(texts: list[str], model: str = None) -> np.ndarray:
    return llm_client.embed_texts(texts, model)
