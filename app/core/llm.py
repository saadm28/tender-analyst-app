# app/core/llm.py
import os
from dotenv import load_dotenv, find_dotenv

# Load .env for local dev
load_dotenv(find_dotenv(), override=False)

# --- Streamlit secrets shim (Cloud-safe) ---
try:
    import streamlit as st  # available when running inside Streamlit
    if "OPENAI_API_KEY" in st.secrets and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    # optional: allow model overrides via secrets too
    if "OPENAI_RESPONSES_MODEL" in st.secrets and not os.getenv("OPENAI_RESPONSES_MODEL"):
        os.environ["OPENAI_RESPONSES_MODEL"] = st.secrets["OPENAI_RESPONSES_MODEL"]
    if "OPENAI_EMBEDDINGS_MODEL" in st.secrets and not os.getenv("OPENAI_EMBEDDINGS_MODEL"):
        os.environ["OPENAI_EMBEDDINGS_MODEL"] = st.secrets["OPENAI_EMBEDDINGS_MODEL"]
except Exception:
    # running outside Streamlit; ignore
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
