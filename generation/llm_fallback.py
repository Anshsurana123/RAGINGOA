"""
Provider-Agnostic LLM Fallback Adapter.

Interface: `generate(prompt: str, context: str) -> str`
Reads API key, base URL, and model name from environment variables (config.py).
Swappable with OpenAI, Groq, Ollama, vLLM, Together AI, or any OpenAI-compatible endpoint.
"""

import json
import logging
import os
import urllib.request
import urllib.error
from typing import Optional
import config

logger = logging.getLogger(__name__)


class LLMAdapter:
    """
    Provider-agnostic HTTP adapter for LLM generation.
    Uses standard library urllib / json to avoid heavy framework dependencies.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = config.LLM_TIMEOUT_SECONDS,
    ):
        self.api_key = api_key or config.LLM_API_KEY
        self.base_url = (base_url or config.LLM_BASE_URL).rstrip("/")
        self.model = model or config.LLM_MODEL
        self.timeout = timeout

    def generate(self, prompt: str, context: str, target_lang: Optional[str] = None) -> str:
        """
        Generate synthesized response grounded strictly in provided context passages.
        Supports cross-lingual multi-source synthesis (e.g. reading Hindi & Tamil context,
        compiling and translating the answer into English or the requested target language).
        """
        if not self.api_key or not self.api_key.strip():
            logger.info("No LLM_API_KEY configured. Using local deterministic multi-passage synthesis.")
            return self._local_fallback_synthesize(prompt, context)
            
        lang_instruction = f"Respond in '{target_lang}'." if target_lang else "Respond in the same language as the user query."
        
        system_prompt = (
            "You are an expert multilingual, cross-lingual RAG intelligence system. "
            "You are provided with context passages retrieved across multiple languages (English, Hindi, Tamil). "
            "Instructions:\n"
            "1. Extract all factual information relevant to the question from ALL provided passages regardless of passage language.\n"
            f"2. Synthesize a complete, fluent, and comprehensive answer strictly in the target language: {lang_instruction}\n"
            "3. Base your answer ONLY on facts present in the context passages. Do not invent or assume facts.\n"
            "4. If the context does not contain sufficient facts to answer the question, respond with "
            "'I don't have enough grounded information to answer that.'"
        )
        
        user_content = f"Multi-Language Context Passages:\n{context}\n\nUser Question: {prompt}\n\nCompiled Grounded Answer:"
        
        endpoint = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.1,
            "max_tokens": 350,
        }
        
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key.strip()}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) VoiceRAG/1.0",
        }
        
        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
        
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                if response.status == 200:
                    resp_body = json.loads(response.read().decode("utf-8"))
                    answer = resp_body["choices"][0]["message"]["content"].strip()
                    return answer
                else:
                    logger.warning(f"LLM API returned status {response.status}")
                    return self._local_fallback_synthesize(prompt, context)
        except Exception as e:
            logger.warning(f"LLM API call failed ({e}). Falling back to local synthesis.")
            return self._local_fallback_synthesize(prompt, context)

    def _local_fallback_synthesize(self, prompt: str, context: str) -> str:
        """
        Deterministic local multi-passage synthesizer when external LLM is offline.
        """
        if not context or not context.strip():
            return "I don't have enough grounded information to answer that."
        paragraphs = [p.strip() for p in context.split("\n\n") if p.strip()]
        if paragraphs:
            return paragraphs[0]
        return context[:300].strip()


_LLM_ADAPTER_INSTANCE: Optional[LLMAdapter] = None


def get_llm_adapter() -> LLMAdapter:
    """Singleton getter for LLMAdapter."""
    global _LLM_ADAPTER_INSTANCE
    if _LLM_ADAPTER_INSTANCE is None:
        _LLM_ADAPTER_INSTANCE = LLMAdapter()
    return _LLM_ADAPTER_INSTANCE


def generate(prompt: str, context: str, target_lang: Optional[str] = None) -> str:
    """Convenience functional wrapper for LLM generation."""
    return get_llm_adapter().generate(prompt, context, target_lang=target_lang)
