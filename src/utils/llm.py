import os
import time
from typing import Callable
import logging

from openai import OpenAI

logger = logging.getLogger(__name__)

class LLMCaller:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        temperature: float = 0.0,
        max_retries: int = 3,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries

    def __call__(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                )
                content = response.choices[0].message.content
                return content if content else ""
            except Exception:
                logger.exception("[LLM Caller] Error on attempt %d", attempt + 1)
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                else:
                    logger.error("[LLM Caller] Failed after %d attempts.", self.max_retries)
                    return ""


def create_llm_caller(
    base_url: str = None,
    api_key: str = None,
    model_name: str = None,
) -> Callable[[str], str]:
    """"""
    base_url = base_url or os.environ.get("LLM_BASE_URL", "http://localhost/v1")
    api_key = api_key or os.environ.get("LLM_API_KEY", "sk-placeholder")
    model_name = model_name or os.environ.get("LLM_MODEL_NAME", "qwen3-max")
    return LLMCaller(base_url, api_key, model_name)
