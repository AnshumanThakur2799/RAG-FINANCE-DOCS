from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI


class LLMClient(Protocol):
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


@dataclass
class DeepInfraChatClient:
    api_key: str
    base_url: str
    model: str
    max_tokens: int | None = None
    temperature: float = 0.2

    def __post_init__(self) -> None:
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @staticmethod
    def _sanitize_response(raw: str) -> str:
        # Some reasoning models return internal reasoning in <think> blocks.
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE)
        return cleaned.strip()

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self.max_tokens or None,
            temperature=self.temperature,
        )
        content = response.choices[0].message.content or ""
        return self._sanitize_response(content)


def build_llm_client(
    provider: str,
    *,
    deepinfra_api_key: str | None,
    deepinfra_base_url: str,
    deepinfra_model: str,
    max_tokens: int | None = None,
    temperature: float,
) -> LLMClient:
    provider_normalized = provider.strip().lower()
    if provider_normalized in {"deepinfra", "openai-compatible"}:
        if not deepinfra_api_key:
            raise ValueError("DEEPINFRA_API_KEY is required for DeepInfra.")
        return DeepInfraChatClient(
            api_key=deepinfra_api_key,
            base_url=deepinfra_base_url,
            model=deepinfra_model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
