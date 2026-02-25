from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Protocol

from openai import AzureOpenAI, OpenAI
from sentence_transformers import SentenceTransformer


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str], *, input_type: str = "document") -> list[list[float]]:
        raise NotImplementedError


def _apply_prompt(texts: list[str], *, input_type: str, prompt_style: str) -> list[str]:
    if prompt_style.strip().lower() != "e5":
        return texts
    prefix = "query: " if input_type == "query" else "passage: "
    return [f"{prefix}{text}" for text in texts]


@dataclass
class OpenAIEmbeddingClient:
    api_key: str
    model: str
    base_url: str
    prompt_style: str = "none"
    request_timeout_seconds: float = 90.0
    max_retries: int = 2
    retry_base_delay_seconds: float = 1.0
    retry_max_delay_seconds: float = 8.0

    def __post_init__(self) -> None:
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        text = str(exc).strip().lower()
        retryable_markers = (
            "timeout",
            "timed out",
            "rate limit",
            "429",
            "connection",
            "temporarily unavailable",
            "service unavailable",
            "internal server error",
            "502",
            "503",
            "504",
        )
        return any(marker in text for marker in retryable_markers)

    def embed(self, texts: list[str], *, input_type: str = "document") -> list[list[float]]:
        prepared = _apply_prompt(
            texts,
            input_type=input_type,
            prompt_style=self.prompt_style,
        )
        for attempt in range(1, self.max_retries + 2):
            try:
                response = self._client.embeddings.create(
                    model=self.model,
                    input=prepared,
                    encoding_format="float",
                    timeout=self.request_timeout_seconds,
                )
                return [item.embedding for item in response.data]
            except Exception as exc:
                should_retry = self._is_retryable_error(exc) and attempt <= self.max_retries
                if not should_retry:
                    raise
                delay = min(
                    self.retry_max_delay_seconds,
                    self.retry_base_delay_seconds * (2 ** (attempt - 1)),
                )
                logging.warning(
                    "OpenAI embedding request failed (attempt %s/%s): %s. Retrying in %.1fs",
                    attempt,
                    self.max_retries + 1,
                    str(exc),
                    delay,
                )
                time.sleep(delay)
        raise RuntimeError("Unexpected embedding retry flow.")


@dataclass
class AzureOpenAIEmbeddingClient:
    api_key: str
    endpoint: str
    api_version: str
    deployment: str
    prompt_style: str = "none"
    request_timeout_seconds: float = 90.0
    max_retries: int = 2
    retry_base_delay_seconds: float = 1.0
    retry_max_delay_seconds: float = 8.0

    def __post_init__(self) -> None:
        self._client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version=self.api_version,
        )

    def embed(self, texts: list[str], *, input_type: str = "document") -> list[list[float]]:
        prepared = _apply_prompt(
            texts,
            input_type=input_type,
            prompt_style=self.prompt_style,
        )
        for attempt in range(1, self.max_retries + 2):
            try:
                response = self._client.embeddings.create(
                    model=self.deployment,
                    input=prepared,
                    timeout=self.request_timeout_seconds,
                )
                return [item.embedding for item in response.data]
            except Exception as exc:
                should_retry = OpenAIEmbeddingClient._is_retryable_error(
                    exc
                ) and attempt <= self.max_retries
                if not should_retry:
                    raise
                delay = min(
                    self.retry_max_delay_seconds,
                    self.retry_base_delay_seconds * (2 ** (attempt - 1)),
                )
                logging.warning(
                    "Azure embedding request failed (attempt %s/%s): %s. Retrying in %.1fs",
                    attempt,
                    self.max_retries + 1,
                    str(exc),
                    delay,
                )
                time.sleep(delay)
        raise RuntimeError("Unexpected embedding retry flow.")


@dataclass
class LocalEmbeddingClient:
    model_name: str
    normalize: bool = True
    prompt_style: str = "e5"
    device: str | None = None
    batch_size: int = 64

    def __post_init__(self) -> None:
        self._model = SentenceTransformer(self.model_name, device=self.device)

    def _apply_prompt(self, texts: list[str], input_type: str) -> list[str]:
        return _apply_prompt(texts, input_type=input_type, prompt_style=self.prompt_style)

    def embed(self, texts: list[str], *, input_type: str = "document") -> list[list[float]]:
        prepared = self._apply_prompt(texts, input_type)
        vectors = self._model.encode(
            prepared,
            normalize_embeddings=self.normalize,
            batch_size=self.batch_size,
            show_progress_bar=True,
        )
        return [vector.tolist() for vector in vectors]


def build_embedding_client(
    provider: str,
    *,
    openai_api_key: str | None,
    openai_model: str,
    azure_endpoint: str | None,
    azure_api_key: str | None,
    azure_api_version: str,
    azure_deployment: str | None,
    local_model: str,
    local_normalize: bool,
    local_prompt_style: str,
    local_device: str | None,
    deepinfra_base_url: str,
) -> EmbeddingClient:
    provider_normalized = provider.strip().lower()
    if provider_normalized == "openai":
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
        return OpenAIEmbeddingClient(
            api_key=openai_api_key,
            model=openai_model,
            base_url=deepinfra_base_url,
            prompt_style="none",
        )
    if provider_normalized == "azure":
        if not (azure_endpoint and azure_api_key and azure_deployment):
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and "
                "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT are required for Azure embeddings."
            )
        return AzureOpenAIEmbeddingClient(
            api_key=azure_api_key,
            endpoint=azure_endpoint,
            api_version=azure_api_version,
            deployment=azure_deployment,
            prompt_style="none",
        )
    if provider_normalized in {"local", "hf", "huggingface"}:
        return LocalEmbeddingClient(
            model_name=local_model,
            normalize=local_normalize,
            prompt_style=local_prompt_style,
            device=local_device,
        )
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")
