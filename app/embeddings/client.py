from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from openai import AzureOpenAI, OpenAI
from sentence_transformers import SentenceTransformer


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str], *, input_type: str = "document") -> list[list[float]]:
        raise NotImplementedError


@dataclass
class OpenAIEmbeddingClient:
    api_key: str
    model: str

    def __post_init__(self) -> None:
        self._client = OpenAI(api_key=self.api_key)

    def embed(self, texts: list[str], *, input_type: str = "document") -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


@dataclass
class AzureOpenAIEmbeddingClient:
    api_key: str
    endpoint: str
    api_version: str
    deployment: str

    def __post_init__(self) -> None:
        self._client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version=self.api_version,
        )

    def embed(self, texts: list[str], *, input_type: str = "document") -> list[list[float]]:
        response = self._client.embeddings.create(model=self.deployment, input=texts)
        return [item.embedding for item in response.data]


@dataclass
class LocalEmbeddingClient:
    model_name: str
    normalize: bool = True
    prompt_style: str = "e5"
    device: str | None = None

    def __post_init__(self) -> None:
        self._model = SentenceTransformer(self.model_name, device=self.device)

    def _apply_prompt(self, texts: list[str], input_type: str) -> list[str]:
        if self.prompt_style.strip().lower() != "e5":
            return texts
        prefix = "query: " if input_type == "query" else "passage: "
        return [f"{prefix}{text}" for text in texts]

    def embed(self, texts: list[str], *, input_type: str = "document") -> list[list[float]]:
        prepared = self._apply_prompt(texts, input_type)
        vectors = self._model.encode(
            prepared,
            normalize_embeddings=self.normalize,
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
) -> EmbeddingClient:
    provider_normalized = provider.strip().lower()
    if provider_normalized == "openai":
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
        return OpenAIEmbeddingClient(api_key=openai_api_key, model=openai_model)
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
        )
    if provider_normalized in {"local", "hf", "huggingface"}:
        return LocalEmbeddingClient(
            model_name=local_model,
            normalize=local_normalize,
            prompt_style=local_prompt_style,
            device=local_device,
        )
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")
