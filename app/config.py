from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    embedding_provider: str
    openai_api_key: str | None
    openai_embedding_model: str
    azure_openai_endpoint: str | None
    azure_openai_api_key: str | None
    azure_openai_api_version: str
    azure_openai_embeddings_deployment: str | None
    local_embedding_model: str
    local_embedding_normalize: bool
    local_embedding_prompt_style: str
    local_embedding_device: str | None
    llm_provider: str
    deepinfra_api_key: str | None
    deepinfra_base_url: str
    deepinfra_model: str
    llm_max_tokens: int | None
    llm_temperature: float
    sqlite_db_path: Path
    vector_db_provider: str
    lancedb_dir: Path
    qdrant_url: str
    qdrant_api_key: str | None
    qdrant_timeout_seconds: float
    qdrant_prefer_grpc: bool
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    retrieval_mode: str
    hybrid_rrf_k: int
    hybrid_candidate_multiplier: int
    multi_query_enabled: bool
    multi_query_count: int
    multi_query_language: str
    reranker_enabled: bool
    reranker_model: str
    reranker_instruction: str
    reranker_top_k_multiplier: int
    reranker_service_tier: str
    reranker_base_url: str

    @staticmethod
    def from_env() -> "Settings":
        base_dir = Path(os.getenv("POC_BASE_DIR", Path.cwd()))
        sqlite_default = base_dir / "data" / "metadata.db"
        lancedb_default = base_dir / "data" / "lancedb"

        return Settings(
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "openai"),
            openai_api_key=(
                os.getenv("OPENAI_API_KEY")
                or os.getenv("DEEPINFRA_TOKEN")
                or os.getenv("DEEPINFRA_API_KEY")
            ),
            openai_embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B"
            ),
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_openai_api_version=os.getenv(
                "AZURE_OPENAI_API_VERSION", "2024-02-15-preview"
            ),
            azure_openai_embeddings_deployment=os.getenv(
                "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"
            ),
            local_embedding_model=os.getenv(
                "LOCAL_EMBEDDING_MODEL", "intfloat/e5-small-v2"
            ),
            local_embedding_normalize=_parse_bool(
                os.getenv("LOCAL_EMBEDDING_NORMALIZE"), True
            ),
            local_embedding_prompt_style=os.getenv(
                "LOCAL_EMBEDDING_PROMPT_STYLE", "none"
            ),
            local_embedding_device=os.getenv("LOCAL_EMBEDDING_DEVICE"),
            llm_provider=os.getenv("LLM_PROVIDER", "deepinfra"),
            deepinfra_api_key=os.getenv("DEEPINFRA_API_KEY") or os.getenv("DEEPINFRA_TOKEN"),
            deepinfra_base_url=os.getenv(
                "DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai"
            ),
            deepinfra_model=os.getenv(
                "DEEPINFRA_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"
            ),
            llm_max_tokens=os.getenv("LLM_MAX_TOKENS", "None") if os.getenv("LLM_MAX_TOKENS", "None") != "None" else None,
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            sqlite_db_path=Path(os.getenv("SQLITE_DB_PATH", sqlite_default)),
            vector_db_provider=os.getenv("VECTOR_DB_PROVIDER", "lancedb"),
            lancedb_dir=Path(os.getenv("LANCEDB_DIR", lancedb_default)),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            qdrant_timeout_seconds=float(os.getenv("QDRANT_TIMEOUT_SECONDS", "30")),
            qdrant_prefer_grpc=_parse_bool(
                os.getenv("QDRANT_PREFER_GRPC"), False
            ),
            chunk_size_tokens=int(os.getenv("CHUNK_SIZE_TOKENS", "500")),
            chunk_overlap_tokens=int(os.getenv("CHUNK_OVERLAP_TOKENS", "50")),
            retrieval_mode=os.getenv("RETRIEVAL_MODE", "hybrid"),
            hybrid_rrf_k=int(os.getenv("HYBRID_RRF_K", "60")),
            hybrid_candidate_multiplier=int(
                os.getenv("HYBRID_CANDIDATE_MULTIPLIER", "4")
            ),
            multi_query_enabled=_parse_bool(
                os.getenv("MULTI_QUERY_ENABLED"), True
            ),
            multi_query_count=max(1, int(os.getenv("MULTI_QUERY_COUNT", "3"))),
            multi_query_language=os.getenv("MULTI_QUERY_LANGUAGE", "English"),
            reranker_enabled=_parse_bool(os.getenv("RERANKER_ENABLED"), True),
            reranker_model=os.getenv(
                "RERANKER_MODEL", "Qwen/Qwen3-Reranker-4B"
            ).strip()
            or "Qwen/Qwen3-Reranker-4B",
            reranker_instruction=os.getenv(
                "RERANKER_INSTRUCTION",
                "Given a web search query, retrieve relevant passages that answer the query",
            ),
            reranker_top_k_multiplier=max(
                1, int(os.getenv("RERANKER_TOP_K_MULTIPLIER", "4"))
            ),
            reranker_service_tier=os.getenv(
                "RERANKER_SERVICE_TIER", "default"
            ).strip()
            or "default",
            reranker_base_url=os.getenv(
                "DEEPINFRA_RERANKER_BASE_URL",
                "https://api.deepinfra.com/v1/inference",
            ).rstrip("/"),
        )
