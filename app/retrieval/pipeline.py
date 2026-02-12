from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from app.db.sqlite_store import SQLiteDocumentStore
from app.db.vector_store import VectorStore


class Retriever(Protocol):
    def retrieve(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        raise NotImplementedError


@dataclass(frozen=True)
class DenseVectorRetriever:
    embed_client: Any
    vector_store: VectorStore

    def retrieve(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        embedding = self.embed_client.embed([query], input_type="query")[0]
        return self.vector_store.search(embedding, top_k=max(1, top_k))


@dataclass(frozen=True)
class LexicalFTSRetriever:
    document_store: SQLiteDocumentStore

    def retrieve(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        return self.document_store.search_chunks(query, top_k=max(1, top_k))


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    *,
    top_k: int,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    fused_scores: dict[str, float] = {}
    merged_items: dict[str, dict[str, Any]] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            chunk_id = str(item.get("id", ""))
            if not chunk_id:
                continue
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (
                1.0 / float(rrf_k + rank)
            )
            if chunk_id not in merged_items:
                merged_items[chunk_id] = dict(item)

    for chunk_id, score in fused_scores.items():
        merged_items[chunk_id]["_rrf_score"] = score

    ranked_items = sorted(
        merged_items.values(),
        key=lambda item: float(item.get("_rrf_score", 0.0)),
        reverse=True,
    )
    return ranked_items[: max(1, top_k)]


@dataclass(frozen=True)
class HybridRetriever:
    dense_retriever: DenseVectorRetriever
    lexical_retriever: LexicalFTSRetriever
    rrf_k: int = 60
    candidate_multiplier: int = 4

    def retrieve(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        candidate_k = max(1, top_k) * max(1, self.candidate_multiplier)
        dense_results = self.dense_retriever.retrieve(query, top_k=candidate_k)
        lexical_results = self.lexical_retriever.retrieve(query, top_k=candidate_k)
        return reciprocal_rank_fusion(
            [dense_results, lexical_results],
            top_k=max(1, top_k),
            rrf_k=max(1, self.rrf_k),
        )


RETRIEVAL_MODES = ("dense", "lexical", "hybrid")


def build_retriever(
    *,
    mode: str,
    embed_client: Any,
    vector_store: VectorStore,
    document_store: SQLiteDocumentStore,
    hybrid_rrf_k: int = 60,
    hybrid_candidate_multiplier: int = 4,
) -> Retriever:
    normalized_mode = mode.strip().lower()
    dense = DenseVectorRetriever(embed_client=embed_client, vector_store=vector_store)
    lexical = LexicalFTSRetriever(document_store=document_store)

    if normalized_mode == "dense":
        return dense
    if normalized_mode == "lexical":
        return lexical
    if normalized_mode == "hybrid":
        return HybridRetriever(
            dense_retriever=dense,
            lexical_retriever=lexical,
            rrf_k=hybrid_rrf_k,
            candidate_multiplier=hybrid_candidate_multiplier,
        )
    raise ValueError(
        f"Unsupported retrieval mode '{mode}'. Expected one of: {', '.join(RETRIEVAL_MODES)}"
    )
