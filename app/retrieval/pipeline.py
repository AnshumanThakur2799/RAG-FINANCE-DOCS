from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from app.db.sqlite_store import SQLiteDocumentStore
from app.db.vector_store import VectorStore

import json 

class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, dict[str, Any] | Any] | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError


TENDER_ID_PATTERN = re.compile(r"\d{4}_[A-Za-z0-9]+_\d+_\d+")


def extract_tender_id(result: dict[str, Any]) -> str | None:
    tender_id = result.get("tender_id")
    if isinstance(tender_id, str) and tender_id.strip():
        return tender_id.strip()
    source_path = result.get("source_path")
    if isinstance(source_path, str) and source_path:
        match = TENDER_ID_PATTERN.search(source_path)
        if match:
            return match.group(0)
    return None


def collect_top_unique_tender_ids(
    results: list[dict[str, Any]],
    *,
    max_unique_tenders: int = 3,
) -> list[str]:
    selected: list[str] = []
    for result in results:
        tender_id = extract_tender_id(result)
        if not tender_id:
            continue
        if tender_id in selected:
            continue
        selected.append(tender_id)
        if len(selected) >= max(1, max_unique_tenders):
            break
    return selected


def build_full_tender_context(
    *,
    results: list[dict[str, Any]],
    document_store: SQLiteDocumentStore,
    max_unique_tenders: int = 3,
) -> tuple[str, list[str]]:
    selected_tender_ids = collect_top_unique_tender_ids(
        results, max_unique_tenders=max_unique_tenders
    )
    if not selected_tender_ids:
        context_lines = []
        for result in results:
            source = result.get("source_name") or result.get("source_path", "unknown")
            chunk_id = result.get("chunk_id", "n/a")
            text = (result.get("text") or "").strip()
            if text:
                context_lines.append(f"[{source}#{chunk_id}] {text}")
        return "\n\n".join(context_lines), []

    context_lines: list[str] = []
    for tender_id in selected_tender_ids:
        full_text = document_store.get_tender_full_text(tender_id).strip()
        if full_text:
            context_lines.append(f"[{tender_id}/full_text#full]\n{full_text}")
            continue

        # Fallback for older indexed data where tender-level full text is unavailable.
        fallback_chunks = [
            item
            for item in results
            if extract_tender_id(item) == tender_id and (item.get("text") or "").strip()
        ]
        chunk_lines: list[str] = []
        for item in fallback_chunks:
            source = item.get("source_name") or item.get("source_path", "unknown")
            chunk_id = item.get("chunk_id", "n/a")
            text = (item.get("text") or "").strip()
            chunk_lines.append(f"[{source}#{chunk_id}] {text}")
        if chunk_lines:
            context_lines.append("\n\n".join(chunk_lines))

    return "\n\n".join(context_lines), selected_tender_ids


def _matches_filters(
    item: dict[str, Any], filters: dict[str, dict[str, Any] | Any] | None
) -> bool:
    if not filters:
        return True
    for field, raw_ops in filters.items():
        if isinstance(raw_ops, dict):
            ops = raw_ops
        else:
            ops = {"eq": raw_ops}
        item_value = item.get(field)
        if item_value is None:
            return False
        for op, expected in ops.items():
            if op == "eq" and item_value != expected:
                return False
            if op == "gt" and not (item_value > expected):
                return False
            if op == "gte" and not (item_value >= expected):
                return False
            if op == "lt" and not (item_value < expected):
                return False
            if op == "lte" and not (item_value <= expected):
                return False
    return True


@dataclass(frozen=True)
class DenseVectorRetriever:
    embed_client: Any
    vector_store: VectorStore
    document_store: SQLiteDocumentStore

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, dict[str, Any] | Any] | None = None,
    ) -> list[dict[str, Any]]:
        embedding = self.embed_client.embed([query], input_type="query")[0]
        results = self.vector_store.search(
            embedding,
            top_k=max(1, top_k),
            filters=filters,
        )
        return self.document_store.attach_chunk_metadata(results)


@dataclass(frozen=True)
class LexicalFTSRetriever:
    document_store: SQLiteDocumentStore

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, dict[str, Any] | Any] | None = None,
    ) -> list[dict[str, Any]]:
        results = self.document_store.search_chunks(query, top_k=max(1, top_k))
        results = self.document_store.attach_chunk_metadata(results)
        if not filters:
            return results
        return [item for item in results if _matches_filters(item, filters)]


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

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, dict[str, Any] | Any] | None = None,
    ) -> list[dict[str, Any]]:
        candidate_k = max(1, top_k) * max(1, self.candidate_multiplier)
        dense_results = self.dense_retriever.retrieve(
            query, top_k=candidate_k, filters=filters
        )

        # write into json file overwrite along with length of results  
        with open("dense_results.json", "w") as f:
            json.dump({"length": len(dense_results), "results": dense_results}, f)
        lexical_results = self.lexical_retriever.retrieve(
            query, top_k=candidate_k, filters=filters
        )
        with open("lexical_results.json", "w") as f:
            json.dump({"results": lexical_results, "length": len(lexical_results)}, f)
        fused = reciprocal_rank_fusion(
            [dense_results, lexical_results],
            top_k=max(1, top_k),
            rrf_k=max(1, self.rrf_k),
        )

        with open("fused_results.json", "w") as f:
            json.dump({"results": fused, "length": len(fused)}, f)
        if not filters:
            return fused
        return [item for item in fused if _matches_filters(item, filters)]


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
    dense = DenseVectorRetriever(
        embed_client=embed_client,
        vector_store=vector_store,
        document_store=document_store,
    )
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
