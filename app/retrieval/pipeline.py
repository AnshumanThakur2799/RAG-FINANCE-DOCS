from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol
from urllib import error as urllib_error
from urllib import request as urllib_request

from app.db.sqlite_store import SQLiteDocumentStore
from app.db.vector_store import VectorStore
from app.llm.call_logger import log_llm_call

class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, dict[str, Any] | Any] | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError


class Reranker(Protocol):
    def rerank(self, *, query: str, documents: list[str]) -> list[float]:
        raise NotImplementedError


@dataclass(frozen=True)
class DeepInfraReranker:
    api_key: str
    base_url: str
    model: str
    instruction: str
    service_tier: str = "default"
    timeout_seconds: float = 30.0

    def rerank(self, *, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []

        payload: dict[str, Any] = {
            "queries": [query] * len(documents),
            "documents": documents,
        }
        if self.instruction.strip():
            payload["instruction"] = self.instruction.strip()
        if self.service_tier in {"default", "priority"}:
            payload["service_tier"] = self.service_tier

        body = json.dumps(payload).encode("utf-8")
        url = f"{self.base_url.rstrip('/')}/{self.model}"
        request = urllib_request.Request(
            url=url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib_request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib_error.URLError as exc:
            raise RuntimeError(f"DeepInfra reranker request failed: {exc}") from exc

        parsed = json.loads(raw)
        scores = parsed.get("scores")
        if not isinstance(scores, list) or len(scores) != len(documents):
            raise RuntimeError("DeepInfra reranker returned invalid scores.")
        return [float(score) for score in scores]


def _merge_ranked_unique(
    ranked_lists: list[list[dict[str, Any]]], *, limit: int
) -> list[dict[str, Any]]:
    target = max(1, limit)
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    max_length = max((len(items) for items in ranked_lists), default=0)

    for idx in range(max_length):
        for ranked in ranked_lists:
            if idx >= len(ranked):
                continue
            item = ranked[idx]
            chunk_key = str(item.get("id", "")).strip()
            if not chunk_key:
                source_path = str(item.get("source_path", "")).strip()
                source_name = str(item.get("source_name", "")).strip()
                chunk_id = str(item.get("chunk_id", "")).strip()
                if source_path and chunk_id:
                    chunk_key = f"{source_path}#{chunk_id}"
                elif source_name and chunk_id:
                    chunk_key = f"{source_name}#{chunk_id}"
            if not chunk_key or chunk_key in seen:
                continue
            seen.add(chunk_key)
            merged.append(dict(item))
            if len(merged) >= target:
                return merged
    return merged


def _apply_rerank(
    *,
    query: str,
    candidates: list[dict[str, Any]],
    reranker: Reranker,
    top_k: int,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    documents = [str(item.get("text", "")).strip() for item in candidates]
    scores = reranker.rerank(query=query, documents=documents)
    # include all the score greater than 0.95
    scored_items: list[dict[str, Any]] = []
    for item, score in zip(candidates, scores):
        enriched = dict(item)
        enriched["_reranker_score"] = float(score)
        if score > 0.95:
            scored_items.append(enriched)

    scored_items.sort(key=lambda item: float(item.get("_reranker_score", 0.0)), reverse=True)
    return scored_items


MULTI_QUERY_PROMPT_TEMPLATE = """

### TASK
Generate a minimum of <MIN> and a maximum of <MAX> paraphrased versions of the input query: "<USER_QUERY>".

### CONSTRAINTS & RULES
1. **Preserve Intent:** Do not change the core meaning. Do not hallucinate or add facts not present in the original query.
2. **Entity Protection:** Under no circumstances should you change or remove named entities (people, organizations, addresses, product IDs), dates, amounts, or specific locations.
3. **Anchor Phrase Rule (for short queries):** <ANCHOR_RULES>
4. **Linguistic Variance:** Use the requested language (<LANGUAGE>). Ensure variations include:
    - **Direct:** Clear and concise phrasing.
    - **Natural:** Conversational phrasing a typical user might type.
    - **Keyword:** Search-optimized, keyword-focused phrasing.
    - **Question Form:** A natural question variant when appropriate.
    - **Reordered Terms:** Same meaning with different term order.
5. **Structural Diversity:** Vary the use of stopwords (and, the, of), punctuation, and sentence length to cater to both vector-based and keyword-based search engines.

### OUTPUT FORMAT
- Return **ONLY** a valid JSON array of objects. 
- No markdown code blocks, no preamble, and no concluding remarks.
- Each object must follow this schema:
  {
    "query": "The paraphrased string",
    "variant_type": "One of: [exact, technical, layperson, action_oriented, conceptual, keyword_short, boolean_style, typo_robust]",
    "rationale": "One brief sentence explaining why this variant helps recall."
  }

### EXECUTION
Generate paraphrases for: "<USER_QUERY>"
Language: <LANGUAGE>
Target Count (N): <N> (Stay within range <MIN> to <MAX>)
"""


TENDER_ID_PATTERN = re.compile(r"\d{4}_[A-Za-z0-9]+_\d+_\d+")


METADATA_CONTEXT_KEYS: tuple[str, ...] = (
    "tender_id",
    "organization",
    "title",
    "publish_date",
    "bid_opening_date",
    "bid_submission_start",
    "bid_submission_end",
    "tender_url",
)


def _format_result_metadata(result: dict[str, Any]) -> str:
    values: list[str] = []
    for key in METADATA_CONTEXT_KEYS:
        value = result.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        values.append(f"{key}={text}")

    source_name = str(result.get("source_name", "")).strip()
    if source_name:
        values.append(f"source_name={source_name}")

    chunk_id = result.get("chunk_id")
    if chunk_id is not None and str(chunk_id).strip():
        values.append(f"chunk_id={chunk_id}")

    return " | ".join(values)


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
    def _wrap_context_block(identifier: str, body: str) -> str:
        return (
            f"===== BEGIN DOCUMENT: {identifier} =====\n"
            f"{body}\n"
            f"===== END DOCUMENT: {identifier} ====="
        )

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
                metadata_line = _format_result_metadata(result)
                if metadata_line:
                    context_lines.append(
                        _wrap_context_block(
                            f"{source}#{chunk_id}",
                            f"[{source}#{chunk_id}] META: {metadata_line}\n{text}",
                        )
                    )
                else:
                    context_lines.append(
                        _wrap_context_block(
                            f"{source}#{chunk_id}", f"[{source}#{chunk_id}] {text}"
                        )
                    )
        return "\n\n".join(context_lines), []

    context_lines: list[str] = []
    for tender_id in selected_tender_ids:
        representative = next(
            (item for item in results if extract_tender_id(item) == tender_id), None
        )
        metadata_line = (
            _format_result_metadata(representative) if representative is not None else ""
        )
        full_text = document_store.get_tender_full_text(tender_id).strip()
        if full_text:
            if metadata_line:
                context_lines.append(
                    _wrap_context_block(
                        tender_id,
                        f"[{tender_id}/full_text#full] META: {metadata_line}\n{full_text}",
                    )
                )
            else:
                context_lines.append(
                    _wrap_context_block(
                        tender_id, f"[{tender_id}/full_text#full]\n{full_text}"
                    )
                )
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
            chunk_metadata_line = _format_result_metadata(item)
            if chunk_metadata_line:
                chunk_lines.append(
                    f"[{source}#{chunk_id}] META: {chunk_metadata_line}\n{text}"
                )
            else:
                chunk_lines.append(f"[{source}#{chunk_id}] {text}")
        if chunk_lines:
            context_lines.append(
                _wrap_context_block(tender_id, "\n\n".join(chunk_lines))
            )

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
        with open("dense_results.json", "a") as f:
            json.dump({"query": query, "results": results, "length": len(results)}, f)
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
        with open("lexical_results.json", "a") as f:
            json.dump({"query": query, "results": results, "length": len(results)}, f)
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


def _extract_json_array(payload: str) -> list[dict[str, Any]]:
    text = (payload or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _is_short_query(query: str) -> bool:
    tokens = [token for token in re.split(r"\s+", query.strip()) if token]
    return len(tokens) <= 4


def _normalize_for_anchor(text: str) -> str:
    lowered = text.casefold()
    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def _contains_anchor_phrase(candidate: str, anchor: str) -> bool:
    anchor_norm = _normalize_for_anchor(anchor)
    candidate_norm = _normalize_for_anchor(candidate)
    if not anchor_norm or not candidate_norm:
        return False
    return anchor_norm in candidate_norm


def _build_short_query_variants(query: str, target_count: int) -> list[str]:
    anchor = query.strip()
    templates = [
        "{anchor}",
        "information about {anchor}",
        "{anchor} details",
        "{anchor} overview",
        "{anchor} meaning",
        "{anchor} use cases",
        "{anchor} examples",
        "guide to {anchor}",
        "best practices for {anchor}",
    ]
    variants: list[str] = []
    seen: set[str] = set()
    for template in templates:
        candidate = template.format(anchor=anchor).strip()
        key = candidate.casefold()
        if key in seen:
            continue
        seen.add(key)
        variants.append(candidate)
        if len(variants) >= target_count:
            break
    return variants


def _generate_query_variants(
    *,
    llm_client: Any,
    query: str,
    count: int,
    language: str,
) -> list[str]:
    target_count = max(1, int(count))
    short_query = _is_short_query(query)
    anchor_rules = (
        f"Input is a short query. Every generated query MUST include the exact anchor phrase '{query}' unchanged. "
        "Build wording around it, but do not replace that phrase."
        if short_query
        else "Not a short query. Natural paraphrasing is allowed while preserving intent and protected entities."
    )
    prompt = (
        MULTI_QUERY_PROMPT_TEMPLATE.replace("<USER_QUERY>", query)
        .replace("<N>", str(target_count))
        .replace("<MIN>", str(target_count))
        .replace("<MAX>", str(target_count))
        .replace("<LANGUAGE>", language)
        .replace("<ANCHOR_RULES>", anchor_rules)
    )
    system_prompt = "You are an expert Search Engineer and Query Expansion Assistant. Your goal is to transform a single user query into a set of diverse, high-recall search variations to improve document retrieval in a RAG or search pipeline."
    log_llm_call(
        source="retrieval.pipeline",
        operation="chat.query_variants",
        status="started",
        query=query,
        details={"target_count": target_count, "language": language},
    )
    try:
        response = llm_client.chat(system_prompt, prompt)
        log_llm_call(
            source="retrieval.pipeline",
            operation="chat.query_variants",
            status="succeeded",
            query=query,
            details={"target_count": target_count, "language": language},
        )
    except Exception as exc:
        log_llm_call(
            source="retrieval.pipeline",
            operation="chat.query_variants",
            status="failed",
            query=query,
            details={"target_count": target_count, "language": language},
            error=str(exc),
        )
        raise
    rows = _extract_json_array(response)

    unique: list[str] = []
    seen: set[str] = set()
    for row in rows:
        candidate = str(row.get("query", "")).strip()
        if not candidate:
            continue
        if short_query and not _contains_anchor_phrase(candidate, query):
            continue
        key = candidate.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
        if len(unique) >= target_count:
            break
    if short_query and len(unique) < target_count:
        for candidate in _build_short_query_variants(query, target_count):
            key = candidate.casefold()
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
            if len(unique) >= target_count:
                break
    return unique


@dataclass(frozen=True)
class HybridRetriever:
    dense_retriever: DenseVectorRetriever
    lexical_retriever: LexicalFTSRetriever
    rrf_k: int = 60
    candidate_multiplier: int = 4
    reranker: Reranker | None = None
    reranker_top_k_multiplier: int = 4

    def retrieve_ranked_lists(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, dict[str, Any] | Any] | None = None,
    ) -> list[list[dict[str, Any]]]:
        candidate_k = max(1, top_k) * max(1, self.candidate_multiplier)
        dense_results = self.dense_retriever.retrieve(
            query, top_k=candidate_k, filters=filters
        )
        lexical_results = self.lexical_retriever.retrieve(
            query, top_k=candidate_k, filters=filters
        )
        with open("dense_results.json", "a") as f:
            json.dump({"query": query, "length": len(dense_results), "results": dense_results}, f)
        with open("lexical_results.json", "a") as f:
            json.dump({"query": query, "results": lexical_results, "length": len(lexical_results)}, f)

        return [dense_results, lexical_results]

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, dict[str, Any] | Any] | None = None,
    ) -> list[dict[str, Any]]:
        dense_results, lexical_results = self.retrieve_ranked_lists(
            query, top_k=top_k, filters=filters
        )
        # write into json file overwrite along with length of results
        with open("dense_results.json", "a") as f:
            json.dump({"query": query, "length": len(dense_results), "results": dense_results}, f)
        with open("lexical_results.json", "a") as f:
            json.dump({"query": query, "results": lexical_results, "length": len(lexical_results)}, f)
        if self.reranker is not None:
            reranker_candidates = _merge_ranked_unique(
                [dense_results, lexical_results],
                limit=max(1, top_k) * max(1, self.reranker_top_k_multiplier),
            )

            with open("reranker_candidates.json", "a") as f:
                json.dump({"query": query, "results": reranker_candidates, "length": len(reranker_candidates)}, f)
            try:
                fused = _apply_rerank(
                    query=query,
                    candidates=reranker_candidates,
                    reranker=self.reranker,
                    top_k=max(1, top_k),
                )
                with open("reranked_results.json", "a") as f:
                    json.dump(
                        {
                            "query": query,
                            "source": "hybrid",
                            "candidate_length": len(reranker_candidates),
                            "results": fused,
                            "length": len(fused),
                        },
                        f,
                    )
            except Exception:
                fused = reciprocal_rank_fusion(
                    [dense_results, lexical_results],
                    top_k=max(1, top_k),
                    rrf_k=max(1, self.rrf_k),
                )
        else:
            fused = reciprocal_rank_fusion(
                [dense_results, lexical_results],
                top_k=max(1, top_k),
                rrf_k=max(1, self.rrf_k),
            )

        with open("fused_results.json", "a") as f:
            json.dump({"query": query, "results": fused, "length": len(fused)}, f)
        if not filters:
            return fused
        return [item for item in fused if _matches_filters(item, filters)]


@dataclass(frozen=True)
class MultiQueryRetriever:
    base_retriever: Retriever
    llm_client: Any | None = None
    query_count: int = 3
    query_language: str = "English"
    rrf_k: int = 60
    reranker: Reranker | None = None
    reranker_top_k_multiplier: int = 4

    def _build_queries(self, query: str) -> list[str]:
        normalized = (query or "").strip()
        if not normalized:
            return []

        target = max(1, self.query_count)
        all_queries: list[str] = [normalized]
        seen = {normalized.casefold()}
        debug_payload: dict[str, Any] = {
            "query": normalized,
            "target": target,
            "results": [],
            "length": 0,
            "status": "skipped",
            "reason": "",
        }

        if self.llm_client is not None and target > 1:
            try:
                generated = _generate_query_variants(
                    llm_client=self.llm_client,
                    query=normalized,
                    count=target,
                    language=self.query_language,
                )
                debug_payload.update(
                    {
                        "status": "ok",
                        "reason": "generated",
                        "results": generated,
                        "length": len(generated),
                    }
                )

                for candidate in generated:
                    key = candidate.casefold()
                    if key in seen:
                        continue
                    seen.add(key)
                    all_queries.append(candidate)
                    if len(all_queries) >= target:
                        break
            except Exception as exc:
                # Fallback to base query when paraphrase generation fails.
                debug_payload.update(
                    {
                        "status": "error",
                        "reason": "llm_query_generation_failed",
                        "error": str(exc),
                    }
                )
        elif self.llm_client is None:
            debug_payload.update(
                {
                    "status": "skipped",
                    "reason": "llm_client_unavailable",
                }
            )
        else:
            debug_payload.update(
                {
                    "status": "skipped",
                    "reason": "multi_query_count_is_one",
                }
            )

        debug_payload["final_queries"] = all_queries[:target]
        try:
            with open("generated_queries.json", "w") as f:
                json.dump(debug_payload, f)
        except Exception:
            # Query generation diagnostics should not break retrieval.
            pass

        return all_queries[:target]

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, dict[str, Any] | Any] | None = None,
    ) -> list[dict[str, Any]]:
        queries = self._build_queries(query)
        if not queries:
            return []
        if len(queries) == 1:
            return self.base_retriever.retrieve(query, top_k=top_k, filters=filters)

        ranked_lists: list[list[dict[str, Any]]] = []
        if isinstance(self.base_retriever, HybridRetriever):
            with ThreadPoolExecutor(max_workers=len(queries)) as executor:
                futures = [
                    executor.submit(
                        self.base_retriever.retrieve_ranked_lists,
                        variant,
                        top_k=top_k,
                        filters=filters,
                    )
                    for variant in queries
                ]
                for future in futures:
                    ranked_lists.extend(future.result())
        else:
            with ThreadPoolExecutor(max_workers=len(queries)) as executor:
                futures = [
                    executor.submit(
                        self.base_retriever.retrieve,
                        variant,
                        top_k=top_k * max(1,self.reranker_top_k_multiplier),
                        filters=filters,
                    )
                    for variant in queries
                ]
                ranked_lists = [future.result() for future in futures]
        if self.reranker is not None:
            reranker_candidates = _merge_ranked_unique(
                ranked_lists,
                limit=max(1, top_k) * max(1, self.reranker_top_k_multiplier),
            )
            with open("reranker_candidates.json", "a") as f:
                json.dump({"query": query, "results": reranker_candidates, "length": len(reranker_candidates)}, f)
            try:
                fused = _apply_rerank(
                    query=query,
                    candidates=reranker_candidates,
                    reranker=self.reranker,
                    top_k=max(1, top_k),
                )
                with open("reranked_results.json", "a") as f:
                    json.dump(
                        {
                            "query": query,
                            "source": "multi_query",
                            "candidate_length": len(reranker_candidates),
                            "results": fused,
                            "length": len(fused),
                        },
                        f,
                    )
            except Exception:
                fused = reciprocal_rank_fusion(
                    ranked_lists,
                    top_k=max(1, top_k),
                    rrf_k=max(1, self.rrf_k),
                )
        else:
            fused = reciprocal_rank_fusion(
                ranked_lists,
                top_k=max(1, top_k),
                rrf_k=max(1, self.rrf_k),
            )
        with open("fused_results.json", "a") as f:
            json.dump({"query": query, "results": fused, "length": len(fused)}, f)
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
    deepinfra_api_key: str | None = None,
    reranker_enabled: bool = True,
    reranker_model: str = "Qwen/Qwen3-Reranker-4B",
    reranker_instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
    reranker_service_tier: str = "default",
    reranker_base_url: str = "https://api.deepinfra.com/v1/inference",
    reranker_top_k_multiplier: int = 4,
    llm_client: Any | None = None,
    multi_query_enabled: bool = False,
    multi_query_count: int = 3,
    multi_query_language: str = "English",
) -> Retriever:
    normalized_mode = mode.strip().lower()
    dense = DenseVectorRetriever(
        embed_client=embed_client,
        vector_store=vector_store,
        document_store=document_store,
    )
    lexical = LexicalFTSRetriever(document_store=document_store)
    reranker: Reranker | None = None
    if reranker_enabled and deepinfra_api_key:
        reranker = DeepInfraReranker(
            api_key=deepinfra_api_key,
            base_url=reranker_base_url,
            model=reranker_model,
            instruction=reranker_instruction,
            service_tier=reranker_service_tier,
        )

    if normalized_mode == "dense":
        base: Retriever = dense
    elif normalized_mode == "lexical":
        base = lexical
    elif normalized_mode == "hybrid":
        base = HybridRetriever(
            dense_retriever=dense,
            lexical_retriever=lexical,
            rrf_k=hybrid_rrf_k,
            candidate_multiplier=hybrid_candidate_multiplier,
            reranker=reranker,
            reranker_top_k_multiplier=reranker_top_k_multiplier,
        )
    else:
        raise ValueError(
            f"Unsupported retrieval mode '{mode}'. Expected one of: {', '.join(RETRIEVAL_MODES)}"
        )

    if not multi_query_enabled:
        return base
    return MultiQueryRetriever(
        base_retriever=base,
        llm_client=llm_client,
        query_count=max(1, multi_query_count),
        query_language=multi_query_language.strip() or "English",
        rrf_k=max(1, hybrid_rrf_k),
        reranker=reranker,
        reranker_top_k_multiplier=reranker_top_k_multiplier,
    )