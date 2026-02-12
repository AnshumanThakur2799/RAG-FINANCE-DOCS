from __future__ import annotations

import argparse
import csv
import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.db.sqlite_store import SQLiteDocumentStore
from app.db.vector_store import build_vector_store
from app.embeddings.client import build_embedding_client
from app.retrieval import RETRIEVAL_MODES, build_retriever


def canonicalize_column_name(name: str) -> str:
    return name.replace("\ufeff", "").strip().lower()


def resolve_column_name(requested: str, available: list[str]) -> str:
    requested_key = canonicalize_column_name(requested)
    canonical_map = {canonicalize_column_name(col): col for col in available}
    if requested_key in canonical_map:
        return canonical_map[requested_key]
    raise KeyError(
        f"Column '{requested}' was not found. Available columns: {', '.join(available)}"
    )


def normalize_doc_name(value: str) -> str:
    cleaned = value.strip().strip('"').strip("'")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.lower()


def parse_k_values(raw: str) -> list[int]:
    values: list[int] = []
    for piece in raw.split(","):
        item = piece.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed <= 0:
            raise ValueError("--k-values must contain positive integers only.")
        values.append(parsed)
    if not values:
        raise ValueError("--k-values cannot be empty.")
    return sorted(set(values))


def parse_source_docs_patterns(raw: str) -> list[str]:
    if not raw:
        return []
    patterns = []
    for token in raw.split(","):
        cleaned = token.strip().strip('"').strip("'")
        if cleaned:
            patterns.append(cleaned)
    return patterns


def extract_pdf_names(raw: str) -> set[str]:
    if not raw:
        return set()
    matches = re.findall(r"([A-Za-z0-9][A-Za-z0-9 _\-\(\)\.]*\.pdf)", raw, flags=re.I)
    return {normalize_doc_name(match) for match in matches}


def pattern_matches_doc(pattern: str, doc_name: str) -> bool:
    normalized_doc = normalize_doc_name(doc_name)
    normalized_pattern = normalize_doc_name(pattern)
    if not normalized_pattern:
        return False

    # Wildcard-friendly matching: *AAPL* -> contains "aapl"
    if "*" in normalized_pattern:
        escaped = re.escape(normalized_pattern).replace(r"\*", ".*")
        return re.fullmatch(escaped, normalized_doc) is not None

    # If a precise filename is provided, require exact match.
    if normalized_pattern.endswith(".pdf"):
        return normalized_doc == normalized_pattern

    # Label-like hints (e.g., AAPL) match by containment.
    return normalized_pattern in normalized_doc


def reciprocal_rank(relevance: list[int], k: int) -> float:
    limit = min(k, len(relevance))
    for idx in range(limit):
        if relevance[idx] == 1:
            return 1.0 / float(idx + 1)
    return 0.0


def precision_at_k(relevance: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    cutoff = relevance[:k]
    return sum(cutoff) / float(k)


def recall_at_k(relevance: list[int], k: int, total_relevant: int) -> float:
    if total_relevant <= 0:
        return 0.0
    return sum(relevance[:k]) / float(total_relevant)


def ndcg_at_k(relevance: list[int], k: int, total_relevant: int) -> float:
    limit = min(k, len(relevance))
    dcg = 0.0
    for idx in range(limit):
        rel = relevance[idx]
        if rel > 0:
            dcg += rel / math.log2(idx + 2)

    ideal_hits = min(total_relevant, k)
    if ideal_hits <= 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


@dataclass(frozen=True)
class QueryMetrics:
    embedding_time_seconds: float
    search_time_seconds: float
    total_time_seconds: float
    retrieved_docs: list[str]
    relevance: list[int]
    total_relevant: int


def evaluate_query(
    question: str,
    *,
    top_k: int,
    retriever,
    relevant_doc_names: set[str],
    relevant_patterns: list[str],
) -> QueryMetrics:
    started_at = time.perf_counter()
    search_start = time.perf_counter()
    results = retriever.retrieve(question, top_k=max(1, top_k))
    search_elapsed = time.perf_counter() - search_start

    retrieved_docs: list[str] = []
    relevance: list[int] = []
    normalized_relevant_docs = {normalize_doc_name(doc) for doc in relevant_doc_names}

    for result in results:
        source = str(result.get("source_name") or result.get("source_path", "unknown"))
        retrieved_docs.append(source)

        normalized_source = normalize_doc_name(source)
        is_relevant = normalized_source in normalized_relevant_docs
        if not is_relevant and relevant_patterns:
            is_relevant = any(
                pattern_matches_doc(pattern, source) for pattern in relevant_patterns
            )
        relevance.append(1 if is_relevant else 0)

    total_relevant = len(normalized_relevant_docs)
    if total_relevant == 0 and relevant_patterns:
        total_relevant = len(relevant_patterns)

    return QueryMetrics(
        embedding_time_seconds=0.0,
        search_time_seconds=search_elapsed,
        total_time_seconds=time.perf_counter() - started_at,
        retrieved_docs=retrieved_docs,
        relevance=relevance,
        total_relevant=total_relevant,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark retrieval quality and speed for vector search."
    )
    parser.add_argument(
        "--input",
        default="qna_data.csv",
        help="Input CSV with question and source labels.",
    )
    parser.add_argument(
        "--output",
        default="retrieval_benchmark_results.csv",
        help="Output CSV with per-query retrieval metrics.",
    )
    parser.add_argument(
        "--table",
        default="document_chunks",
        help="Vector collection/table name.",
    )
    parser.add_argument(
        "--question-column",
        default="Question",
        help="Question column name.",
    )
    parser.add_argument(
        "--source-docs-column",
        default="Source Docs",
        help="Column containing expected source documents or labels.",
    )
    parser.add_argument(
        "--answer-column",
        default="Answer",
        help="Answer column used to extract explicit PDF citations when available.",
    )
    parser.add_argument(
        "--k-values",
        default="1,3,5,10",
        help="Comma-separated K values, e.g. 1,3,5,10.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for number of rows to evaluate (0 = all).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--retrieval-mode",
        default="",
        help=f"Retrieval mode: {', '.join(RETRIEVAL_MODES)}.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    k_values = parse_k_values(args.k_values)
    max_k = max(k_values)

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    settings = Settings.from_env()
    embed_client = build_embedding_client(
        settings.embedding_provider,
        openai_api_key=settings.openai_api_key,
        openai_model=settings.openai_embedding_model,
        azure_endpoint=settings.azure_openai_endpoint,
        azure_api_key=settings.azure_openai_api_key,
        azure_api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_embeddings_deployment,
        local_model=settings.local_embedding_model,
        local_normalize=settings.local_embedding_normalize,
        local_prompt_style=settings.local_embedding_prompt_style,
        local_device=settings.local_embedding_device,
    )
    document_store = SQLiteDocumentStore(settings.sqlite_db_path)
    vector_store = build_vector_store(settings, table_name=args.table)
    retrieval_mode = args.retrieval_mode.strip().lower() or settings.retrieval_mode
    retriever = build_retriever(
        mode=retrieval_mode,
        embed_client=embed_client,
        vector_store=vector_store,
        document_store=document_store,
        hybrid_rrf_k=settings.hybrid_rrf_k,
        hybrid_candidate_multiplier=settings.hybrid_candidate_multiplier,
    )

    with input_path.open("r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not fieldnames:
        raise ValueError("Input CSV does not have a header row.")
    if not rows:
        raise ValueError("Input CSV has no rows to evaluate.")
    if args.limit > 0:
        rows = rows[: args.limit]

    question_column = resolve_column_name(args.question_column, fieldnames)
    source_docs_column = resolve_column_name(args.source_docs_column, fieldnames)
    answer_column = resolve_column_name(args.answer_column, fieldnames)

    output_fields = list(rows[0].keys()) + [
        "ground_truth_docs",
        "ground_truth_patterns",
        "retrieved_docs",
        "embedding_time_seconds",
        "search_time_seconds",
        "total_time_seconds",
    ]
    for k in k_values:
        output_fields.extend(
            [
                f"recall@{k}",
                f"precision@{k}",
                f"mrr@{k}",
                f"ndcg@{k}",
            ]
        )

    aggregates: dict[str, float] = {field: 0.0 for field in output_fields if "@" in field}
    aggregate_time = {
        "embedding_time_seconds": 0.0,
        "search_time_seconds": 0.0,
        "total_time_seconds": 0.0,
    }
    evaluated_count = 0
    output_rows: list[dict[str, str]] = []

    run_start = time.perf_counter()

    for idx, row in enumerate(rows, start=1):
        question = (row.get(question_column) or "").strip()
        if not question:
            logging.warning("Skipping row %s because question is empty.", idx)
            continue

        source_docs_value = (row.get(source_docs_column) or "").strip()
        answer_value = (row.get(answer_column) or "").strip()

        explicit_docs = extract_pdf_names(answer_value)
        source_patterns = parse_source_docs_patterns(source_docs_value)
        source_docs_as_exact = {
            normalize_doc_name(item) for item in source_patterns if item.lower().endswith(".pdf")
        }

        ground_truth_docs = explicit_docs or source_docs_as_exact

        metrics = evaluate_query(
            question,
            top_k=max_k,
            retriever=retriever,
            relevant_doc_names=ground_truth_docs,
            relevant_patterns=source_patterns,
        )

        out_row = dict(row)
        out_row["ground_truth_docs"] = "; ".join(sorted(ground_truth_docs))
        out_row["ground_truth_patterns"] = "; ".join(source_patterns)
        out_row["retrieved_docs"] = "; ".join(metrics.retrieved_docs)
        out_row["embedding_time_seconds"] = f"{metrics.embedding_time_seconds:.4f}"
        out_row["search_time_seconds"] = f"{metrics.search_time_seconds:.4f}"
        out_row["total_time_seconds"] = f"{metrics.total_time_seconds:.4f}"

        for k in k_values:
            recall_value = recall_at_k(metrics.relevance, k, metrics.total_relevant)
            precision_value = precision_at_k(metrics.relevance, k)
            mrr_value = reciprocal_rank(metrics.relevance, k)
            ndcg_value = ndcg_at_k(metrics.relevance, k, metrics.total_relevant)

            out_row[f"recall@{k}"] = f"{recall_value:.4f}"
            out_row[f"precision@{k}"] = f"{precision_value:.4f}"
            out_row[f"mrr@{k}"] = f"{mrr_value:.4f}"
            out_row[f"ndcg@{k}"] = f"{ndcg_value:.4f}"

            aggregates[f"recall@{k}"] += recall_value
            aggregates[f"precision@{k}"] += precision_value
            aggregates[f"mrr@{k}"] += mrr_value
            aggregates[f"ndcg@{k}"] += ndcg_value

        aggregate_time["embedding_time_seconds"] += metrics.embedding_time_seconds
        aggregate_time["search_time_seconds"] += metrics.search_time_seconds
        aggregate_time["total_time_seconds"] += metrics.total_time_seconds

        output_rows.append(out_row)
        evaluated_count += 1

        logging.info(
            "[%s/%s] Retrieved %s docs | search=%.3fs",
            idx,
            len(rows),
            len(metrics.retrieved_docs),
            metrics.search_time_seconds,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    elapsed = time.perf_counter() - run_start
    if evaluated_count == 0:
        logging.warning("No rows were evaluated.")
        return

    summary_parts = [
        f"rows={evaluated_count}",
        f"avg_embedding_s={aggregate_time['embedding_time_seconds'] / evaluated_count:.4f}",
        f"avg_search_s={aggregate_time['search_time_seconds'] / evaluated_count:.4f}",
        f"avg_total_s={aggregate_time['total_time_seconds'] / evaluated_count:.4f}",
    ]
    for k in k_values:
        summary_parts.extend(
            [
                f"avg_recall@{k}={aggregates[f'recall@{k}'] / evaluated_count:.4f}",
                f"avg_precision@{k}={aggregates[f'precision@{k}'] / evaluated_count:.4f}",
                f"avg_mrr@{k}={aggregates[f'mrr@{k}'] / evaluated_count:.4f}",
                f"avg_ndcg@{k}={aggregates[f'ndcg@{k}'] / evaluated_count:.4f}",
            ]
        )

    logging.info(
        "Retrieval benchmark complete | input=%s | output=%s | retrieval_mode=%s | runtime_s=%.2f | %s",
        str(input_path.resolve()),
        str(output_path.resolve()),
        retrieval_mode,
        elapsed,
        " | ".join(summary_parts),
    )


if __name__ == "__main__":
    main()

