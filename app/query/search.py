from __future__ import annotations

import argparse
from datetime import datetime, timezone

from app.config import Settings
from app.db.sqlite_store import SQLiteDocumentStore
from app.db.vector_store import build_vector_store
from app.embeddings.client import build_embedding_client
from app.retrieval import RETRIEVAL_MODES, build_retriever


def _to_epoch(value: str) -> int:
    parsed = datetime.strptime(value.strip(), "%d-%b-%Y %I:%M %p")
    return int(parsed.replace(tzinfo=timezone.utc).timestamp())


def _build_filters(args: argparse.Namespace) -> dict[str, dict[str, int | str]]:
    filters: dict[str, dict[str, int | str]] = {}

    if args.organization_eq:
        filters["organization"] = {"eq": args.organization_eq.strip()}
    if args.tender_id_eq:
        filters["tender_id"] = {"eq": args.tender_id_eq.strip()}

    if args.publish_date_after:
        filters.setdefault("publish_date_epoch", {})["gt"] = _to_epoch(
            args.publish_date_after
        )
    if args.publish_date_before:
        filters.setdefault("publish_date_epoch", {})["lt"] = _to_epoch(
            args.publish_date_before
        )
    if args.bid_opening_after:
        filters.setdefault("bid_opening_date_epoch", {})["gt"] = _to_epoch(
            args.bid_opening_after
        )
    if args.bid_opening_before:
        filters.setdefault("bid_opening_date_epoch", {})["lt"] = _to_epoch(
            args.bid_opening_before
        )
    if args.bid_close_after:
        filters.setdefault("bid_submission_end_epoch", {})["gt"] = _to_epoch(
            args.bid_close_after
        )
    if args.bid_close_before:
        filters.setdefault("bid_submission_end_epoch", {})["lt"] = _to_epoch(
            args.bid_close_before
        )

    return filters


def main() -> None:
    parser = argparse.ArgumentParser(description="Search vector store for matching chunks.")
    parser.add_argument("--query", required=True, help="Query text.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results.")
    parser.add_argument(
        "--table", default="document_chunks", help="Vector collection/table name."
    )
    parser.add_argument(
        "--retrieval-mode",
        default="",
        help=f"Retrieval mode: {', '.join(RETRIEVAL_MODES)}.",
    )
    parser.add_argument("--organization-eq", default="", help="Filter by organization.")
    parser.add_argument("--tender-id-eq", default="", help="Filter by tender ID.")
    parser.add_argument(
        "--publish-date-after",
        default="",
        help="Filter publish date greater than this datetime (e.g. 10-Feb-2026 10:00 AM).",
    )
    parser.add_argument(
        "--publish-date-before",
        default="",
        help="Filter publish date less than this datetime.",
    )
    parser.add_argument(
        "--bid-opening-after",
        default="",
        help="Filter bid opening date greater than this datetime.",
    )
    parser.add_argument(
        "--bid-opening-before",
        default="",
        help="Filter bid opening date less than this datetime.",
    )
    parser.add_argument(
        "--bid-close-after",
        default="",
        help="Filter bid close (submission end) date greater than this datetime.",
    )
    parser.add_argument(
        "--bid-close-before",
        default="",
        help="Filter bid close (submission end) date less than this datetime.",
    )
    args = parser.parse_args()

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
        deepinfra_base_url=settings.deepinfra_base_url,
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

    filters = _build_filters(args)
    results = retriever.retrieve(
        args.query,
        top_k=max(1, args.top_k),
        filters=filters or None,
    )

    if not results:
        print("No results found. Have you indexed PDFs yet?")
        return

    print("Top matches:")
    for idx, result in enumerate(results, start=1):
        source = result.get("source_name") or result.get("source_path", "unknown")
        chunk_id = result.get("chunk_id", "n/a")
        score = result.get(
            "_rrf_score",
            result.get("_distance", result.get("_lexical_score", result.get("score", "n/a"))),
        )
        if isinstance(score, float):
            score_display = f"{score:.4f}"
        else:
            score_display = str(score)
        text = (result.get("text") or "").replace("\n", " ").strip()
        if len(text) > 400:
            text = f"{text[:400]}..."
        print(f"{idx}. {source} (chunk {chunk_id}, score {score_display})")
        publish_date = result.get("publish_date")
        bid_close_date = result.get("bid_submission_end")
        bid_opening_date = result.get("bid_opening_date")
        if publish_date or bid_close_date or bid_opening_date:
            print(
                f"   publish={publish_date or 'n/a'} | bid_close={bid_close_date or 'n/a'} | opening={bid_opening_date or 'n/a'}"
            )
        print(f"   {text}")


if __name__ == "__main__":
    main()
