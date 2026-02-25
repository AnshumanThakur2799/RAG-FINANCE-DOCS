from __future__ import annotations

import argparse
import math
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import Settings
from app.embeddings.client import build_embedding_client


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vector dimension mismatch: {len(vec_a)} (query) vs {len(vec_b)} (chunk)"
        )
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Cannot compute cosine similarity for zero-norm vectors.")
    return dot_product / (norm_a * norm_b)


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _fetch_chunk_from_lancedb(
    *,
    settings: Settings,
    table_name: str,
    tender_id: str,
    chunk_id: int,
) -> dict[str, Any] | None:
    import lancedb

    db = lancedb.connect(str(settings.lancedb_dir))
    if table_name not in db.table_names():
        raise ValueError(
            f"LanceDB table '{table_name}' was not found in '{settings.lancedb_dir}'."
        )

    table = db.open_table(table_name)
    where_clause = f"tender_id = {_sql_literal(tender_id)} AND chunk_id = {chunk_id}"
    rows = table.search().where(where_clause).limit(1).to_list()
    if not rows:
        return None
    return rows[0]


def _fetch_chunk_from_qdrant(
    *,
    settings: Settings,
    collection_name: str,
    tender_id: str,
    chunk_id: int,
) -> dict[str, Any] | None:
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout_seconds,
        prefer_grpc=settings.qdrant_prefer_grpc,
    )

    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="tender_id",
                    match=qmodels.MatchValue(value=tender_id),
                ),
                qmodels.FieldCondition(
                    key="chunk_id",
                    match=qmodels.MatchValue(value=chunk_id),
                ),
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=True,
    )

    if not points:
        return None
    point = points[0]
    payload = dict(point.payload or {})
    payload["vector"] = list(point.vector) if point.vector is not None else None
    # print(f"payload from the qdrant: {payload}")
    return payload


def _fetch_chunk_record(
    *,
    settings: Settings,
    table_name: str,
    tender_id: str,
    chunk_id: int,
) -> dict[str, Any] | None:
    provider = settings.vector_db_provider.strip().lower()
    if provider == "lancedb":
        return _fetch_chunk_from_lancedb(
            settings=settings,
            table_name=table_name,
            tender_id=tender_id,
            chunk_id=chunk_id,
        )
    if provider == "qdrant":
        return _fetch_chunk_from_qdrant(
            settings=settings,
            collection_name=table_name,
            tender_id=tender_id,
            chunk_id=chunk_id,
        )
    raise ValueError(
        f"Unsupported VECTOR_DB_PROVIDER '{settings.vector_db_provider}'. "
        "Expected 'lancedb' or 'qdrant'."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute cosine similarity between a stored tender chunk vector and a query."
        )
    )
    parser.add_argument("--tender-id", required=True, help="Tender ID to filter by.")
    parser.add_argument("--chunk-id", required=True, type=int, help="Chunk ID to match.")
    parser.add_argument("--query", required=True, help="Query text to embed and compare.")
    parser.add_argument(
        "--table",
        default="tenders_data",
        help="Vector table/collection name (default: tenders_data).",
    )
    parser.add_argument("--passage", required=True, help="Passage text to embed and compare.")
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

    chunk_record = _fetch_chunk_record(
        settings=settings,
        table_name=args.table,
        tender_id=args.tender_id,
        chunk_id=args.chunk_id,
    )
    if not chunk_record:
        raise SystemExit(
            "No matching chunk found for "
            f"tender_id='{args.tender_id}', chunk_id={args.chunk_id}, table='{args.table}'."
        )

    chunk_vector = chunk_record.get("vector")
    if not isinstance(chunk_vector, list) or not chunk_vector:
        raise SystemExit(
            "Matching chunk was found, but no vector is available in the stored record."
        )

    query_vector = embed_client.embed([args.query], input_type="query")[0]
    # print(f"query_vector: {query_vector}")
    passage_vector = embed_client.embed([args.passage], input_type="document")[0]
    # print(f"passage_vector: {passage_vector}")
    similarity = cosine_similarity(query_vector, chunk_vector)

    passage_similarity = cosine_similarity(passage_vector, chunk_vector)

    text_preview = str(chunk_record.get("text", "")).replace("\n", " ").strip()
    if len(text_preview) > 220:
        text_preview = text_preview[:220] + "..."

    print(f"provider={settings.vector_db_provider}")
    print(f"table={args.table}")
    print(f"tender_id={args.tender_id}")
    print(f"chunk_id={args.chunk_id}")
    print(f"query='{args.query}'")
    print(f"cosine_similarity={similarity:.6f}")
    print(f"chunk_text_preview={text_preview}")
    print(f"passage_similarity={passage_similarity:.6f}")


if __name__ == "__main__":
    main()
