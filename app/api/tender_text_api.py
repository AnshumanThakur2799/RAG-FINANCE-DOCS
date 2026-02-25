from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import Settings


app = FastAPI(title="RAG Finance Docs Utility API", version="0.1.0")


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _fetch_tender_chunks(
    *,
    client: QdrantClient,
    collection_name: str,
    tender_id: str,
    scroll_limit: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    next_offset: Any = None

    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="tender_id",
                        match=qmodels.MatchValue(value=tender_id),
                    )
                ]
            ),
            limit=scroll_limit,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        for point in points:
            payload = dict(point.payload or {})
            payload.setdefault("id", str(point.id))
            results.append(payload)

        if next_offset is None:
            break

    return results


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/utility/tenders/{tender_id}/text")
def get_tender_text(
    tender_id: str,
    collection_name: str = Query(default="tender_chunks"),
    page: int | None = Query(default=None, ge=1),
    page_size: int = Query(default=200, ge=1, le=2000),
    include_full_text: bool = Query(default=True),
    scroll_limit: int = Query(default=256, ge=1, le=1000),
) -> dict[str, Any]:
    settings = Settings.from_env()
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout_seconds,
        prefer_grpc=settings.qdrant_prefer_grpc,
    )

    try:
        chunks = _fetch_tender_chunks(
            client=client,
            collection_name=collection_name,
            tender_id=tender_id,
            scroll_limit=scroll_limit,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Qdrant query failed for tender_id '{tender_id}': {exc}",
        ) from exc

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail=f"No chunks found for tender_id '{tender_id}' in '{collection_name}'.",
        )

    chunks.sort(
        key=lambda item: (
            str(item.get("doc_id", "")),
            _to_int(item.get("chunk_id"), 0),
            str(item.get("id", "")),
        )
    )

    total_chunks = len(chunks)
    total_pages = (total_chunks + page_size - 1) // page_size

    if page is None:
        selected_chunks = chunks
        current_page = None
        has_next_page = False
    else:
        if page > total_pages:
            raise HTTPException(
                status_code=400,
                detail=f"Requested page {page} exceeds total pages {total_pages}.",
            )
        start = (page - 1) * page_size
        end = start + page_size
        selected_chunks = chunks[start:end]
        current_page = page
        has_next_page = page < total_pages

    page_text = "\n\n".join(
        str(chunk.get("text", "")).strip() for chunk in selected_chunks if chunk.get("text")
    )
    response: dict[str, Any] = {
        "tender_id": tender_id,
        "collection_name": collection_name,
        "total_chunks": total_chunks,
        "total_pages": total_pages if page is not None else 1,
        "page": current_page,
        "page_size": page_size if page is not None else total_chunks,
        "has_next_page": has_next_page,
        "concatenated_text": page_text,
    }

    if include_full_text:
        response["full_concatenated_text"] = "\n\n".join(
            str(chunk.get("text", "")).strip() for chunk in chunks if chunk.get("text")
        )

    return response
