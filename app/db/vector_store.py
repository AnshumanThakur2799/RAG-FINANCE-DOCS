from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Protocol

import lancedb
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

if TYPE_CHECKING:
    from app.config import Settings


VECTOR_DB_PROVIDERS = ("lancedb", "qdrant")


class VectorStore(Protocol):
    def delete_by_doc_id(self, doc_id: str) -> None:
        raise NotImplementedError

    def upsert_chunks(
        self,
        records: Iterable[dict],
        *,
        recreate_on_dim_mismatch: bool = False,
    ) -> None:
        raise NotImplementedError

    def search(self, vector: list[float], *, top_k: int = 5) -> list[dict]:
        raise NotImplementedError


class LanceDBVectorStore:
    def __init__(self, db_dir: Path, table_name: str = "document_chunks") -> None:
        self.db_dir = db_dir
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.table_name = table_name
        self._db = lancedb.connect(str(self.db_dir))

    def _table_exists(self) -> bool:
        return self.table_name in self._db.table_names()

    def delete_by_doc_id(self, doc_id: str) -> None:
        if not self._table_exists():
            return
        table = self._db.open_table(self.table_name)
        table.delete(f"doc_id = '{doc_id}'")

    def upsert_chunks(
        self,
        records: Iterable[dict],
        *,
        recreate_on_dim_mismatch: bool = False,
    ) -> None:
        records = list(records)
        if not records:
            return
        if not self._table_exists():
            self._db.create_table(self.table_name, data=records)
            return

        table = self._db.open_table(self.table_name)
        try:
            table.add(records)
        except RuntimeError as exc:
            message = str(exc)
            dim_mismatch = (
                "FixedSizeListType" in message
                or "expected size" in message
                or "ArrowInvalid" in message
            )
            if not (recreate_on_dim_mismatch and dim_mismatch):
                raise

            logging.warning(
                "Vector dimension mismatch detected for table '%s'. "
                "Recreating table with new embedding dimension.",
                self.table_name,
            )
            self._db.create_table(self.table_name, data=records, mode="overwrite")

    def search(self, vector: list[float], *, top_k: int = 5) -> list[dict]:
        if not self._table_exists():
            return []
        table = self._db.open_table(self.table_name)
        return table.search(vector).limit(top_k).to_list()


class QdrantVectorStore:
    def __init__(
        self,
        *,
        url: str,
        collection_name: str = "document_chunks",
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        prefer_grpc: bool = False,
    ) -> None:
        self.collection_name = collection_name
        self._client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=timeout_seconds,
            prefer_grpc=prefer_grpc,
        )

    def _collection_exists(self) -> bool:
        try:
            return bool(self._client.collection_exists(self.collection_name))
        except Exception:
            # Older clients may not support collection_exists.
            try:
                self._client.get_collection(self.collection_name)
                return True
            except Exception:
                return False

    def _create_collection(self, dim: int) -> None:
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        )

    def _recreate_collection(self, dim: int) -> None:
        if self._collection_exists():
            self._client.delete_collection(collection_name=self.collection_name)
        self._create_collection(dim)

    @staticmethod
    def _to_point_id(raw_id: str) -> str:
        # Qdrant accepts uint64 or UUID values as point IDs; keep deterministic UUIDs
        # so repeated ingests of the same chunk overwrite instead of duplicating.
        return str(uuid.uuid5(uuid.NAMESPACE_URL, raw_id))

    def delete_by_doc_id(self, doc_id: str) -> None:
        if not self._collection_exists():
            return

        points: list[Any] = []
        next_offset: Any = None
        while True:
            batch, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="doc_id",
                            match=qmodels.MatchValue(value=doc_id),
                        )
                    ]
                ),
                limit=256,
                offset=next_offset,
                with_payload=False,
                with_vectors=False,
            )
            if not batch:
                break
            points.extend(batch)
            if next_offset is None:
                break

        if not points:
            return

        self._client.delete(
            collection_name=self.collection_name,
            points_selector=qmodels.PointIdsList(points=[point.id for point in points]),
        )

    def upsert_chunks(
        self,
        records: Iterable[dict],
        *,
        recreate_on_dim_mismatch: bool = False,
    ) -> None:
        records = list(records)
        if not records:
            return

        dim = len(records[0].get("vector") or [])
        if dim <= 0:
            raise ValueError("Vector records must include non-empty 'vector' values.")

        if not self._collection_exists():
            self._create_collection(dim)

        points = [
            qmodels.PointStruct(
                id=self._to_point_id(str(record["id"])),
                vector=record["vector"],
                payload={
                    "id": record.get("id"),
                    "doc_id": record.get("doc_id"),
                    "chunk_id": record.get("chunk_id"),
                    "text": record.get("text"),
                    "source_path": record.get("source_path"),
                    "source_name": record.get("source_name"),
                },
            )
            for record in records
        ]

        try:
            self._client.upsert(collection_name=self.collection_name, points=points)
        except Exception as exc:
            message = str(exc).lower()
            dim_mismatch = "dimension" in message or "vector" in message
            if not (recreate_on_dim_mismatch and dim_mismatch):
                raise
            logging.warning(
                "Vector dimension mismatch detected for qdrant collection '%s'. "
                "Recreating collection with new embedding dimension.",
                self.collection_name,
            )
            self._recreate_collection(dim)
            self._client.upsert(collection_name=self.collection_name, points=points)

    def search(self, vector: list[float], *, top_k: int = 5) -> list[dict]:
        if not self._collection_exists():
            return []

        limit = max(1, top_k)
        # qdrant-client API differs across versions:
        # - older: client.search(...)
        # - newer: client.query_points(...)
        if hasattr(self._client, "search"):
            results = self._client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        else:
            response = self._client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            results = list(response.points)

        items: list[dict] = []
        for hit in results:
            payload = dict(hit.payload or {})
            items.append(
                {
                    "id": str(payload.get("id") or hit.id),
                    "doc_id": payload.get("doc_id"),
                    "chunk_id": payload.get("chunk_id"),
                    "text": payload.get("text"),
                    "source_path": payload.get("source_path"),
                    "source_name": payload.get("source_name"),
                    "score": float(hit.score),
                }
            )
        return items


def build_vector_store(settings: "Settings", *, table_name: str = "document_chunks") -> VectorStore:
    provider = settings.vector_db_provider.strip().lower()
    if provider == "lancedb":
        return LanceDBVectorStore(settings.lancedb_dir, table_name=table_name)
    if provider == "qdrant":
        return QdrantVectorStore(
            url=settings.qdrant_url,
            collection_name=table_name,
            api_key=settings.qdrant_api_key,
            timeout_seconds=settings.qdrant_timeout_seconds,
            prefer_grpc=settings.qdrant_prefer_grpc,
        )
    raise ValueError(
        f"Unsupported VECTOR_DB_PROVIDER '{settings.vector_db_provider}'. "
        f"Expected one of: {', '.join(VECTOR_DB_PROVIDERS)}"
    )
