from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.db.sqlite_store import ChunkRecord, DocumentRecord, SQLiteDocumentStore
from app.db.vector_store import VectorStore, build_vector_store
from app.embeddings.client import build_embedding_client
from app.ingestion.chunker import TokenChunker
from app.ingestion.index_pdfs import (
    IndexResult,
    TenderMetadata,
    batch_items,
    compute_file_hash,
    iter_pdf_paths,
    load_tender_metadata_index,
)
from app.ingestion.pdf_reader_llm import build_pdf_text_extractor


@dataclass
class _PreparedIndexPayload:
    path: Path
    file_hash: str
    chunks: list[str]
    embeddings: list[list[float]]
    metadata_payload: dict[str, str | int]


def _extract_tender_id_from_path(path: Path) -> str | None:
    import re

    pattern = re.compile(r"^\d{4}_[A-Za-z0-9]+_\d+_\d+$")
    for part in path.parts:
        if pattern.match(part):
            return part
    return None


def _normalize_path_value(value: str) -> str:
    import re

    normalized = value.replace("\\", "/").strip()
    normalized = re.sub(r"/+", "/", normalized)
    return normalized.lower()


def _default_max_concurrency(settings: Settings) -> int:
    provider = settings.embedding_provider.strip().lower()
    if provider in {"local", "hf", "huggingface"}:
        return 1
    cpu_count = os.cpu_count() or 4
    return max(2, min(8, cpu_count))


async def _prepare_pdf_payload(
    path: Path,
    *,
    input_root: Path,
    chunker: TokenChunker,
    embed_client,
    embed_batch_size: int,
    store: SQLiteDocumentStore,
    metadata_by_path: dict[str, TenderMetadata],
    metadata_by_tender_id: dict[str, TenderMetadata],
    reindex: bool,
    text_extractor: Callable[[Path], str],
) -> tuple[IndexResult, _PreparedIndexPayload | None]:
    file_hash = await asyncio.to_thread(compute_file_hash, path)
    existing = await asyncio.to_thread(store.get_by_hash, file_hash)
    if existing and not reindex:
        return IndexResult(indexed=False, num_chunks=0, status="already_indexed"), None

    text = await asyncio.to_thread(text_extractor, path)
    if not text:
        return IndexResult(indexed=False, num_chunks=0, status="empty_pdf"), None

    chunks = await asyncio.to_thread(chunker.chunk, text)
    embeddings: list[list[float]] = []
    for batch in batch_items(chunks, embed_batch_size):
        batch_embeddings = await asyncio.to_thread(
            embed_client.embed, batch, input_type="document"
        )
        embeddings.extend(batch_embeddings)

    if len(chunks) != len(embeddings):
        raise RuntimeError("Embedding count mismatch.")

    metadata: TenderMetadata | None = None
    try:
        relative_path = path.relative_to(input_root)
        metadata = metadata_by_path.get(_normalize_path_value(relative_path.as_posix()))
    except ValueError:
        pass

    if metadata is None:
        tender_id = _extract_tender_id_from_path(path)
        if tender_id:
            metadata = metadata_by_tender_id.get(tender_id)

    payload = _PreparedIndexPayload(
        path=path,
        file_hash=file_hash,
        chunks=chunks,
        embeddings=embeddings,
        metadata_payload=metadata.to_payload() if metadata else {},
    )
    return IndexResult(indexed=True, num_chunks=len(chunks), status="prepared"), payload


def _persist_payload(
    payload: _PreparedIndexPayload,
    *,
    store: SQLiteDocumentStore,
    vector_store: VectorStore,
    recreate_table_on_dim_mismatch: bool,
) -> IndexResult:
    doc_id = payload.file_hash
    path = payload.path

    vector_store.delete_by_doc_id(doc_id)
    records: list[dict[str, str | int | float | list[float]]] = []
    lexical_records: list[ChunkRecord] = []
    for idx, (chunk, embedding) in enumerate(zip(payload.chunks, payload.embeddings)):
        chunk_uid = f"{doc_id}::chunk_{idx}"
        records.append(
            {
                "id": chunk_uid,
                "doc_id": doc_id,
                "chunk_id": idx,
                "text": chunk,
                "vector": embedding,
                "source_path": str(path.resolve()),
                "source_name": path.name,
                **payload.metadata_payload,
            }
        )
        lexical_records.append(
            ChunkRecord(
                id=chunk_uid,
                doc_id=doc_id,
                chunk_id=idx,
                text=chunk,
                source_path=str(path.resolve()),
                source_name=path.name,
            )
        )

    vector_store.upsert_chunks(
        records, recreate_on_dim_mismatch=recreate_table_on_dim_mismatch
    )
    store.replace_chunks(doc_id, lexical_records)

    stat = path.stat()
    store.upsert_document(
        DocumentRecord(
            id=doc_id,
            file_path=str(path.resolve()),
            file_name=path.name,
            file_hash=payload.file_hash,
            file_size=stat.st_size,
            modified_time=stat.st_mtime,
            indexed_at=store.now_iso(),
            num_chunks=len(payload.chunks),
        )
    )
    return IndexResult(indexed=True, num_chunks=len(payload.chunks), status="indexed")


async def _index_one_pdf(
    file_number: int,
    total_files: int,
    pdf_path: Path,
    *,
    semaphore: asyncio.Semaphore,
    write_lock: asyncio.Lock,
    input_root: Path,
    chunker: TokenChunker,
    embed_client,
    store: SQLiteDocumentStore,
    vector_store: VectorStore,
    metadata_by_path: dict[str, TenderMetadata],
    metadata_by_tender_id: dict[str, TenderMetadata],
    reindex: bool,
    embed_batch_size: int,
    recreate_table_on_dim_mismatch: bool,
    text_extractor: Callable[[Path], str],
) -> tuple[IndexResult, float]:
    file_start = time.perf_counter()
    async with semaphore:
        prepared_result, payload = await _prepare_pdf_payload(
            pdf_path,
            input_root=input_root,
            chunker=chunker,
            embed_client=embed_client,
            embed_batch_size=embed_batch_size,
            store=store,
            metadata_by_path=metadata_by_path,
            metadata_by_tender_id=metadata_by_tender_id,
            reindex=reindex,
            text_extractor=text_extractor,
        )

    if not prepared_result.indexed or payload is None:
        elapsed = time.perf_counter() - file_start
        logging.info(
            "[%s/%s] Skipped %s | reason=%s | time=%.2fs",
            file_number,
            total_files,
            pdf_path.name,
            prepared_result.status,
            elapsed,
        )
        return prepared_result, elapsed

    async with write_lock:
        result = await asyncio.to_thread(
            _persist_payload,
            payload,
            store=store,
            vector_store=vector_store,
            recreate_table_on_dim_mismatch=recreate_table_on_dim_mismatch,
        )

    elapsed = time.perf_counter() - file_start
    logging.info(
        "[%s/%s] Indexed %s | chunks=%s | time=%.2fs",
        file_number,
        total_files,
        pdf_path.name,
        result.num_chunks,
        elapsed,
    )
    return result, elapsed


async def _run_async_indexing(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    input_dir = Path(args.input)
    metadata_csv_path = Path(args.metadata_csv)
    store = SQLiteDocumentStore(settings.sqlite_db_path)
    vector_store = build_vector_store(settings, table_name=args.table)
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
    chunker = TokenChunker(
        chunk_size=settings.chunk_size_tokens,
        overlap=settings.chunk_overlap_tokens,
    )
    text_extractor = build_pdf_text_extractor(
        settings,
        reader_mode=args.pdf_reader_mode,
    )

    all_pdfs = list(iter_pdf_paths(input_dir))
    metadata_by_path, metadata_by_tender_id = load_tender_metadata_index(metadata_csv_path)

    max_concurrency = args.max_concurrency or _default_max_concurrency(settings)
    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    write_lock = asyncio.Lock()

    run_start = time.perf_counter()
    wall_start = store.now_iso()

    logging.info(
        "Starting async indexing run | input=%s | total_files=%s | reindex=%s | embed_batch_size=%s | max_concurrency=%s | metadata_csv=%s | metadata_path_keys=%s | metadata_tender_ids=%s",
        str(input_dir.resolve()),
        len(all_pdfs),
        args.reindex,
        args.embed_batch_size,
        max_concurrency,
        str(metadata_csv_path.resolve()),
        len(metadata_by_path),
        len(metadata_by_tender_id),
    )

    tasks = [
        asyncio.create_task(
            _index_one_pdf(
                file_number,
                len(all_pdfs),
                pdf_path,
                semaphore=semaphore,
                write_lock=write_lock,
                input_root=input_dir,
                chunker=chunker,
                embed_client=embed_client,
                store=store,
                vector_store=vector_store,
                metadata_by_path=metadata_by_path,
                metadata_by_tender_id=metadata_by_tender_id,
                reindex=args.reindex,
                embed_batch_size=args.embed_batch_size,
                recreate_table_on_dim_mismatch=(
                    args.reindex and not args.no_table_recreate_on_dim_mismatch
                ),
                text_extractor=text_extractor,
            )
        )
        for file_number, pdf_path in enumerate(all_pdfs, start=1)
    ]

    indexed_count = 0
    skipped_count = 0
    failed_count = 0
    total_chunks_indexed = 0
    total_file_processing_seconds = 0.0

    for task in asyncio.as_completed(tasks):
        try:
            result, elapsed = await task
            total_file_processing_seconds += elapsed
            if result.indexed:
                indexed_count += 1
                total_chunks_indexed += result.num_chunks
            else:
                skipped_count += 1
        except Exception as exc:
            failed_count += 1
            logging.exception("Async indexing task failed | error=%s", str(exc))

    run_elapsed = time.perf_counter() - run_start
    avg_file_seconds = (
        total_file_processing_seconds / len(all_pdfs) if all_pdfs else 0.0
    )
    avg_chunks_per_indexed = (
        total_chunks_indexed / indexed_count if indexed_count else 0.0
    )

    logging.info(
        "Async indexing complete | started_at=%s | finished_at=%s | total_files=%s | indexed=%s | skipped=%s | failed=%s | total_chunks_indexed=%s | avg_chunks_per_indexed=%.2f | processing_time_seconds=%.2f | avg_time_per_file_seconds=%.2f",
        wall_start,
        store.now_iso(),
        len(all_pdfs),
        indexed_count,
        skipped_count,
        failed_count,
        total_chunks_indexed,
        avg_chunks_per_indexed,
        run_elapsed,
        avg_file_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Asynchronously index PDFs into vector DB with controlled concurrency."
    )
    parser.add_argument("--input", required=True, help="Folder containing PDFs to index.")
    parser.add_argument(
        "--table", default="min_data_set", help="Vector collection/table name."
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Reindex even if the PDF hash already exists.",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=64,
        help="Number of chunks per embedding request.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=0,
        help=(
            "Max number of PDFs to prepare in parallel. "
            "Set 0 to auto-pick based on embedding provider."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--no-table-recreate-on-dim-mismatch",
        action="store_true",
        help=(
            "Do not recreate vector collection/table when embedding dimension mismatches "
            "(advanced/debug use)."
        ),
    )
    parser.add_argument(
        "--metadata-csv",
        default="tenders_data.csv",
        help="CSV file containing tender metadata to attach to each indexed chunk.",
    )
    parser.add_argument(
        "--pdf-reader-mode",
        default="baseline",
        choices=["baseline", "llm", "llm_vision"],
        help=(
            "PDF reader mode: 'baseline' uses deterministic extraction, "
            "'llm' runs text reconstruction with the configured LLM, "
            "'llm_vision' renders PDF pages as images and uses multimodal LLM input."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    asyncio.run(_run_async_indexing(args))


if __name__ == "__main__":
    main()
