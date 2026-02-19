from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from app.config import Settings
from app.db.sqlite_store import ChunkRecord, DocumentRecord, SQLiteDocumentStore
from app.db.vector_store import VectorStore, build_vector_store
from app.embeddings.client import build_embedding_client
from app.ingestion.chunker import TokenChunker
from app.ingestion.pdf_reader import extract_text_from_pdf


@dataclass(frozen=True)
class IndexResult:
    indexed: bool
    num_chunks: int
    status: str


@dataclass(frozen=True)
class TenderMetadata:
    organization: str | None
    tender_id: str | None
    title: str | None
    publish_date: str | None
    publish_date_epoch: int | None
    bid_opening_date: str | None
    bid_opening_date_epoch: int | None
    bid_submission_start: str | None
    bid_submission_start_epoch: int | None
    bid_submission_end: str | None
    bid_submission_end_epoch: int | None
    tender_url: str | None
    local_file_path: str | None

    def to_payload(self) -> dict[str, str | int]:
        payload: dict[str, str | int] = {}
        for key, value in self.__dict__.items():
            if value is not None and value != "":
                payload[key] = value
        return payload


def _clean_value(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _normalize_path_value(value: str) -> str:
    normalized = value.replace("\\", "/").strip()
    normalized = re.sub(r"/+", "/", normalized)
    return normalized.lower()


def _split_local_paths(value: str | None) -> list[str]:
    cleaned = _clean_value(value)
    if not cleaned:
        return []
    return [part.strip() for part in cleaned.split("|") if part.strip()]


def _parse_tender_datetime(value: str | None) -> tuple[str | None, int | None]:
    cleaned = _clean_value(value)
    if not cleaned:
        return None, None
    parsed = datetime.strptime(cleaned, "%d-%b-%Y %I:%M %p")
    return parsed.isoformat(), int(parsed.replace(tzinfo=timezone.utc).timestamp())


def load_tender_metadata_index(
    csv_path: Path,
) -> tuple[dict[str, TenderMetadata], dict[str, TenderMetadata]]:
    by_file_path: dict[str, TenderMetadata] = {}
    by_tender_id: dict[str, TenderMetadata] = {}

    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                publish_iso, publish_epoch = _parse_tender_datetime(
                    row.get("Publish Date")
                )
                opening_iso, opening_epoch = _parse_tender_datetime(
                    row.get("Bid Opening Date")
                )
                start_iso, start_epoch = _parse_tender_datetime(
                    row.get("Bid Submission Start")
                )
                end_iso, end_epoch = _parse_tender_datetime(
                    row.get("Bid Submission End")
                )
            except ValueError as exc:
                logging.warning(
                    "Skipping metadata row due to invalid date format | tender_id=%s | error=%s",
                    row.get("Tender ID", ""),
                    str(exc),
                )
                continue

            metadata = TenderMetadata(
                organization=_clean_value(row.get("Organization")),
                tender_id=_clean_value(row.get("Tender ID")),
                title=_clean_value(row.get("Title")),
                publish_date=publish_iso,
                publish_date_epoch=publish_epoch,
                bid_opening_date=opening_iso,
                bid_opening_date_epoch=opening_epoch,
                bid_submission_start=start_iso,
                bid_submission_start_epoch=start_epoch,
                bid_submission_end=end_iso,
                bid_submission_end_epoch=end_epoch,
                tender_url=_clean_value(row.get("Tender URL")),
                local_file_path=_clean_value(row.get("Local File Path")),
            )

            if metadata.tender_id and metadata.tender_id not in by_tender_id:
                by_tender_id[metadata.tender_id] = metadata

            for local_path in _split_local_paths(row.get("Local File Path")):
                by_file_path[_normalize_path_value(local_path)] = metadata

    return by_file_path, by_tender_id


def _extract_tender_id_from_path(path: Path) -> str | None:
    pattern = re.compile(r"^\d{4}_[A-Za-z0-9]+_\d+_\d+$")
    for part in path.parts:
        if pattern.match(part):
            return part
    return None


def iter_pdf_paths(input_dir: Path) -> Iterator[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    for path in sorted(input_dir.rglob("*.pdf")):
        if path.is_file():
            yield path


def compute_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def batch_items(items: list[str], batch_size: int) -> Iterator[list[str]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def index_pdf(
    path: Path,
    *,
    input_root: Path,
    chunker: TokenChunker,
    embed_client,
    vector_store: VectorStore,
    store: SQLiteDocumentStore,
    metadata_by_path: dict[str, TenderMetadata],
    metadata_by_tender_id: dict[str, TenderMetadata],
    reindex: bool,
    embed_batch_size: int,
    recreate_table_on_dim_mismatch: bool,
) -> IndexResult:
    step_timings: dict[str, float] = {}

    step_start = time.perf_counter()
    file_hash = compute_file_hash(path)
    step_timings["compute_file_hash"] = time.perf_counter() - step_start

    step_start = time.perf_counter()
    existing = store.get_by_hash(file_hash)
    step_timings["store_get_by_hash"] = time.perf_counter() - step_start
    if existing and not reindex:
        logging.info(
            "File timing breakdown | file=%s | status=already_indexed | total=%.3fs | %s=%.3fs | %s=%.3fs",
            path.name,
            sum(step_timings.values()),
            "compute_file_hash",
            step_timings["compute_file_hash"],
            "store_get_by_hash",
            step_timings["store_get_by_hash"],
        )
        return IndexResult(indexed=False, num_chunks=0, status="already_indexed")

    step_start = time.perf_counter()
    text = extract_text_from_pdf(path)
    step_timings["extract_text_from_pdf"] = time.perf_counter() - step_start
    if not text:
        logging.info(
            "File timing breakdown | file=%s | status=empty_pdf | total=%.3fs | hash=%.3fs | hash_lookup=%.3fs | extract_text=%.3fs",
            path.name,
            sum(step_timings.values()),
            step_timings["compute_file_hash"],
            step_timings["store_get_by_hash"],
            step_timings["extract_text_from_pdf"],
        )
        return IndexResult(indexed=False, num_chunks=0, status="empty_pdf")

    step_start = time.perf_counter()
    chunks = chunker.chunk(text)
    step_timings["chunker_chunk"] = time.perf_counter() - step_start
    embeddings: list[list[float]] = []
    embedding_total = 0.0
    for batch in batch_items(chunks, embed_batch_size):
        embed_batch_start = time.perf_counter()
        embeddings.extend(embed_client.embed(batch, input_type="document"))
        batch_elapsed = time.perf_counter() - embed_batch_start
        embedding_total += batch_elapsed
        logging.info(
            "Embedding batch timing | file=%s | batch_size=%s | elapsed=%.3fs",
            path.name,
            len(batch),
            batch_elapsed,
        )
    step_timings["embed_batches_total"] = embedding_total

    if len(chunks) != len(embeddings):
        raise RuntimeError("Embedding count mismatch.")

    doc_id = file_hash
    if existing:
        step_start = time.perf_counter()
        vector_store.delete_by_doc_id(doc_id)
        step_timings["vector_delete_by_doc_id"] = time.perf_counter() - step_start

    metadata: TenderMetadata | None = None
    try:
        relative_path = path.relative_to(input_root)
        metadata = metadata_by_path.get(_normalize_path_value(relative_path.as_posix()))
    except ValueError:
        relative_path = None

    if metadata is None:
        tender_id = _extract_tender_id_from_path(path)
        if tender_id:
            metadata = metadata_by_tender_id.get(tender_id)

    metadata_payload = metadata.to_payload() if metadata else {}

    records = []
    lexical_records: list[ChunkRecord] = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
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
                **metadata_payload,
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

    step_start = time.perf_counter()
    vector_store.upsert_chunks(
        records, recreate_on_dim_mismatch=recreate_table_on_dim_mismatch
    )
    step_timings["vector_upsert_chunks"] = time.perf_counter() - step_start

    step_start = time.perf_counter()
    store.replace_chunks(doc_id, lexical_records)
    step_timings["store_replace_chunks"] = time.perf_counter() - step_start

    step_start = time.perf_counter()
    stat = path.stat()
    step_timings["path_stat"] = time.perf_counter() - step_start

    step_start = time.perf_counter()
    store.upsert_document(
        DocumentRecord(
            id=doc_id,
            file_path=str(path.resolve()),
            file_name=path.name,
            file_hash=file_hash,
            file_size=stat.st_size,
            modified_time=stat.st_mtime,
            indexed_at=store.now_iso(),
            num_chunks=len(chunks),
        )
    )
    step_timings["store_upsert_document"] = time.perf_counter() - step_start

    total_elapsed = sum(step_timings.values())
    logging.info(
        (
            "File timing breakdown | file=%s | status=indexed | chunks=%s | total=%.3fs | "
            "hash=%.3fs | hash_lookup=%.3fs | extract_text=%.3fs | chunking=%.3fs | "
            "embedding_total=%.3fs | vector_delete=%.3fs | vector_upsert=%.3fs | "
            "sqlite_replace_chunks=%.3fs | file_stat=%.3fs | sqlite_upsert_document=%.3fs"
        ),
        path.name,
        len(chunks),
        total_elapsed,
        step_timings.get("compute_file_hash", 0.0),
        step_timings.get("store_get_by_hash", 0.0),
        step_timings.get("extract_text_from_pdf", 0.0),
        step_timings.get("chunker_chunk", 0.0),
        step_timings.get("embed_batches_total", 0.0),
        step_timings.get("vector_delete_by_doc_id", 0.0),
        step_timings.get("vector_upsert_chunks", 0.0),
        step_timings.get("store_replace_chunks", 0.0),
        step_timings.get("path_stat", 0.0),
        step_timings.get("store_upsert_document", 0.0),
    )
    return IndexResult(indexed=True, num_chunks=len(chunks), status="indexed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index PDFs into configured vector DB and track metadata in SQLite."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Folder containing PDFs to index.",
    )
    parser.add_argument(
        "--table",
        default="tender_chunks",
        help="Vector collection/table name.",
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
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

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
    )
    chunker = TokenChunker(
        chunk_size=settings.chunk_size_tokens,
        overlap=settings.chunk_overlap_tokens,
    )

    all_pdfs = list(iter_pdf_paths(input_dir))
    metadata_by_path, metadata_by_tender_id = load_tender_metadata_index(metadata_csv_path)
    run_start = time.perf_counter()
    wall_start = store.now_iso()

    indexed_count = 0
    skipped_count = 0
    failed_count = 0
    total_chunks_indexed = 0
    total_file_processing_seconds = 0.0

    logging.info(
        "Starting indexing run | input=%s | total_files=%s | reindex=%s | embed_batch_size=%s | metadata_csv=%s | metadata_path_keys=%s | metadata_tender_ids=%s",
        str(input_dir.resolve()),
        len(all_pdfs),
        args.reindex,
        args.embed_batch_size,
        str(metadata_csv_path.resolve()),
        len(metadata_by_path),
        len(metadata_by_tender_id),
    )

    for file_number, pdf_path in enumerate(all_pdfs, start=1):
        file_start = time.perf_counter()
        try:
            result = index_pdf(
                pdf_path,
                input_root=input_dir,
                chunker=chunker,
                embed_client=embed_client,
                vector_store=vector_store,
                store=store,
                metadata_by_path=metadata_by_path,
                metadata_by_tender_id=metadata_by_tender_id,
                reindex=args.reindex,
                embed_batch_size=args.embed_batch_size,
                recreate_table_on_dim_mismatch=(
                    args.reindex and not args.no_table_recreate_on_dim_mismatch
                ),
            )
            file_elapsed = time.perf_counter() - file_start
            total_file_processing_seconds += file_elapsed

            if result.indexed:
                indexed_count += 1
                total_chunks_indexed += result.num_chunks
                logging.info(
                    "[%s/%s] Indexed %s | chunks=%s | time=%.2fs",
                    file_number,
                    len(all_pdfs),
                    pdf_path.name,
                    result.num_chunks,
                    file_elapsed,
                )
            else:
                skipped_count += 1
                logging.info(
                    "[%s/%s] Skipped %s | reason=%s | time=%.2fs",
                    file_number,
                    len(all_pdfs),
                    pdf_path.name,
                    result.status,
                    file_elapsed,
                )
        except Exception as exc:
            file_elapsed = time.perf_counter() - file_start
            total_file_processing_seconds += file_elapsed
            failed_count += 1
            logging.exception(
                "[%s/%s] Failed %s | time=%.2fs | error=%s",
                file_number,
                len(all_pdfs),
                pdf_path.name,
                file_elapsed,
                str(exc),
            )

    run_elapsed = time.perf_counter() - run_start
    avg_file_seconds = (
        total_file_processing_seconds / len(all_pdfs) if all_pdfs else 0.0
    )
    avg_chunks_per_indexed = (
        total_chunks_indexed / indexed_count if indexed_count else 0.0
    )

    logging.info(
        "Indexing complete | started_at=%s | finished_at=%s | total_files=%s | indexed=%s | skipped=%s | failed=%s | total_chunks_indexed=%s | avg_chunks_per_indexed=%.2f | processing_time_seconds=%.2f | avg_time_per_file_seconds=%.2f",
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


if __name__ == "__main__":
    main()
