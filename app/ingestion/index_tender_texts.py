from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from app.config import Settings
from app.db.sqlite_store import (
    ChunkMetadataRecord,
    ChunkRecord,
    DocumentRecord,
    SQLiteDocumentStore,
    TenderTextRecord,
)
from app.db.vector_store import VectorStore, build_vector_store
from app.embeddings.client import build_embedding_client
from app.ingestion.chunker import TokenChunker
from app.ingestion.index_pdfs import (
    TenderMetadata,
    _extract_tender_id_from_path,
    _normalize_path_value,
    batch_items,
    load_tender_metadata_index,
)


@dataclass(frozen=True)
class IndexResult:
    indexed: bool
    num_chunks: int
    status: str


@dataclass(frozen=True)
class ChunkForIndexing:
    retrieval_text: str
    raw_table_json: str | None = None


def iter_text_paths(input_dir: Path) -> Iterator[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    for path in sorted(input_dir.rglob("*.txt")):
        if path.is_file():
            yield path


def compute_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _is_table_object(obj: object) -> bool:
    if not isinstance(obj, dict):
        return False
    return (
        "columns" in obj
        and "rows" in obj
        and isinstance(obj.get("columns"), list)
        and isinstance(obj.get("rows"), list)
    )


def _extract_table_ranges(text: str) -> list[tuple[int, int]]:
    decoder = json.JSONDecoder()
    ranges: list[tuple[int, int]] = []
    cursor = 0
    text_len = len(text)

    while cursor < text_len:
        start = text.find("{", cursor)
        if start == -1:
            break
        try:
            obj, relative_end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            cursor = start + 1
            continue

        end = start + relative_end
        if _is_table_object(obj):
            ranges.append((start, end))
            cursor = end
        else:
            cursor = start + 1

    return ranges


_PAGE_MARKER_RE = re.compile(
    r"^(?:contd\.?\s*/\s*)?page\s*[-:]?\s*\d+\s*$",
    re.IGNORECASE,
)
_DOT_LEADER_RE = re.compile(r"^[\.\-`'\"~_=\s]+$")
_TABLE_CONTEXT_MAX_CHARS = 1200
_SMALL_TABLE_THRESHOLD_CHARS = 700
_TABLE_TARGET_RETRIEVAL_CHARS = 1200


def _is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if _PAGE_MARKER_RE.match(stripped):
        return True
    if _DOT_LEADER_RE.match(stripped) and len(re.sub(r"\s+", "", stripped)) >= 12:
        return True
    return False


def _clean_inter_table_text(text: str) -> str:
    kept_lines = [line for line in text.splitlines() if not _is_noise_line(line)]
    return "\n".join(kept_lines).strip()


def _trim_table_context(
    text: str,
    *,
    keep_from_end: bool,
    max_chars: int = _TABLE_CONTEXT_MAX_CHARS,
) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return normalized
    if keep_from_end:
        return normalized[-max_chars:]
    return normalized[:max_chars]


def _table_value_to_text(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _parse_table_json_preserving_number_text(raw_table_json: str) -> dict | None:
    try:
        parsed = json.loads(
            raw_table_json,
            parse_int=lambda token: token,
            parse_float=lambda token: token,
        )
    except json.JSONDecodeError:
        return None
    if not _is_table_object(parsed):
        return None
    return parsed


def _table_json_to_single_passage(raw_table_json: str) -> str:
    parsed_table = _parse_table_json_preserving_number_text(raw_table_json)
    if not parsed_table:
        return raw_table_json.strip()

    raw_columns = parsed_table.get("columns") or []
    columns = [str(column) for column in raw_columns]
    columns_section = "; ".join(columns)
    prefix = f"Columns: {columns_section}. Rows:"

    row_sections: list[str] = []
    raw_rows = parsed_table.get("rows") or []
    for row_index, raw_row in enumerate(raw_rows, start=1):
        if isinstance(raw_row, dict):
            row_values = [
                f"{column}: {_table_value_to_text(raw_row.get(column, ''))}"
                for column in columns
            ]
            row_sections.append(f"{row_index} — " + "; ".join(row_values) + ".")
        else:
            row_sections.append(
                f"{row_index} — value: {_table_value_to_text(raw_row)}."
            )
    if not row_sections:
        return prefix
    return f"{prefix} {' '.join(row_sections)}"


def _table_retrieval_text_with_context(
    raw_table_json: str,
    *,
    context_above: str,
    context_below: str,
) -> str:
    table_content = _table_json_to_single_passage(raw_table_json)
    parts: list[str] = [f"Table content: {table_content}"]

    # Add surrounding context only for very small tables to avoid losing intent.
    if len(table_content) >= _SMALL_TABLE_THRESHOLD_CHARS:
        return "\n\n".join(parts)

    needed_chars = max(0, _TABLE_TARGET_RETRIEVAL_CHARS - len(table_content))
    if needed_chars == 0:
        return "\n\n".join(parts)

    per_side_target = min(_TABLE_CONTEXT_MAX_CHARS, max(120, needed_chars // 2))
    above = (
        _trim_table_context(
            context_above,
            keep_from_end=True,
            max_chars=per_side_target,
        )
        if context_above
        else ""
    )
    below = (
        _trim_table_context(
            context_below,
            keep_from_end=False,
            max_chars=per_side_target,
        )
        if context_below
        else ""
    )
    if above:
        parts.insert(0, f"Context above table: {above}")
    if below:
        parts.append(f"Context below table: {below}")
    return "\n\n".join(parts)


def chunk_text_preserving_tables(
    text: str,
    *,
    chunker: TokenChunker,
    table_parse_max_chars: int,
    tender_id: str,
    source_name: str,
) -> list[ChunkForIndexing]:
    if len(text) > table_parse_max_chars:
        logging.warning(
            "Skipping table-preserving parse | tender_id=%s | file=%s | chars=%s > %s. "
            "Using normal chunking path.",
            tender_id,
            source_name,
            len(text),
            table_parse_max_chars,
        )
        return [ChunkForIndexing(retrieval_text=chunk) for chunk in chunker.chunk(text)]

    table_ranges = _extract_table_ranges(text)
    if not table_ranges:
        return [ChunkForIndexing(retrieval_text=chunk) for chunk in chunker.chunk(text)]

    chunks: list[ChunkForIndexing] = []
    cursor = 0
    for idx, (start, end) in enumerate(table_ranges):
        context_above = ""
        if start > cursor:
            context_above = _clean_inter_table_text(text[cursor:start])
            if context_above:
                chunks.extend(
                    ChunkForIndexing(retrieval_text=chunk)
                    for chunk in chunker.chunk(context_above)
                )

        next_start = (
            table_ranges[idx + 1][0] if idx + 1 < len(table_ranges) else len(text)
        )
        context_below = _clean_inter_table_text(text[end:next_start])

        table_text = text[start:end].strip()
        if table_text:
            chunks.append(
                ChunkForIndexing(
                    retrieval_text=_table_retrieval_text_with_context(
                        table_text,
                        context_above=context_above,
                        context_below=context_below,
                    ),
                    raw_table_json=table_text,
                )
            )
        cursor = end

    if cursor < len(text):
        tail_text = _clean_inter_table_text(text[cursor:])
        if tail_text:
            chunks.extend(
                ChunkForIndexing(retrieval_text=chunk)
                for chunk in chunker.chunk(tail_text)
            )
    return chunks


def chunk_text_fast_by_chars(
    text: str,
    *,
    chunk_size_chars: int,
    overlap_chars: int,
) -> list[str]:
    if not text:
        return []
    if overlap_chars >= chunk_size_chars:
        raise ValueError("Fast chunk overlap must be smaller than chunk size.")

    step = chunk_size_chars - overlap_chars
    chunks: list[str] = []
    for start in range(0, len(text), step):
        end = min(len(text), start + chunk_size_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
    return chunks


def _find_metadata_for_text_path(
    *,
    path: Path,
    input_root: Path,
    metadata_by_path: dict[str, TenderMetadata],
    metadata_by_tender_id: dict[str, TenderMetadata],
) -> tuple[TenderMetadata | None, str]:
    metadata: TenderMetadata | None = None
    try:
        rel = path.relative_to(input_root).as_posix()
        as_pdf_rel = str(Path(rel).with_suffix(".pdf"))
        metadata = metadata_by_path.get(_normalize_path_value(as_pdf_rel))
    except ValueError:
        pass

    tender_id = _extract_tender_id_from_path(path) or "unknown"
    if metadata is None and tender_id != "unknown":
        metadata = metadata_by_tender_id.get(tender_id)
    if metadata and metadata.tender_id:
        tender_id = metadata.tender_id

    return metadata, tender_id


def index_text_file(
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
    table_parse_max_chars: int,
    fast_chunk_threshold_chars: int,
    fast_chunk_size_chars: int,
    fast_chunk_overlap_chars: int,
) -> IndexResult:
    file_hash = compute_file_hash(path)
    existing = store.get_by_hash(file_hash)
    if existing and not reindex:
        return IndexResult(indexed=False, num_chunks=0, status="already_indexed")

    read_start = time.perf_counter()
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return IndexResult(indexed=False, num_chunks=0, status="empty_text")
    logging.info(
        "Loaded %s | chars=%s | read_time=%.2fs",
        path.name,
        len(text),
        time.perf_counter() - read_start,
    )

    metadata, tender_id = _find_metadata_for_text_path(
        path=path,
        input_root=input_root,
        metadata_by_path=metadata_by_path,
        metadata_by_tender_id=metadata_by_tender_id,
    )

    chunk_start = time.perf_counter()
    if len(text) > fast_chunk_threshold_chars:
        logging.warning(
            "Using fast char-based chunking | tender_id=%s | file=%s | chars=%s > %s.",
            tender_id,
            path.name,
            len(text),
            fast_chunk_threshold_chars,
        )
        chunk_payloads = [
            ChunkForIndexing(retrieval_text=chunk)
            for chunk in chunk_text_fast_by_chars(
                text,
                chunk_size_chars=fast_chunk_size_chars,
                overlap_chars=fast_chunk_overlap_chars,
            )
        ]
    else:
        chunk_payloads = chunk_text_preserving_tables(
            text,
            chunker=chunker,
            table_parse_max_chars=table_parse_max_chars,
            tender_id=tender_id,
            source_name=path.name,
        )
    if not chunk_payloads:
        return IndexResult(indexed=False, num_chunks=0, status="no_chunks")
    logging.info(
        "Chunked %s | chunks=%s | chunk_time=%.2fs",
        path.name,
        len(chunk_payloads),
        time.perf_counter() - chunk_start,
    )
    chunk_texts_for_embedding = [item.retrieval_text for item in chunk_payloads]

    embeddings: list[list[float]] = []
    total_batches = max(
        1, (len(chunk_texts_for_embedding) + embed_batch_size - 1) // embed_batch_size
    )
    for batch_number, batch in enumerate(
        batch_items(chunk_texts_for_embedding, embed_batch_size), start=1
    ):
        batch_start = time.perf_counter()
        embeddings.extend(embed_client.embed(batch, input_type="document"))
        logging.info(
            "Embedded %s | batch=%s/%s | batch_chunks=%s | embed_time=%.2fs",
            path.name,
            batch_number,
            total_batches,
            len(batch),
            time.perf_counter() - batch_start,
        )
    if len(chunk_texts_for_embedding) != len(embeddings):
        raise RuntimeError("Embedding count mismatch.")

    metadata_payload = metadata.to_payload() if metadata else {}
    if tender_id != "unknown":
        metadata_payload.setdefault("tender_id", tender_id)

    doc_id = file_hash
    vector_store.delete_by_doc_id(doc_id)
    records = []
    lexical_records: list[ChunkRecord] = []
    metadata_records: list[ChunkMetadataRecord] = []
    for idx, (chunk_payload, embedding) in enumerate(zip(chunk_payloads, embeddings)):
        chunk_uid = f"{doc_id}::chunk_{idx}"
        records.append(
            {
                "id": chunk_uid,
                "doc_id": doc_id,
                "chunk_id": idx,
                "text": chunk_payload.retrieval_text,
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
                text=chunk_payload.retrieval_text,
                source_path=str(path.resolve()),
                source_name=path.name,
            )
        )
        if chunk_payload.raw_table_json:
            metadata_records.append(
                ChunkMetadataRecord(
                    chunk_uid=chunk_uid,
                    doc_id=doc_id,
                    raw_table_json=chunk_payload.raw_table_json,
                )
            )

    vector_store.upsert_chunks(
        records, recreate_on_dim_mismatch=recreate_table_on_dim_mismatch
    )
    store.replace_chunks(doc_id, lexical_records)
    store.replace_chunk_metadata(doc_id, metadata_records)

    stat = path.stat()
    store.upsert_document(
        DocumentRecord(
            id=doc_id,
            file_path=str(path.resolve()),
            file_name=path.name,
            file_hash=file_hash,
            file_size=stat.st_size,
            modified_time=stat.st_mtime,
            indexed_at=store.now_iso(),
            num_chunks=len(chunk_payloads),
        )
    )
    if tender_id != "unknown":
        store.upsert_tender_text(
            TenderTextRecord(
                tender_id=tender_id,
                doc_id=doc_id,
                source_path=str(path.resolve()),
                source_name=path.name,
                full_text=text,
                metadata=metadata_payload,
                indexed_at=store.now_iso(),
            )
        )
    return IndexResult(indexed=True, num_chunks=len(chunk_payloads), status="indexed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index pre-extracted tender text files into vector DB and SQLite."
    )
    parser.add_argument(
        "--input",
        default="mini_tenders_data_text",
        help="Folder containing tender text files.",
    )
    parser.add_argument(
        "--table",
        default="mini_data_text",
        help="Vector collection/table name.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Reindex even if file hash already exists.",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=64,
        help="Number of chunks per embedding request.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size in tokens for normal text.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Token overlap for normal text chunks.",
    )
    parser.add_argument(
        "--metadata-csv",
        default="tenders_data.csv",
        help="CSV file containing tender metadata.",
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
        help="Do not recreate vector table on embedding dimension mismatch.",
    )
    parser.add_argument(
        "--table-parse-max-chars",
        type=int,
        default=600_000,
        help=(
            "Skip table-preserving JSON parsing for files larger than this many chars "
            "to avoid very slow scans on huge OCR text files."
        ),
    )
    parser.add_argument(
        "--fast-chunk-threshold-chars",
        type=int,
        default=2_000_000,
        help=(
            "For files larger than this size, use fast char-window chunking "
            "instead of token-based chunking."
        ),
    )
    parser.add_argument(
        "--fast-chunk-size-chars",
        type=int,
        default=2400,
        help="Character window size for fast chunking path.",
    )
    parser.add_argument(
        "--fast-chunk-overlap-chars",
        type=int,
        default=300,
        help="Character overlap for fast chunking path.",
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
        deepinfra_base_url=settings.deepinfra_base_url,
    )
    chunker = TokenChunker(
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
    )

    all_txt = list(iter_text_paths(input_dir))
    metadata_by_path, metadata_by_tender_id = load_tender_metadata_index(metadata_csv_path)
    logging.info(
        "Starting text indexing run | input=%s | total_files=%s | metadata_path_keys=%s | metadata_tender_ids=%s",
        str(input_dir.resolve()),
        len(all_txt),
        len(metadata_by_path),
        len(metadata_by_tender_id),
    )

    run_start = time.perf_counter()
    indexed_count = 0
    skipped_count = 0
    failed_count = 0
    total_chunks_indexed = 0

    for file_number, txt_path in enumerate(all_txt, start=1):
        file_start = time.perf_counter()
        try:
            logging.info(
                "[%s/%s] Processing %s",
                file_number,
                len(all_txt),
                txt_path.name,
            )
            result = index_text_file(
                txt_path,
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
                table_parse_max_chars=args.table_parse_max_chars,
                fast_chunk_threshold_chars=args.fast_chunk_threshold_chars,
                fast_chunk_size_chars=args.fast_chunk_size_chars,
                fast_chunk_overlap_chars=args.fast_chunk_overlap_chars,
            )
            elapsed = time.perf_counter() - file_start
            if result.indexed:
                indexed_count += 1
                total_chunks_indexed += result.num_chunks
                logging.info(
                    "[%s/%s] Indexed %s | chunks=%s | time=%.2fs",
                    file_number,
                    len(all_txt),
                    txt_path.name,
                    result.num_chunks,
                    elapsed,
                )
            else:
                skipped_count += 1
                logging.info(
                    "[%s/%s] Skipped %s | reason=%s | time=%.2fs",
                    file_number,
                    len(all_txt),
                    txt_path.name,
                    result.status,
                    elapsed,
                )
        except Exception as exc:
            failed_count += 1
            logging.exception(
                "[%s/%s] Failed %s | error=%s",
                file_number,
                len(all_txt),
                txt_path.name,
                str(exc),
            )

    elapsed = time.perf_counter() - run_start
    logging.info(
        "Text indexing complete | total_files=%s | indexed=%s | skipped=%s | failed=%s | total_chunks_indexed=%s | elapsed_seconds=%.2f",
        len(all_txt),
        indexed_count,
        skipped_count,
        failed_count,
        total_chunks_indexed,
        elapsed,
    )


if __name__ == "__main__":
    main()
