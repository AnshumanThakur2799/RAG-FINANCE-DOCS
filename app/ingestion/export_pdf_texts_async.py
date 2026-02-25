from __future__ import annotations

import argparse
import asyncio
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.config import Settings
from app.ingestion.pdf_reader_llm import build_pdf_text_extractor


def iter_pdf_paths(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    return sorted(path for path in input_dir.rglob("*.pdf") if path.is_file())


@dataclass(frozen=True)
class ExportJob:
    index: int
    total: int
    pdf_path: Path
    target_text_path: Path


def _is_retryable_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    if isinstance(status_code, int) and 500 <= status_code <= 599:
        return True

    text = str(exc).lower()
    retryable_markers = (
        "429",
        "rate limit",
        "rate-limited",
        "too many requests",
        "timeout",
        "timed out",
        "connection reset",
        "temporarily unavailable",
        "server overloaded",
    )
    return any(marker in text for marker in retryable_markers)


async def _extract_with_retries(
    *,
    text_extractor: Callable[[Path], str],
    pdf_path: Path,
    max_retries: int,
    retry_base_delay_seconds: float,
    retry_max_delay_seconds: float,
) -> str:
    for attempt in range(1, max_retries + 2):
        try:
            return await asyncio.to_thread(text_extractor, pdf_path)
        except Exception as exc:
            is_retryable = _is_retryable_error(exc)
            if attempt > max_retries or not is_retryable:
                raise

            delay = min(
                retry_max_delay_seconds,
                retry_base_delay_seconds * (2 ** (attempt - 1)),
            )
            # Small random jitter avoids synchronized retries under burst load.
            jitter = random.uniform(0.0, max(0.1, delay * 0.2))
            sleep_for = delay + jitter
            logging.warning(
                "Retrying extraction after transient failure | pdf=%s | attempt=%s/%s | sleep=%.2fs | error=%s",
                str(pdf_path),
                attempt,
                max_retries + 1,
                sleep_for,
                str(exc),
            )
            await asyncio.sleep(sleep_for)

    raise RuntimeError(f"Retries exhausted for {pdf_path}")


async def _process_job(
    *,
    job: ExportJob,
    text_extractor: Callable[[Path], str],
    semaphore: asyncio.Semaphore,
    max_retries: int,
    retry_base_delay_seconds: float,
    retry_max_delay_seconds: float,
) -> tuple[bool, bool]:
    async with semaphore:
        try:
            extracted_text = await _extract_with_retries(
                text_extractor=text_extractor,
                pdf_path=job.pdf_path,
                max_retries=max_retries,
                retry_base_delay_seconds=retry_base_delay_seconds,
                retry_max_delay_seconds=retry_max_delay_seconds,
            )
            job.target_text_path.parent.mkdir(parents=True, exist_ok=True)
            job.target_text_path.write_text(extracted_text, encoding="utf-8")
            logging.info(
                "[%s/%s] Exported text | pdf=%s | out=%s | chars=%s",
                job.index,
                job.total,
                str(job.pdf_path),
                str(job.target_text_path),
                len(extracted_text),
            )
            return True, False
        except Exception as exc:
            logging.exception(
                "[%s/%s] Failed to export text | pdf=%s | error=%s",
                job.index,
                job.total,
                str(job.pdf_path),
                str(exc),
            )
            return False, True


async def export_pdf_texts_async(
    *,
    input_dir: Path,
    output_dir: Path,
    text_extractor: Callable[[Path], str],
    overwrite: bool,
    max_concurrent_requests: int,
    max_retries: int,
    retry_base_delay_seconds: float,
    retry_max_delay_seconds: float,
) -> tuple[int, int, int]:
    pdf_paths = iter_pdf_paths(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[ExportJob] = []
    skipped_existing = 0
    total_pdfs = len(pdf_paths)

    for index, pdf_path in enumerate(pdf_paths, start=1):
        relative_pdf_path = pdf_path.relative_to(input_dir)
        target_text_path = (output_dir / relative_pdf_path).with_suffix(".txt")

        if target_text_path.exists() and not overwrite:
            skipped_existing += 1
            logging.info(
                "[%s/%s] Skipped existing text file: %s",
                index,
                total_pdfs,
                str(target_text_path),
            )
            continue

        jobs.append(
            ExportJob(
                index=index,
                total=total_pdfs,
                pdf_path=pdf_path,
                target_text_path=target_text_path,
            )
        )

    if not jobs:
        return 0, 0, skipped_existing

    semaphore = asyncio.Semaphore(max_concurrent_requests)
    tasks = [
        _process_job(
            job=job,
            text_extractor=text_extractor,
            semaphore=semaphore,
            max_retries=max_retries,
            retry_base_delay_seconds=retry_base_delay_seconds,
            retry_max_delay_seconds=retry_max_delay_seconds,
        )
        for job in jobs
    ]
    results = await asyncio.gather(*tasks)

    written = sum(1 for success, _ in results if success)
    failed = sum(1 for _, failure in results if failure)
    return written, failed, skipped_existing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Asynchronously extract text from PDFs and save .txt files while preserving folder structure."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Root folder containing PDF files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination root folder for extracted .txt files.",
    )
    parser.add_argument(
        "--pdf-reader-mode",
        default="llm_vision",
        choices=["baseline", "llm", "llm_vision"],
        help=(
            "PDF reader mode: 'baseline' uses deterministic extraction, "
            "'llm' runs text reconstruction with the configured LLM, "
            "'llm_vision' renders PDF pages as images and uses multimodal LLM input."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .txt files in the output folder.",
    )
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=120,
        help=(
            "Maximum number of in-flight extraction jobs. Keep under DeepInfra model limit "
            "(200 concurrent by default). Recommended: 100-150."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Retries per PDF for transient/rate-limit errors.",
    )
    parser.add_argument(
        "--retry-base-delay-seconds",
        type=float,
        default=1.0,
        help="Initial retry delay in seconds (exponential backoff base).",
    )
    parser.add_argument(
        "--retry-max-delay-seconds",
        type=float,
        default=60.0,
        help="Maximum retry delay in seconds.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    args = parser.parse_args()

    if args.max_concurrent_requests < 1:
        parser.error("--max-concurrent-requests must be >= 1")
    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0")
    if args.retry_base_delay_seconds <= 0:
        parser.error("--retry-base-delay-seconds must be > 0")
    if args.retry_max_delay_seconds <= 0:
        parser.error("--retry-max-delay-seconds must be > 0")
    if args.retry_max_delay_seconds < args.retry_base_delay_seconds:
        parser.error("--retry-max-delay-seconds must be >= --retry-base-delay-seconds")

    return args


async def main_async() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    settings = Settings.from_env()
    text_extractor = build_pdf_text_extractor(
        settings,
        reader_mode=args.pdf_reader_mode,
    )

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    pdf_count = len(iter_pdf_paths(input_dir))
    logging.info(
        "Starting async text export | input=%s | output=%s | pdf_count=%s | mode=%s | overwrite=%s | concurrency=%s",
        str(input_dir.resolve()),
        str(output_dir.resolve()),
        pdf_count,
        args.pdf_reader_mode,
        args.overwrite,
        args.max_concurrent_requests,
    )

    written, failed, skipped_existing = await export_pdf_texts_async(
        input_dir=input_dir,
        output_dir=output_dir,
        text_extractor=text_extractor,
        overwrite=args.overwrite,
        max_concurrent_requests=args.max_concurrent_requests,
        max_retries=args.max_retries,
        retry_base_delay_seconds=args.retry_base_delay_seconds,
        retry_max_delay_seconds=args.retry_max_delay_seconds,
    )

    logging.info(
        "Async text export complete | total_pdfs=%s | written=%s | skipped_existing=%s | failed=%s",
        pdf_count,
        written,
        skipped_existing,
        failed,
    )


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
