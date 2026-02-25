from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.config import Settings
from app.ingestion.pdf_reader_llm import build_pdf_text_extractor


def iter_pdf_paths(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    return sorted(path for path in input_dir.rglob("*.pdf") if path.is_file())


def export_pdf_texts(
    *,
    input_dir: Path,
    output_dir: Path,
    text_extractor,
    overwrite: bool,
) -> tuple[int, int]:
    pdf_paths = iter_pdf_paths(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    failed = 0
    for index, pdf_path in enumerate(pdf_paths, start=1):
        try:
            relative_pdf_path = pdf_path.relative_to(input_dir)
            target_text_path = (output_dir / relative_pdf_path).with_suffix(".txt")

            if target_text_path.exists() and not overwrite:
                logging.info(
                    "[%s/%s] Skipped existing text file: %s",
                    index,
                    len(pdf_paths),
                    str(target_text_path),
                )
                continue

            extracted_text = text_extractor(pdf_path)
            target_text_path.parent.mkdir(parents=True, exist_ok=True)
            target_text_path.write_text(extracted_text, encoding="utf-8")

            logging.info(
                "[%s/%s] Exported text | pdf=%s | out=%s | chars=%s",
                index,
                len(pdf_paths),
                str(pdf_path),
                str(target_text_path),
                len(extracted_text),
            )
            written += 1
        except Exception as exc:
            failed += 1
            logging.exception(
                "[%s/%s] Failed to export text | pdf=%s | error=%s",
                index,
                len(pdf_paths),
                str(pdf_path),
                str(exc),
            )

    return written, failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract text from PDFs and save .txt files while preserving folder structure."
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
        default="baseline",
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
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    args = parser.parse_args()

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
        "Starting text export | input=%s | output=%s | pdf_count=%s | mode=%s | overwrite=%s",
        str(input_dir.resolve()),
        str(output_dir.resolve()),
        pdf_count,
        args.pdf_reader_mode,
        args.overwrite,
    )

    written, failed = export_pdf_texts(
        input_dir=input_dir,
        output_dir=output_dir,
        text_extractor=text_extractor,
        overwrite=args.overwrite,
    )

    logging.info(
        "Text export complete | total_pdfs=%s | written=%s | failed=%s",
        pdf_count,
        written,
        failed,
    )


if __name__ == "__main__":
    main()
