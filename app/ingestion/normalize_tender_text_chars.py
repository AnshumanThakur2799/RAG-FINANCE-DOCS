from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path


LONG_CHAR_RUNS_PATTERN = re.compile(r"([*_.])\1{2,}")


def iter_text_paths(root_dir: Path) -> list[Path]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {root_dir}")
    return sorted(path for path in root_dir.rglob("*.txt") if path.is_file())


def normalize_long_char_runs(text: str) -> tuple[str, int]:
    replacement_count = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal replacement_count
        replacement_count += 1
        repeated_char = match.group(1)
        return repeated_char * 2

    normalized_text = LONG_CHAR_RUNS_PATTERN.sub(_replace, text)
    return normalized_text, replacement_count


def normalize_text_files(root_dir: Path, dry_run: bool = False) -> tuple[int, int, int]:
    file_paths = iter_text_paths(root_dir)
    files_changed = 0
    total_replacements = 0

    for index, file_path in enumerate(file_paths, start=1):
        original_text = file_path.read_text(encoding="utf-8")
        normalized_text, replacements = normalize_long_char_runs(original_text)

        if replacements == 0:
            continue

        files_changed += 1
        total_replacements += replacements

        if not dry_run:
            file_path.write_text(normalized_text, encoding="utf-8")

        logging.info(
            "[%s/%s] Normalized %s run(s) in %s",
            index,
            len(file_paths),
            replacements,
            str(file_path),
        )

    return len(file_paths), files_changed, total_replacements


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively normalize long runs of '*', '_' and '.' in .txt files "
            "to exactly 2 characters."
        )
    )
    parser.add_argument(
        "--input",
        default="mini_tenders_data_text",
        help="Root folder containing .txt files to normalize.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing files.",
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

    input_dir = Path(args.input)
    logging.info(
        "Starting normalization | input=%s | dry_run=%s",
        str(input_dir.resolve()),
        args.dry_run,
    )

    total_files, files_changed, replacements = normalize_text_files(
        input_dir,
        dry_run=args.dry_run,
    )

    logging.info(
        "Normalization complete | total_files=%s | files_changed=%s | replacements=%s",
        total_files,
        files_changed,
        replacements,
    )


if __name__ == "__main__":
    main()
