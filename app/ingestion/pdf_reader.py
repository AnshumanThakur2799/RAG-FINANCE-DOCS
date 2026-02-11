from __future__ import annotations

from pathlib import Path

from markitdown import MarkItDown


def extract_text_from_pdf(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path}")

    converter = MarkItDown()
    result = converter.convert(str(path))
    markdown_text = (result.text_content or "").strip()
    return markdown_text
