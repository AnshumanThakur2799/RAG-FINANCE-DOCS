from __future__ import annotations

from pathlib import Path

import fitz


def extract_text_from_pdf(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    text_chunks: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            page_text = page.get_text("text") or ""
            if page_text:
                text_chunks.append(page_text.strip())

    return "\n".join(text_chunks).strip()
