from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Callable, TypeVar

from app.config import Settings
from app.ingestion.pdf_reader import extract_text_from_pdf
from app.llm.client import LLMClient, build_llm_client

T = TypeVar("T")
MIN_NON_WHITESPACE_CHARS_FOR_VISION = 1000000000


LLM_RECONSTRUCTION_SYSTEM_PROMPT = """You are a document reconstruction assistant.

Your task is to convert the provided PDF content (which may include normal text, tables, OCR text from scanned pages, or mixed formatting) into clean, well-structured, human-readable plain text.

OUTPUT REQUIREMENTS:

1. The final output must be primarily clean plain text.
2. Preserve the logical structure:
   - Maintain headings and subheadings clearly.
   - Maintain paragraph breaks.
   - Preserve bullet lists and numbered lists.
   - Remove repeated headers/footers and page numbers.
   - Fix OCR errors where obvious (e.g., broken words, incorrect spacing).

3. TABLE HANDLING (Very Important):
   - When you detect a table, DO NOT flatten it into plain text.
   - Represent each table in one of the following formats:

   Option A (Preferred - if structure is clear):
   Output table as JSON:
   {
     "table_title": "If available",
     "columns": ["Column1", "Column2", ...],
     "rows": [
        {"Column1": "Value", "Column2": 123},
        ...
     ]
   }

   Option B (If structure is moderately clear):
   Use Markdown table format.

   Option C (If structure is unclear due to OCR):
   Provide:
   - A cleaned CSV-style representation
   - A short explanation note: "Table structure partially reconstructed due to OCR noise"

4. Do NOT add:
   - chunk IDs
   - metadata
   - summaries (unless explicitly present in the document)
   - explanations
   - extra commentary
   - decorative separators or filler lines made of repeated symbols

5. The output must look like a natural reconstructed document.
6. Maintain original wording as much as possible.
7. Preserve dates and all numeric values exactly as visible in the source.
   - Do NOT normalize, correct, infer, or reformat numbers/currency.
   - Do NOT remove/add commas, decimal points, minus signs, or digit separators.
   - If a number is unclear, keep the nearest faithful reading and add: [Unclear number].

IMPORTANT:
The result must be suitable for downstream RAG chunking.
That means:
- Keep logical sections intact.
- Do not merge unrelated sections.
- Do not hallucinate missing content.
- If text is unreadable, write: [Unreadable text due to OCR].
- Do NOT output long runs of repeated punctuation/symbols and line separators (for example:
  "_____", "-----", "=====", "......", repeated box-drawing characters).
- If a line is only a visual form rule/blank (signature line, underline, separator),
  OMIT that line instead of reproducing it.
- If a field is blank in a form, keep the field label but do not emit long
  placeholder underscores.

Return ONLY the reconstructed document."""


def _clean_reconstruction_artifacts(text: str) -> str:
    if not text:
        return ""

    cleaned = text
    # Replace pathological underscore runs from form lines/ruling artifacts.
    cleaned = re.sub(r"_{80,}", " ", cleaned)
    # Drop lines that are mostly underscores/hyphens from scanned forms.
    cleaned = re.sub(r"(?m)^[\s_\-]{60,}$", "", cleaned)
    # Keep paragraph spacing readable.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _split_for_llm(text: str, max_chars: int = 30000) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            paragraph_break = text.rfind("\n\n", start, end)
            line_break = text.rfind("\n", start, end)
            split_point = paragraph_break if paragraph_break > start else line_break
            if split_point > start:
                end = split_point
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    return chunks


def _reconstruct_chunk_with_llm(
    llm_client: LLMClient,
    chunk_text: str,
    chunk_number: int,
    total_chunks: int,
) -> str:
    user_prompt = (
        "Reconstruct the following extracted PDF text.\n\n"
        f"Chunk {chunk_number}/{total_chunks}:\n"
        f"{chunk_text}"
    )
    return llm_client.chat(LLM_RECONSTRUCTION_SYSTEM_PROMPT, user_prompt).strip()


def _has_sufficient_extracted_text(text: str, *, min_chars: int) -> bool:
    non_whitespace_chars = len("".join(text.split()))
    return non_whitespace_chars >= min_chars


def _batch_list(items: list[T], batch_size: int) -> list[list[T]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _render_pdf_pages_as_data_urls(
    path: Path,
    *,
    dpi: int = 300,
    image_format: str = "png",
) -> list[str]:
    import fitz

    urls: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY, alpha=False)
            encoded = base64.b64encode(pix.tobytes(image_format)).decode("ascii")
            urls.append(f"data:image/{image_format};base64,{encoded}")
    return urls


def extract_text_from_pdf_with_llm_vision(
    path: Path,
    *,
    llm_client: LLMClient,
    max_pages_per_call: int = 1,
    min_non_whitespace_chars_for_vision: int = MIN_NON_WHITESPACE_CHARS_FOR_VISION,
) -> str:
    baseline_text = extract_text_from_pdf(path)
    if _has_sufficient_extracted_text(
        baseline_text,
        min_chars=min_non_whitespace_chars_for_vision,
    ):
        logging.info(
            "LLM vision skipped | file=%s | reason=sufficient_baseline_text | min_chars=%s",
            path.name,
            min_non_whitespace_chars_for_vision,
        )
        return baseline_text

    page_image_urls = _render_pdf_pages_as_data_urls(path)
    if not page_image_urls:
        return baseline_text

    reconstructed_parts: list[str] = []
    page_batches = _batch_list(page_image_urls, max_pages_per_call)
    total_batches = len(page_batches)

    for idx, batch in enumerate(page_batches, start=1):
        logging.info(
            "LLM vision PDF reconstruction | file=%s | batch=%s/%s | pages_in_batch=%s",
            path.name,
            idx,
            total_batches,
            len(batch),
        )
        user_prompt = (
            "Reconstruct the document text from these PDF page images. "
            f"Batch {idx}/{total_batches}. Keep output faithful to visible content."
        )
        reconstructed = llm_client.chat_with_images(
            LLM_RECONSTRUCTION_SYSTEM_PROMPT,
            user_prompt,
            batch,
        ).strip()
        if reconstructed:
            reconstructed_parts.append(reconstructed)

    joined = "\n\n".join(reconstructed_parts).strip()
    return _clean_reconstruction_artifacts(joined)


def extract_text_from_pdf_with_llm(
    path: Path,
    *,
    llm_client: LLMClient,
    max_input_chars_per_call: int = 30000,
) -> str:
    try:
        logging.info(
            "LLM PDF reconstruction | file=%s | mode=direct_pdf",
            path.name,
        )
        direct_result = llm_client.chat_with_pdf(
            LLM_RECONSTRUCTION_SYSTEM_PROMPT,
            "Reconstruct this PDF into clean structured plain text.",
            path,
        ).strip()
        if direct_result:
            return _clean_reconstruction_artifacts(direct_result)
    except Exception as exc:
        logging.info(
            "Direct PDF LLM mode unavailable; using extracted text fallback | file=%s | error=%s",
            path.name,
            str(exc),
        )

    extracted = extract_text_from_pdf(path)
    if not extracted:
        return ""

    chunks = _split_for_llm(extracted, max_chars=max_input_chars_per_call)
    reconstructed_parts: list[str] = []
    total = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        logging.info(
            "LLM PDF reconstruction | file=%s | chunk=%s/%s | chars=%s",
            path.name,
            idx,
            total,
            len(chunk),
        )
        reconstructed_parts.append(
            _reconstruct_chunk_with_llm(llm_client, chunk, idx, total)
        )

    joined = "\n\n".join(part for part in reconstructed_parts if part).strip()
    return _clean_reconstruction_artifacts(joined)


def build_pdf_text_extractor(
    settings: Settings,
    *,
    reader_mode: str,
) -> Callable[[Path], str]:
    normalized_mode = reader_mode.strip().lower()
    if normalized_mode == "baseline":
        return extract_text_from_pdf

    if normalized_mode not in {"llm", "llm_vision"}:
        raise ValueError(f"Unsupported reader mode: {reader_mode}")

    llm_client = build_llm_client(
        settings.llm_provider,
        deepinfra_api_key=settings.deepinfra_api_key,
        deepinfra_base_url=settings.deepinfra_base_url,
        deepinfra_model=settings.deepinfra_model,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        frequency_penalty=0.4,
    )

    def _extractor(path: Path) -> str:
        try:
            if normalized_mode == "llm_vision":
                return extract_text_from_pdf_with_llm_vision(path, llm_client=llm_client)
            return extract_text_from_pdf_with_llm(path, llm_client=llm_client)
        except Exception as exc:
            logging.warning(
                "LLM PDF reader failed; falling back to baseline extractor | file=%s | error=%s",
                path.name,
                str(exc),
            )
            return extract_text_from_pdf(path)

    return _extractor
