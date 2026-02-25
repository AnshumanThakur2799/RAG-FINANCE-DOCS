from __future__ import annotations

import logging
import re
from pathlib import Path
from functools import lru_cache
import shutil
import os
import json
from collections.abc import Mapping

from markitdown import MarkItDown


def _normalize_text(text: str) -> str:
    return re.sub(r"[ \t]+\n", "\n", text).strip()


def _extract_with_markitdown(path: Path) -> str:
    converter = MarkItDown()
    result = converter.convert(str(path))
    return _normalize_text(result.text_content or "")


def _extract_with_pdfplumber(path: Path) -> str:
    import pdfplumber

    page_texts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_texts.append((page.extract_text(layout=True) or "").strip())
    return _normalize_text("\n\n".join(page_texts))


def _extract_with_pymupdf(path: Path) -> str:
    import fitz

    page_texts: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            page_texts.append((page.get_text("text") or "").strip())
    return _normalize_text("\n\n".join(page_texts))


def _extract_with_tesseract_ocr(path: Path) -> str:
    from pdf2image import convert_from_path
    import pytesseract

    # Skip this extractor when external OCR dependencies are unavailable.
    if not shutil.which("pdftoppm") or not shutil.which("pdfinfo"):
        logging.info(
            "Skipping tesseract OCR | file=%s | reason=poppler_not_found",
            path.name,
        )
        return ""
    if not shutil.which("tesseract"):
        logging.info(
            "Skipping tesseract OCR | file=%s | reason=tesseract_not_found",
            path.name,
        )
        return ""

    page_texts: list[str] = []
    try:
        images = convert_from_path(str(path), dpi=300)
    except Exception as exc:
        logging.info(
            "Skipping tesseract OCR | file=%s | reason=%s",
            path.name,
            str(exc),
        )
        return ""
    for image in images:
        page_texts.append((pytesseract.image_to_string(image) or "").strip())
    return _normalize_text("\n\n".join(page_texts))


@lru_cache(maxsize=1)
def _get_paddle_ocr_engine():
    from paddleocr import PaddleOCR  # pyright: ignore[reportMissingImports]

    # Prefer PaddleOCR 3.x style options for plain OCR inference.
    try:
        return PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except TypeError:
        # Backward-compatible fallback for older PaddleOCR versions.
        return PaddleOCR(use_angle_cls=True, lang="en")


def _extend_lines_from_value(lines: list[str], value: object) -> None:
    if isinstance(value, str):
        text = value.strip()
        if text:
            lines.append(text)
        return
    if isinstance(value, Mapping):
        for nested in value.values():
            _extend_lines_from_value(lines, nested)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _extend_lines_from_value(lines, item)
        return
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            _extend_lines_from_value(lines, tolist())
        except Exception:
            pass


def _collect_strings_deep(value: object) -> list[str]:
    collected: list[str] = []
    stack: list[object] = [value]
    seen_ids: set[int] = set()
    while stack:
        current = stack.pop()
        current_id = id(current)
        if current_id in seen_ids:
            continue
        seen_ids.add(current_id)
        current = _to_plain_python(current)
        if current is None:
            continue
        if isinstance(current, str):
            text = current.strip()
            if text:
                collected.append(text)
            continue
        if isinstance(current, Mapping):
            for nested in current.values():
                stack.append(nested)
            continue
        if isinstance(current, (list, tuple, set)):
            for nested in current:
                stack.append(nested)
            continue
        tolist = getattr(current, "tolist", None)
        if callable(tolist):
            try:
                stack.append(tolist())
            except Exception:
                pass
    return collected


def _to_plain_python(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool, list, tuple, dict)):
        return value
    for method_name in ("to_dict", "model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return method()
            except Exception:
                pass
    for method_name in ("to_json", "json"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                payload = method()
                if isinstance(payload, str):
                    return json.loads(payload)
            except Exception:
                pass
    if hasattr(value, "__dict__"):
        try:
            return vars(value)
        except Exception:
            pass
    return value


def _extract_text_lines_from_paddle_result(result: object) -> list[str]:
    lines: list[str] = []
    stack: list[object] = [_to_plain_python(result)]
    seen_ids: set[int] = set()
    text_keys = {
        "text",
        "texts",
        "rec_text",
        "rec_texts",
        "ocr_text",
        "ocr_texts",
        "transcription",
        "transcriptions",
    }

    while stack:
        current = stack.pop()
        current_id = id(current)
        if current_id in seen_ids:
            continue
        seen_ids.add(current_id)

        current = _to_plain_python(current)
        if current is None:
            continue
        if isinstance(current, str):
            # Keep extraction strict: only collect text from known OCR keys.
            continue
        if isinstance(current, Mapping):
            for key, value in current.items():
                if str(key).lower() in text_keys:
                    _extend_lines_from_value(lines, value)
                else:
                    stack.append(value)
            continue
        if isinstance(current, (list, tuple, set)):
            for item in current:
                stack.append(item)
            continue

    # Backward-compatible handling for classic PaddleOCR output:
    # [[bbox, (text, score)], ...]
    if not lines and isinstance(result, list):
        page_result = result[0] if result else []
        for line in page_result if isinstance(page_result, list) else []:
            if (
                isinstance(line, (list, tuple))
                and len(line) > 1
                and isinstance(line[1], (list, tuple))
                and line[1]
            ):
                lines.append(str(line[1][0]).strip())

    # Last-resort parser for unexpected PaddleOCR 3.x payloads.
    if not lines:
        for text in _collect_strings_deep(_to_plain_python(result)):
            if any(ch.isalnum() for ch in text):
                lines.append(text)

    cleaned: list[str] = []
    seen: set[str] = set()
    for line in lines:
        normalized = re.sub(r"\s+", " ", line).strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)
    return cleaned


def _extract_with_paddle_ocr(path: Path) -> str:
    import fitz
    import numpy as np

    ocr = _get_paddle_ocr_engine()
    page_texts: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=300, alpha=False)
            image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            try:
                if hasattr(ocr, "predict"):
                    try:
                        result = ocr.predict(
                            image,
                            use_doc_orientation_classify=False,
                            use_doc_unwarping=False,
                            use_textline_orientation=False,
                        ) or []
                    except TypeError:
                        result = ocr.predict(image) or []
                else:
                    result = ocr.ocr(image) or []
            except Exception as exc:
                # Known Paddle runtime issue on some Windows CPU setups.
                if "ConvertPirAttribute2RuntimeAttribute" in str(exc):
                    logging.info(
                        "Skipping paddle OCR | file=%s | reason=runtime_unsupported",
                        path.name,
                    )
                    return ""
                raise

            lines = _extract_text_lines_from_paddle_result(result)
            page_texts.append("\n".join(lines).strip())
    return _normalize_text("\n\n".join(page_texts))


def _classify_pdf(path: Path) -> str:
    """
    Returns one of: 'digital', 'scanned', 'mixed'.
    """
    try:
        import pdfplumber
    except Exception:
        # If we cannot inspect page internals, treat as mixed and rely on fallbacks.
        return "mixed"

    total_pages = 0
    pages_with_text = 0
    pages_likely_scanned = 0

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            total_pages += 1
            text = (page.extract_text(layout=False) or "").strip()
            char_count = len(text)
            if char_count >= 100:
                pages_with_text += 1

            page_area = float(page.width * page.height) if page.width and page.height else 0.0
            has_large_image = False
            for image in page.images or []:
                img_w = float(image.get("width") or 0.0)
                img_h = float(image.get("height") or 0.0)
                if page_area > 0 and (img_w * img_h) / page_area >= 0.65:
                    has_large_image = True
                    break

            if char_count < 40 and has_large_image:
                pages_likely_scanned += 1

    if total_pages == 0:
        return "mixed"

    text_ratio = pages_with_text / total_pages
    scanned_ratio = pages_likely_scanned / total_pages
    if text_ratio >= 0.7 and scanned_ratio <= 0.2:
        return "digital"
    if text_ratio <= 0.2 or scanned_ratio >= 0.5:
        return "scanned"
    return "mixed"


def _is_sufficient_text(text: str, min_chars: int = 120) -> bool:
    non_whitespace_chars = len(re.sub(r"\s+", "", text))
    return non_whitespace_chars >= min_chars


def extract_text_from_pdf(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path}")

    pdf_type = _classify_pdf(path)
    logging.info("PDF routing decision | file=%s | detected_type=%s", path.name, pdf_type)

    # Route by detected type, then fallback if output quality is low.
    if pdf_type == "digital":
        extractors = (
            ("pdfplumber", _extract_with_pdfplumber),
            ("pymupdf", _extract_with_pymupdf),
            ("markitdown", _extract_with_markitdown),
            ("ocr_paddle", _extract_with_paddle_ocr),
            ("ocr_tesseract", _extract_with_tesseract_ocr),
        )
    elif pdf_type == "scanned":
        extractors = (
            ("ocr_paddle", _extract_with_paddle_ocr),
            ("ocr_tesseract", _extract_with_tesseract_ocr),
            ("markitdown", _extract_with_markitdown),
            ("pymupdf", _extract_with_pymupdf),
            ("pdfplumber", _extract_with_pdfplumber),
        )
    else:
        extractors = (
            ("pymupdf", _extract_with_pymupdf),
            ("pdfplumber", _extract_with_pdfplumber),
            ("markitdown", _extract_with_markitdown),
            ("ocr_paddle", _extract_with_paddle_ocr),
            ("ocr_tesseract", _extract_with_tesseract_ocr),
        )

    attempted: list[str] = []
    best_text = ""
    for extractor_name, extractor in extractors:
        attempted.append(extractor_name)
        try:
            extracted = extractor(path)
        except Exception as exc:
            logging.warning(
                "PDF extractor failed | file=%s | extractor=%s | error=%s",
                path.name,
                extractor_name,
                str(exc),
            )
            continue

        if len(extracted) > len(best_text):
            best_text = extracted

        if _is_sufficient_text(extracted):
            logging.info(
                "PDF extraction complete | file=%s | extractor=%s | attempted=%s",
                path.name,
                extractor_name,
                ",".join(attempted),
            )
            return extracted

    if best_text:
        logging.info(
            "PDF extraction fallback used | file=%s | attempted=%s",
            path.name,
            ",".join(attempted),
        )
        return best_text

    return ""


# Local extraction smoke test.
if __name__ == "__main__":
    path = Path("app/ingestion/Tendernotice_1 copy.pdf")
    test_extractors = [
        ("extract_text_from_pdf", extract_text_from_pdf),
        ("_extract_with_pdfplumber", _extract_with_pdfplumber),
        ("_extract_with_pymupdf", _extract_with_pymupdf),
        ("_extract_with_markitdown", _extract_with_markitdown),
    ]
    if os.getenv("RUN_OCR_SMOKE", "").strip().lower() in {"1", "true", "yes"}:
        test_extractors.extend(
            [
                ("_extract_with_paddle_ocr", _extract_with_paddle_ocr),
                ("_extract_with_tesseract_ocr", _extract_with_tesseract_ocr),
            ]
        )

    for extractor_name, extractor_fn in test_extractors:
        try:
            text = extractor_fn(path)
            print(f"{extractor_name}: {text}")
        except Exception as exc:
            print(f"{extractor_name} failed: {exc}")