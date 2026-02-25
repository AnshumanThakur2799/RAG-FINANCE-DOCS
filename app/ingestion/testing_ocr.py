from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test scanned-PDF OCR using pdf-craft and export Markdown."
    )
    parser.add_argument(
        "--pdf",
        default="app/ingestion/Tendernotice_1.pdf",
        help="Input PDF path.",
    )
    parser.add_argument(
        "--out-dir",
        default="ocr_test_output",
        help="Output directory for markdown and assets.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Render DPI for PDF pages.",
    )
    parser.add_argument(
        "--ocr-size",
        default="base",
        choices=["tiny", "small", "base", "large", "gundam"],
        help="DeepSeek OCR model size used by pdf-craft.",
    )
    parser.add_argument(
        "--preview-lines",
        type=int,
        default=40,
        help="Print first N lines of generated markdown.",
    )
    parser.add_argument(
        "--poppler-path",
        default=None,
        help="Optional path to Poppler bin folder (contains pdfinfo/pdftoppm).",
    )
    parser.add_argument(
        "--ignore-pdf-errors",
        action="store_true",
        help="Continue even when page rendering fails. Default is strict mode.",
    )
    parser.add_argument(
        "--ignore-ocr-errors",
        action="store_true",
        help="Continue even when OCR errors occur.",
    )
    return parser


def _check_poppler_binaries(poppler_path: str | None) -> None:
    if poppler_path:
        bin_dir = Path(poppler_path).expanduser().resolve()
        pdfinfo = bin_dir / "pdfinfo.exe"
        pdftoppm = bin_dir / "pdftoppm.exe"
        print(f"[preflight] Using custom poppler path: {bin_dir}")
        print(f"[preflight] pdfinfo exists: {pdfinfo.exists()}")
        print(f"[preflight] pdftoppm exists: {pdftoppm.exists()}")
        return

    pdfinfo = shutil.which("pdfinfo")
    pdftoppm = shutil.which("pdftoppm")
    print(f"[preflight] pdfinfo in PATH: {bool(pdfinfo)}")
    print(f"[preflight] pdftoppm in PATH: {bool(pdftoppm)}")
    if pdfinfo:
        print(f"[preflight] pdfinfo path: {pdfinfo}")
    if pdftoppm:
        print(f"[preflight] pdftoppm path: {pdftoppm}")


def _sanity_check_pdf_render(pdf_path: Path) -> None:
    try:
        import fitz

        with fitz.open(pdf_path) as doc:
            page_count = len(doc)
            print(f"[preflight] PyMuPDF opened PDF successfully. pages={page_count}")
            if page_count > 0:
                pix = doc[0].get_pixmap(dpi=150, alpha=False)
                print(
                    f"[preflight] First page render ok. size={pix.width}x{pix.height}, channels={pix.n}"
                )
    except Exception as exc:
        print(f"[preflight] PyMuPDF render check failed: {exc}")


def main() -> None:
    args = build_parser().parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = out_dir / f"{pdf_path.stem}.md"
    assets_dir = out_dir / f"{pdf_path.stem}_assets"
    analysis_dir = out_dir / f"{pdf_path.stem}_analysis"
    models_dir = out_dir / "models_cache"

    try:
        from pdf_craft import transform_markdown
    except ImportError as exc:
        raise RuntimeError(
            "pdf-craft is not installed. Install it with:\n"
            "  pip install pdf-craft\n"
            "Also ensure Poppler is installed and available in PATH."
        ) from exc

    print(f"[pdf-craft] Input PDF: {pdf_path}")
    print(f"[pdf-craft] Output markdown: {markdown_path}")
    print(f"[pdf-craft] Output assets: {assets_dir}")
    _check_poppler_binaries(args.poppler_path)
    _sanity_check_pdf_render(pdf_path)

    kwargs = {
        "pdf_path": str(pdf_path),
        "markdown_path": str(markdown_path),
        "markdown_assets_path": str(assets_dir),
        "analysing_path": str(analysis_dir),
        "models_cache_path": str(models_dir),
        "ocr_size": args.ocr_size,
        "dpi": args.dpi,
        "ignore_pdf_errors": args.ignore_pdf_errors,
        "ignore_ocr_errors": args.ignore_ocr_errors,
    }
    if args.poppler_path:
        from pdf_craft import DefaultPDFHandler

        kwargs["pdf_handler"] = DefaultPDFHandler(poppler_path=args.poppler_path)

    transform_markdown(**kwargs)

    if not markdown_path.exists():
        raise RuntimeError("Conversion completed but markdown output not found.")

    print("\n[pdf-craft] Conversion completed.")
    print(f"[pdf-craft] Markdown saved at: {markdown_path}")
    print("\n--- Markdown preview ---")

    lines = markdown_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    preview = lines[: max(args.preview_lines, 0)]
    if not preview:
        print("(markdown is empty)")
    else:
        for idx, line in enumerate(preview, start=1):
            print(f"{idx:03d}: {line}")


if __name__ == "__main__":
    main()
