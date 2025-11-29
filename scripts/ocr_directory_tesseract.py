#!/usr/bin/env python3
"""
Run Tesseract OCR over all PDFs in a directory (recursively) and save
one TXT per PDF (same path, .txt extension).
"""

import argparse
from pathlib import Path
from typing import List

try:
    import fitz
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyMuPDF (fitz) is required. Install with `pip install pymupdf`."
    ) from exc

from parse.tesseract_parser import TesseractPdfParser


def find_pdfs(root: Path) -> List[Path]:
    return sorted(root.rglob("*.pdf"))


def ocr_pdf(pdf_path: Path, parser: TesseractPdfParser, overwrite: bool) -> None:
    output_path = pdf_path.with_suffix(".txt")
    if output_path.exists() and not overwrite:
        print(f"[SKIP] {output_path} exists (use --overwrite to replace)")
        return

    try:
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        doc.close()
    except Exception as e:
        print(f"[ERROR] Failed to open {pdf_path}: {e}")
        return

    parts = []
    for page_num in range(1, total_pages + 1):
        try:
            content = parser.parse_page(str(pdf_path), page_num)
            parts.append(f"[Page {page_num}/{total_pages}]\n{content.text}\n")
        except Exception as e:
            print(f"[ERROR] {pdf_path} page {page_num}: {e}")
            continue

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))

    print(f"[OK] {pdf_path} -> {output_path} ({len(parts)} pages)")


def main():
    parser = argparse.ArgumentParser(
        description="OCR all PDFs under a directory with Tesseract."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="data/examples",
        help="Root directory containing PDFs (searched recursively).",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="eng",
        help="Tesseract language code (e.g., eng, chi_sim).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Render DPI for PDF pages; higher may improve OCR quality.",
    )
    parser.add_argument(
        "--tesseract-cmd",
        type=str,
        default=None,
        help="Optional path to tesseract executable.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Extra config string passed to tesseract (e.g., '--psm 4').",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Disable grayscale/binarization preprocessing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .txt outputs.",
    )

    args = parser.parse_args()

    root = Path(args.root)
    pdfs = find_pdfs(root)
    if not pdfs:
        print(f"No PDFs found under {root}")
        return

    print(f"Found {len(pdfs)} PDFs under {root}")

    ocr_parser = TesseractPdfParser(
        lang=args.lang,
        tesseract_cmd=args.tesseract_cmd,
        dpi=args.dpi,
        preprocess=not args.no_preprocess,
        config=args.config,
    )

    for pdf_path in pdfs:
        ocr_pdf(pdf_path, ocr_parser, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
