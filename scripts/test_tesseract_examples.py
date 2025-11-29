#!/usr/bin/env python3
"""
Quick sanity check for TesseractPdfParser on sample PDFs under data/examples.
"""

import argparse
from pathlib import Path
import textwrap

try:
    import fitz
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyMuPDF (fitz) is required. Install with `pip install pymupdf`."
    ) from exc
from parse.tesseract_parser import TesseractPdfParser


def summarize(text: str, max_len: int) -> str:
    """Condense whitespace and cap length."""
    normalized = " ".join(text.split())
    if len(normalized) > max_len:
        return normalized[:max_len] + "..."
    return normalized


def main():
    parser = argparse.ArgumentParser(
        description="Run Tesseract OCR on sample PDFs under data/examples."
    )
    parser.add_argument(
        "--examples-dir",
        type=str,
        default="data/examples",
        help="Directory containing example PDFs.",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=1,
        help="Number of pages to OCR per document (starting from page 1).",
    )
    parser.add_argument(
        "--lang", type=str, default="eng", help="Tesseract language code."
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
        "--snippet-len",
        type=int,
        default=1024,
        help="Max characters to show per page snippet.",
    )

    args = parser.parse_args()

    examples_dir = Path(args.examples_dir)
    pdf_paths = sorted(examples_dir.rglob("*.pdf"))

    if not pdf_paths:
        print(f"No PDFs found under {examples_dir}")
        return

    parser = TesseractPdfParser(
        lang=args.lang,
        tesseract_cmd=args.tesseract_cmd,
        dpi=args.dpi,
        preprocess=not args.no_preprocess,
        config=args.config,
    )

    print(f"Found {len(pdf_paths)} PDFs under {examples_dir}\n")

    for pdf_path in pdf_paths:
        print(f"=== {pdf_path} ===")
        try:
            doc = fitz.open(pdf_path)
            total_pages = doc.page_count
            doc.close()
        except Exception as e:
            print(f"  [ERROR] Could not open PDF: {e}")
            continue

        max_pages = min(total_pages, args.pages)
        for page_num in range(1, max_pages + 1):
            try:
                content = parser.parse_page(str(pdf_path), page_num)
                snippet = summarize(content.text, args.snippet_len)
                print(
                    f"  Page {page_num}/{total_pages}: "
                    f"chars={len(content.text)}, snippet="
                )
                print(textwrap.indent(snippet, prefix="    "))
            except Exception as e:
                print(f"  [ERROR] Page {page_num}: {e}")
                break

        print()


if __name__ == "__main__":
    main()
