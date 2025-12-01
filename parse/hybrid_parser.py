import os
from typing import Dict, Tuple
from parse.pdf_parser import PDFParser, PageContent
from parse.pymupdf_parser import PyMuPDFParser
from parse.tesseract_parser import TesseractPdfParser


class HybridPdfParser(PDFParser):
    """
    Hybrid PDF parser that tries PyMuPDF first, falls back to Tesseract OCR if needed.

    Strategy:
    1. Try PyMuPDF (fast, accurate for native PDFs)
    2. If text is too short (< min_text_length chars), use Tesseract OCR fallback
    3. Cache parsed results to avoid re-parsing the same page
    """

    def __init__(
        self,
        min_text_length: int = 50,
        tesseract_lang: str = "eng",
        tesseract_dpi: int = 300,
        enable_cache: bool = True,
    ):
        """
        Args:
            min_text_length: Minimum text length to consider valid extraction.
                           If PyMuPDF extracts less than this, fallback to OCR.
            tesseract_lang: Language code for Tesseract OCR (default: "eng")
            tesseract_dpi: DPI for Tesseract rendering (default: 300)
            enable_cache: Whether to cache parsed results (default: True)
        """
        self.min_text_length = min_text_length
        self.pymupdf_parser = PyMuPDFParser()
        self.tesseract_parser = TesseractPdfParser(
            lang=tesseract_lang, dpi=tesseract_dpi
        )
        self.enable_cache = enable_cache
        # Cache: (pdf_path, page_num) -> PageContent
        self._cache: Dict[Tuple[str, int], PageContent] = {}

    def parse_page(self, pdf_path: str, page_num: int) -> PageContent:
        """
        Parse a specific page from a PDF file using hybrid approach.

        Args:
            pdf_path: Path to the PDF file
            page_num: Page number (1-indexed)

        Returns:
            PageContent object for the specified page
        """
        # Check cache first
        cache_key = (pdf_path, page_num)
        if self.enable_cache and cache_key in self._cache:
            cached_content = self._cache[cache_key]
            # Mark as from cache
            if cached_content.metadata is None:
                cached_content.metadata = {}
            cached_content.metadata["from_cache"] = True
            return cached_content

        # Try PyMuPDF first
        try:
            page_content = self.pymupdf_parser.parse_page(pdf_path, page_num)

            # Check if extracted text is sufficient
            if len(page_content.text.strip()) >= self.min_text_length:
                # Add parser info to metadata
                if page_content.metadata is None:
                    page_content.metadata = {}
                page_content.metadata["parser"] = "pymupdf"
                page_content.metadata["from_cache"] = False

                # Cache the result
                if self.enable_cache:
                    self._cache[cache_key] = page_content

                return page_content

            # Text too short, likely a scanned page
            print(
                f"[INFO] Page {page_num} has only {len(page_content.text.strip())} chars, "
                f"falling back to Tesseract OCR"
            )

        except Exception as e:
            print(
                f"[WARN] PyMuPDF failed for page {page_num}: {e}, trying Tesseract OCR"
            )

        # Fallback to Tesseract OCR
        try:
            page_content = self.tesseract_parser.parse_page(pdf_path, page_num)
            # Add parser info to metadata
            if page_content.metadata is None:
                page_content.metadata = {}
            page_content.metadata["parser"] = "tesseract_fallback"
            page_content.metadata["from_cache"] = False

            # Cache the result
            if self.enable_cache:
                self._cache[cache_key] = page_content

            return page_content

        except Exception as e:
            raise RuntimeError(
                f"Both PyMuPDF and Tesseract failed for page {page_num} from {pdf_path}: {e}"
            )

    def clear_cache(self):
        """Clear the parsing cache to free memory."""
        self._cache.clear()

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "cached_pages": len(self._cache),
            "cache_enabled": self.enable_cache,
        }
