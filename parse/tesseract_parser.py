import io
import os
from typing import Optional

import fitz
import pytesseract
from PIL import Image

from parse.pdf_parser import PDFParser, PageContent


class TesseractPdfParser(PDFParser):
    """PDF parser that renders pages to images and runs Tesseract OCR."""

    def __init__(
        self,
        lang: str = "eng",
        tesseract_cmd: Optional[str] = None,
        dpi: int = 300,
        preprocess: bool = False,
        config: Optional[str] = None,
    ):
        """
        Args:
            lang: Tesseract language code (e.g., "eng", "chi_sim").
            tesseract_cmd: Optional path to tesseract executable.
            dpi: Render resolution for PDF pages; higher improves OCR quality.
            preprocess: Whether to apply simple grayscale/binarization before OCR.
            config: Extra config string passed to pytesseract (e.g., "--psm 4").
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        self.lang = lang
        self.dpi = dpi
        self.preprocess = preprocess
        self.config = config

    def parse_page(self, pdf_path: str, page_num: int) -> PageContent:
        """
        Parse a specific page from a PDF file using Tesseract OCR.

        Args:
            pdf_path: Path to the PDF file.
            page_num: Page number (1-indexed).

        Returns:
            PageContent object for the specified page.
        """
        doc_id = os.path.basename(pdf_path)
        doc = None

        try:
            doc = fitz.open(pdf_path)

            if page_num < 1 or page_num > doc.page_count:
                raise ValueError(
                    f"Page {page_num} out of range. Document has {doc.page_count} pages."
                )

            page = doc.load_page(page_num - 1)
            zoom = self.dpi / 72 if self.dpi else 2.0
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))

            image = Image.open(io.BytesIO(pix.tobytes("png")))
            if self.preprocess:
                image = self._preprocess_image(image)

            text = pytesseract.image_to_string(
                image, lang=self.lang, config=self.config
            )

            metadata = {
                "total_pages": doc.page_count,
                "ocr_engine": "tesseract",
                "lang": self.lang,
                "dpi": self.dpi,
            }

            return PageContent(
                doc_id=doc_id, page_num=page_num, text=text, metadata=metadata
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to parse page {page_num} from {pdf_path} using Tesseract: {e}"
            )
        finally:
            if doc is not None:
                doc.close()

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Lightweight preprocessing to improve OCR quality."""
        gray = image.convert("L")
        # Simple binarization; adjust threshold if needed
        return gray.point(lambda x: 255 if x > 180 else 0, mode="1")
