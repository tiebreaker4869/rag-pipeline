import fitz
import os
from parse.pdf_parser import PDFParser, PageContent


class PyMuPDFParser(PDFParser):
    """PDF parser implementation using PyMuPDF (fitz)"""

    def parse_page(self, pdf_path: str, page_num: int) -> PageContent:
        """
        Parse a specific page from a PDF file using PyMuPDF

        Args:
            pdf_path: Path to the PDF file
            page_num: Page number (1-indexed)

        Returns:
            PageContent object for the specified page
        """
        doc_id = os.path.basename(pdf_path)

        try:
            doc = fitz.open(pdf_path)

            if page_num < 1 or page_num > doc.page_count:
                raise ValueError(f"Page {page_num} out of range. Document has {doc.page_count} pages.")

            # Load the specific page (PyMuPDF uses 0-indexed)
            page = doc.load_page(page_num - 1)

            # Extract text
            text = page.get_text()

            # Extract metadata
            metadata = {
                "total_pages": doc.page_count,
                "page_width": page.rect.width,
                "page_height": page.rect.height,
            }

            doc.close()

            return PageContent(
                doc_id=doc_id,
                page_num=page_num,
                text=text,
                metadata=metadata
            )

        except Exception as e:
            raise RuntimeError(f"Failed to parse page {page_num} from {pdf_path}: {e}")
