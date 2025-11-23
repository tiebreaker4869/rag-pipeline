from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass


@dataclass
class PageContent:
    """Represents the content of a single page"""

    doc_id: str
    page_num: int
    text: str
    metadata: Optional[dict] = None


class PDFParser(ABC):
    """Abstract base class for PDF parsers"""

    @abstractmethod
    def parse_page(self, pdf_path: str, page_num: int) -> PageContent:
        """
        Parse a specific page from a PDF file

        Args:
            pdf_path: Path to the PDF file
            page_num: Page number (1-indexed)

        Returns:
            PageContent object for the specified page
        """
        pass
