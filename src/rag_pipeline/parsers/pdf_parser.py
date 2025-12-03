from abc import ABC, abstractmethod
from typing import Optional, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed


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

    def batch_process(
        self,
        pdf_path: str,
        page_nums: Optional[List[int]] = None,
        max_workers: Optional[int] = None
    ) -> List[PageContent]:
        """
        Parse multiple pages from a PDF file in batch with parallel processing.

        This default implementation parallelizes parse_page calls using ThreadPoolExecutor.
        Subclasses can override this method to provide more efficient batch processing
        (e.g., using GPU batch inference for OCR models).

        Args:
            pdf_path: Path to the PDF file
            page_nums: List of page numbers to process (1-indexed).
                      If None, processes all pages in the document.
            max_workers: Maximum number of parallel workers.
                        If None, defaults to min(32, (os.cpu_count() or 1) + 4)

        Returns:
            List of PageContent objects in the same order as page_nums
        """
        import fitz  # Import here to avoid dependency in base class

        # If page_nums is None, get all pages
        if page_nums is None:
            doc = fitz.open(pdf_path)
            page_nums = list(range(1, doc.page_count + 1))
            doc.close()

        # Process pages in parallel
        results = [None] * len(page_nums)  # Pre-allocate to maintain order

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self.parse_page, pdf_path, page_num): idx
                for idx, page_num in enumerate(page_nums)
            }

            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                page_num = page_nums[idx]
                try:
                    content = future.result()
                    results[idx] = content
                except Exception as e:
                    print(f"Error parsing page {page_num} from {pdf_path}: {e}")
                    # Add empty content on error
                    results[idx] = PageContent(
                        doc_id=pdf_path,
                        page_num=page_num,
                        text="",
                        metadata={"error": str(e)}
                    )

        return results
