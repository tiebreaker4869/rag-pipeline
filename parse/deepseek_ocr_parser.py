from parse.pdf_parser import PDFParser, PageContent
from transformers import AutoModel, AutoTokenizer
import torch
import fitz
import os
import tempfile

class DeepSeekOcrPdfParser(PDFParser):
    def __init__(self, prompt: str = None):
        model_name = 'deepseek-ai/DeepSeek-OCR'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
        self.model = model.eval().cuda().to(torch.bfloat16)
        if prompt is None:
            self.prompt = '<image>\n<|grounding|>Convert the document to markdown.'
        else:
            self.prompt = prompt

    def parse_page(self, pdf_path: str, page_num: int) -> PageContent:
        """
        Parse a specific page from a PDF file using DeepSeek-OCR

        Args:
            pdf_path: Path to the PDF file
            page_num: Page number (1-indexed)

        Returns:
            PageContent object for the specified page
        """
        doc_id = os.path.basename(pdf_path)
        image_path = None

        try:
            # Convert PDF page to image
            image_path = self._pdf_page_to_image(pdf_path, page_num)

            # Create temporary output directory
            with tempfile.TemporaryDirectory() as output_dir:
                # Run OCR inference
                result = self.model.infer(
                    self.tokenizer,
                    prompt=self.prompt,
                    image_file=image_path,
                    output_path=output_dir,
                    base_size=1024,
                    image_size=640,
                    crop_mode=True,
                    save_results=False,
                    test_compress=False,
                )

            # Extract text from result
            text = result if isinstance(result, str) else str(result)

            # Get total page count
            doc = fitz.open(pdf_path)
            total_pages = doc.page_count
            doc.close()

            metadata = {
                "total_pages": total_pages,
                "model": "deepseek-ai/DeepSeek-OCR",
            }

            return PageContent(
                doc_id=doc_id, page_num=page_num, text=text, metadata=metadata
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to parse page {page_num} from {pdf_path} using DeepSeek-OCR: {e}"
            )

        finally:
            # Clean up temporary image file
            if image_path and os.path.exists(image_path):
                os.unlink(image_path)

    def _pdf_page_to_image(self, pdf_path: str, page_num: int) -> str:
        """
        Convert a PDF page to an image file

        Args:
            pdf_path: Path to the PDF file
            page_num: Page number (1-indexed)

        Returns:
            Path to the temporary image file
        """
        doc = fitz.open(pdf_path)

        if page_num < 1 or page_num > doc.page_count:
            doc.close()
            raise ValueError(
                f"Page {page_num} out of range. Document has {doc.page_count} pages."
            )

        # Load the specific page (PyMuPDF uses 0-indexed)
        page = doc.load_page(page_num - 1)

        # Render page to an image (higher resolution for better OCR)
        # zoom = 2.0 means 2x resolution (144 DPI instead of 72 DPI)
        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".png", prefix=f"page_{page_num}_"
        )
        pix.save(temp_file.name)

        doc.close()

        return temp_file.name