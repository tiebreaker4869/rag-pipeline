import os
import argparse
from typing import List, Optional

from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from pdf2image import convert_from_path
from PIL import Image


def ocr_pdf_with_deepseek(
    pdf_path: str,
    max_pages: Optional[int] = None,
    batch_size: int = 8,
    llm: Optional[LLM] = None,
) -> List[str]:
    # 1) PDF -> PIL.Image list
    pages: List[Image.Image] = convert_from_path(pdf_path)
    if max_pages is not None:
        pages = pages[:max_pages]
    pages = [p.convert("RGB") for p in pages]

    if not pages:
        return []

    if llm is None:
        llm = LLM(
            model="deepseek-ai/DeepSeek-OCR",
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            logits_processors=[NGramPerReqLogitsProcessor],
        )

    sampling_param = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        extra_args=dict(
            ngram_size=30,
            window_size=90,
            whitelist_token_ids={128821, 128822},  # whitelist: <td>, </td>
        ),
        skip_special_tokens=False,
    )

    prompt = "<image>\n<|grounding|>Convert the document to markdown."
    all_results: List[str] = []

    num_pages = len(pages)
    for start in range(0, num_pages, batch_size):
        end = min(start + batch_size, num_pages)
        batch_imgs = pages[start:end]

        model_input = [
            {
                "prompt": prompt,
                "multi_modal_data": {"image": img},
            }
            for img in batch_imgs
        ]

        batch_outputs = llm.generate(model_input, sampling_param)

        for output in batch_outputs:
            text = output.outputs[0].text
            all_results.append(text)

    return all_results


def process_pdf_and_save_pages(
    pdf_path: str,
    llm: LLM,
    max_pages: Optional[int] = None,
    batch_size: int = 8,
):
    print(f"[INFO] Processing PDF: {pdf_path}")
    page_texts = ocr_pdf_with_deepseek(
        pdf_path=pdf_path,
        max_pages=max_pages,
        batch_size=batch_size,
        llm=llm,
    )

    if not page_texts:
        print(f"[WARN] No pages OCR result for {pdf_path}")
        return

    pdf_dir = os.path.dirname(pdf_path)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    for idx, text in enumerate(page_texts, start=1):
        md_filename = f"{pdf_name}_page_{idx}.md"
        md_path = os.path.join(pdf_dir, md_filename)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[INFO] Saved: {md_path}")


def find_all_pdfs(root_dir: str) -> List[str]:
    pdf_files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for name in filenames:
            if name.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(dirpath, name))
    return pdf_files


def main():
    parser = argparse.ArgumentParser(
        description="Run DeepSeek-OCR on all PDFs under a directory (page-level, save to .md)."
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="Root directory that contains subdirectories with PDFs.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional: maximum number of pages per PDF to process.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size of pages for vLLM generation.",
    )
    args = parser.parse_args()

    root_dir = args.root_dir
    max_pages = args.max_pages
    batch_size = args.batch_size

    pdf_files = find_all_pdfs(root_dir)
    if not pdf_files:
        print(f"[WARN] No PDF files found under {root_dir}")
        return

    print(f"[INFO] Found {len(pdf_files)} PDF(s) under {root_dir}")

    llm = LLM(
        model="deepseek-ai/DeepSeek-OCR",
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        logits_processors=[NGramPerReqLogitsProcessor],
    )

    for pdf_path in pdf_files:
        try:
            process_pdf_and_save_pages(
                pdf_path=pdf_path,
                llm=llm,
                max_pages=max_pages,
                batch_size=batch_size,
            )
        except Exception as e:
            print(f"[ERROR] Failed to process {pdf_path}: {e}")


if __name__ == "__main__":
    main()
