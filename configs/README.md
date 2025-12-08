# Pipeline Configuration Guide

This directory contains YAML configuration files for different RAG pipeline types.

## Configuration Files

### 1. `text_rag.yaml`
Text-only RAG pipeline using markdown files.

**Requirements:**
- Markdown files in each document directory (format: `{pdf_name}_page_{X}.md`)

**Use case:** When you have pre-extracted text from PDFs

### 2. `multimodal_rag.yaml`
Multimodal RAG with vision-based page retrieval + text retrieval.

**Requirements:**
- ColPali embeddings file (`.pt`) in each document directory
- Markdown files in each document directory

**Use case:** Leverage visual understanding for better page retrieval, then text search

### 3. `multimodal_rag_online.yaml`
Multimodal RAG with online PDF parsing.

**Requirements:**
- ColPali embeddings file (`.pt`) in each document directory
- PDF file in each document directory

**Use case:** Same as multimodal_rag but parses PDF pages on-the-fly (slower but no pre-processing needed)

### 4. `multimodal_rag_llm_rerank.yaml`
Multimodal RAG with LLM-based reranking.

**Requirements:**
- ColPali embeddings file (`.pt`) in each document directory
- Markdown files in each document directory

**Use case:** Vision-based page retrieval → text chunk retrieval → LLM reranking → generation. LLM reranker identifies all relevant chunks (not limited by top_k)

## Configuration Schema

```yaml
pipeline:
  type: text_rag | multimodal_rag | multimodal_rag_online | multimodal_rag_llm_rerank
  params:
    # Common parameters
    chunk_size: 512
    chunk_overlap: 128
    llm_model: gemini-2.5-flash
    embedding_model: BAAI/bge-large-en-v1.5

    # Reranker (optional)
    use_reranker: false
    reranker_model: BAAI/bge-reranker-base
    rerank_top_k: null

    # Pipeline-specific parameters
    # TextRAG:
    top_k: 5

    # MultimodalRAG:
    vision_top_k: 10
    text_top_k: 5

    # MultimodalRAGOnline:
    vision_top_k: 10
    text_top_k: 5
    fallback_to_ocr: true

    # MultimodalRAGLLMRerank:
    vision_top_k: 10
    text_top_k: 10
    rerank_llm_model: gemini-2.5-flash-lite
    rerank_top_k: null  # null means return all relevant chunks

inference:
  metrics_output: output/metrics.csv  # Optional
```

## Usage

### Running Inference

```bash
python scripts/run_inference.py \
  --config configs/multimodal_rag.yaml \
  --samples data/MMLongBench-Doc/data/samples.json \
  --output output/predictions.json \
  --doc_root data/MMLongBench-Doc/data/documents
```

### Parameters

- `--config`: Path to YAML configuration file
- `--samples`: Path to samples JSON file (format: `[{doc_id, question, answer, ...}]`)
- `--output`: Path to save predictions JSON
- `--doc_root`: Root directory containing document subdirectories
- `--limit`: (Optional) Limit number of samples for testing

### Document Directory Structure

The script expects documents organized as:

```
doc_root/
├── document1_name/
│   ├── document1_name.pdf      # Required for multimodal_rag_online
│   ├── embeddings.pt           # Required for multimodal pipelines
│   ├── document1_name_page_1.md  # Required for text/multimodal_rag
│   ├── document1_name_page_2.md
│   └── ...
├── document2_name/
│   ├── document2_name.pdf
│   ├── embeddings.pt
│   └── ...
```

### Output Format

The script outputs a JSON file with predictions and retrieval metadata:

```json
[
  {
    "doc_id": "document1.pdf",
    "question": "What is ...?",
    "gold_answer": "The answer is ...",
    "pred_answer": "According to the document ...",
    "doc_type": "Research report",
    "answer_format": "Str",
    "vision_retrieved_pages": [1, 3, 5, 7, 9],
    "text_retrieved_pages": [1, 3, 5],
    "final_pages": [1, 3],
    "num_chunks": 3
  },
  ...
]
```

**Metadata Fields:**

- `vision_retrieved_pages` (multimodal pipelines only): Pages retrieved by ColPali vision retrieval
- `text_retrieved_pages`: Pages retrieved by text retrieval (before reranking)
- `final_pages`: Pages finally sent to LLM (after reranking if enabled, otherwise same as text_retrieved_pages)
- `num_chunks`: Number of text chunks used for generation

**Note:** For `TextRAGPipeline`, only `text_retrieved_pages`, `final_pages`, and `num_chunks` are included (no vision retrieval).

## Example Commands

### Quick test with 10 samples
```bash
python scripts/run_inference.py \
  --config configs/multimodal_rag.yaml \
  --samples data/MMLongBench-Doc/data/samples.json \
  --output output/test_predictions.json \
  --doc_root data/MMLongBench-Doc/data/documents \
  --limit 10
```

### Full evaluation
```bash
python scripts/run_inference.py \
  --config configs/multimodal_rag.yaml \
  --samples data/MMLongBench-Doc/data/samples.json \
  --output output/full_predictions.json \
  --doc_root data/MMLongBench-Doc/data/documents
```

### With reranker enabled
Edit the config file to set:
```yaml
use_reranker: true
rerank_top_k: 3
```

Then run:
```bash
python scripts/run_inference.py \
  --config configs/multimodal_rag.yaml \
  --samples data/samples.json \
  --output output/predictions_reranked.json \
  --doc_root data/documents
```
