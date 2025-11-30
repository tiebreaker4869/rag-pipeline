#!/usr/bin/env python3
"""
Evaluate text-only RAG pipeline against MMLongBench samples.

Uses TextRAGPipeline to build an index over specified PDFs, then runs
each QA sample and records predictions + simple matching metrics.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from pipeline.vector_pipeline import TextRAGPipeline


def load_samples(path: str, limit: int | None) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    if limit is not None:
        data = data[:limit]
    return data


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def evaluate(
    samples_path: str,
    doc_dir: str,
    output_path: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    generation_model: str,
    embedding_model: str,
    limit: int | None,
):
    samples = load_samples(samples_path, limit)
    if not samples:
        raise SystemExit("No samples loaded.")

    results: List[Dict[str, Any]] = []

    # Group samples by doc_id to build per-document indexes
    samples_by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for sample in samples:
        samples_by_doc.setdefault(sample["doc_id"], []).append(sample)

    print(
        f"Loaded {len(samples)} samples across {len(samples_by_doc)} documents. "
        "Building per-document indexes..."
    )

    for doc_idx, (doc_id, doc_samples) in enumerate(samples_by_doc.items(), start=1):
        print(f"[Doc {doc_idx}/{len(samples_by_doc)}] {doc_id} -> {len(doc_samples)} questions")

        pipeline = TextRAGPipeline(
            doc_dir=doc_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            llm_model=generation_model,
            embedding_model=embedding_model,
            doc_filter=[doc_id],
            single_doc_mode=True,
        )

        for sample in doc_samples:
            question = sample["question"]
            gold = str(sample.get("answer", ""))

            print(f"  Q: {question[:80]}...")
            try:
                pred = pipeline.query(question, doc_id=doc_id)
            except Exception as e:
                print(f"  [ERROR] {e}")
                pred = ""

            results.append(
                {
                    "doc_id": doc_id,
                    "question": question,
                    "gold_answer": gold,
                    "pred_answer": pred,
                }
            )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved results to {output_file} (JSON)")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate text RAG pipeline on MMLongBench samples."
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="data/MMLongBench-Doc/data/samples.json",
        help="Path to samples.json",
    )
    parser.add_argument(
        "--doc_dir",
        type=str,
        default="data/MMLongBench-Doc/data/documents",
        help="Directory containing PDF documents.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/text_rag_eval.json",
        help="Predictions JSON output file.",
    )
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--chunk_overlap", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--generation_model", type=str, default="gemini-1.5-flash"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="Embedding model name (OpenAI text-embedding-3-* or HF model).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of samples for a quick run."
    )

    args = parser.parse_args()

    evaluate(
        samples_path=args.samples,
        doc_dir=args.doc_dir,
        output_path=args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        generation_model=args.generation_model,
        embedding_model=args.embedding_model,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
