#!/usr/bin/env python3
"""
Evaluate the SimpleRAGPipeline (vision + hybrid retrieval) on samples.

Outputs a JSON list with: doc_id, question, gold_answer, pred_answer.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from pipeline.baseline_pipeline import SimpleRAGPipeline
from utils.profile import export_latency


def load_samples(path: str, limit: int | None) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if limit is not None:
        data = data[:limit]
    return data


def evaluate(
    samples_path: str,
    embedding_dir: str,
    doc_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    vision_k: int,
    keyword_k: int,
    dense_k: int,
    final_k: int,
    keyword_weight: float,
    generation_model: str,
    embedding_model: str,
    output_path: str,
    metrics_path: str,
    limit: int | None,
):
    samples = load_samples(samples_path, limit)
    if not samples:
        raise SystemExit("No samples loaded.")

    results: List[Dict[str, Any]] = []

    # Group by document for per-doc indexing
    samples_by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for sample in samples:
        doc_id = sample.get("doc_id")
        samples_by_doc.setdefault(doc_id, []).append(sample)

    print(
        f"Loaded {len(samples)} samples across {len(samples_by_doc)} documents. "
        "Building per-document pipelines..."
    )

    base_doc_root = Path(doc_dir)
    base_emb_root = Path(embedding_dir)

    for doc_idx, (doc_id, doc_samples) in enumerate(samples_by_doc.items(), start=1):
        doc_stem = Path(doc_id).stem
        doc_folder = base_doc_root / doc_stem
        pdf_path = doc_folder / doc_id
        emb_folder = base_emb_root / doc_stem

        if not pdf_path.exists():
            print(f"[WARN] PDF not found for {doc_id} at {pdf_path}, skipping.")
            continue
        if not emb_folder.exists():
            print(f"[WARN] Embedding folder not found for {doc_id} at {emb_folder}, skipping.")
            continue

        print(f"[Doc {doc_idx}/{len(samples_by_doc)}] {doc_id} -> {len(doc_samples)} questions")

        rag = SimpleRAGPipeline(
            vision_embedding_dir=str(emb_folder),
            doc_dir=str(doc_folder),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            vision_topk=vision_k,
            keyword_k=keyword_k,
            dense_k=dense_k,
            final_k=final_k,
            hybrid_weights=[keyword_weight, 1 - keyword_weight],
            model=generation_model,
            embedding_model=embedding_model,
        )

        for sample in doc_samples:
            question = sample["question"]
            gold = str(sample.get("answer", ""))

            print(f"  Q: {question[:80]}...")
            try:
                pred = rag.query(question)
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

    print(f"Saved results to {output_file}")

    metrics_file = Path(metrics_path)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    export_latency(str(metrics_file), format="csv" if metrics_file.suffix == ".csv" else "json")
    print(f"Saved profiling metrics to {metrics_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SimpleRAGPipeline on samples."
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="data/MMLongBench-Doc/data/samples.json",
        help="Path to samples.json",
    )
    parser.add_argument(
        "--embedding_dir",
        type=str,
        required=True,
        help="Directory containing vision embeddings (.pt files).",
    )
    parser.add_argument(
        "--doc_dir",
        type=str,
        default="data/MMLongBench-Doc/data/documents",
        help="Directory containing PDF documents.",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=512, help="Chunk size for text splitter."
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=128,
        help="Chunk overlap for text splitter.",
    )
    parser.add_argument(
        "--vision_k", type=int, default=10, help="Top-k pages from vision retriever."
    )
    parser.add_argument(
        "--keyword_k", type=int, default=20, help="BM25 top-k for hybrid retrieval."
    )
    parser.add_argument(
        "--dense_k", type=int, default=20, help="Dense retrieval top-k."
    )
    parser.add_argument(
        "--final_k", type=int, default=5, help="Final fused top-k before rerank."
    )
    parser.add_argument(
        "--keyword_weight",
        type=float,
        default=0.5,
        help="Weight for keyword (BM25) in RRF fusion; dense gets (1-weight).",
    )
    parser.add_argument(
        "--generation_model",
        type=str,
        default="gemini-1.5-flash",
        help="LLM model name for generation.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="Embedding model for dense retrieval.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/simple_rag_eval.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="output/simple_rag_metrics.csv",
        help="Profiling metrics output path (json or csv).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples for quick evaluation.",
    )

    args = parser.parse_args()

    evaluate(
        samples_path=args.samples,
        embedding_dir=args.embedding_dir,
        doc_dir=args.doc_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        vision_k=args.vision_k,
        keyword_k=args.keyword_k,
        dense_k=args.dense_k,
        final_k=args.final_k,
        keyword_weight=args.keyword_weight,
        generation_model=args.generation_model,
        embedding_model=args.embedding_model,
        output_path=args.output,
        metrics_path=args.metrics,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
