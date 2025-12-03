#!/usr/bin/env python3
"""Test script for MultimodalRAGOnlinePipeline."""

import argparse
import os
import sys

# Add src to path to import rag_pipeline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag_pipeline.pipelines import MultimodalRAGOnlinePipeline
from rag_pipeline.utils.profile import export_latency, get_all_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Test MultimodalRAGOnlinePipeline with interactive queries"
    )
    parser.add_argument(
        "--doc_dir",
        type=str,
        required=True,
        help="Directory containing .pt embeddings and .pdf file",
    )
    parser.add_argument(
        "--vision_top_k",
        type=int,
        default=10,
        help="Number of pages to retrieve using ColPali (default: 10)",
    )
    parser.add_argument(
        "--text_top_k",
        type=int,
        default=5,
        help="Number of text chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="Size of text chunks (default: 512)",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=128,
        help="Overlap between chunks (default: 128)",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="gemini-2.5-flash",
        help="LLM model name (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model name (default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--use_reranker",
        action="store_true",
        help="Enable reranking with BGE cross-encoder",
    )
    parser.add_argument(
        "--reranker_model",
        type=str,
        default="BAAI/bge-reranker-base",
        help="Reranker model name (default: BAAI/bge-reranker-base)",
    )
    parser.add_argument(
        "--rerank_top_k",
        type=int,
        default=None,
        help="Number of chunks after reranking (default: same as text_top_k)",
    )
    parser.add_argument(
        "--no_ocr_fallback",
        action="store_true",
        help="Disable OCR fallback for scanned PDFs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save metrics (default: output)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single query to run (non-interactive mode)",
    )

    args = parser.parse_args()

    # Validate doc_dir
    if not os.path.isdir(args.doc_dir):
        print(f"Error: Directory not found: {args.doc_dir}")
        sys.exit(1)

    print("=" * 80)
    print("MultimodalRAGOnlinePipeline Test Script")
    print("=" * 80)
    print(f"Document directory: {args.doc_dir}")
    print(f"Vision top-k: {args.vision_top_k}")
    print(f"Text top-k: {args.text_top_k}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Chunk overlap: {args.chunk_overlap}")
    print(f"LLM model: {args.llm_model}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Use reranker: {args.use_reranker}")
    if args.use_reranker:
        print(f"Reranker model: {args.reranker_model}")
        print(f"Rerank top-k: {args.rerank_top_k or args.text_top_k}")
    print(f"OCR fallback: {not args.no_ocr_fallback}")
    print("=" * 80)

    # Initialize pipeline
    print("\n[INFO] Initializing MultimodalRAGOnlinePipeline...")
    try:
        pipeline = MultimodalRAGOnlinePipeline(
            doc_dir=args.doc_dir,
            vision_top_k=args.vision_top_k,
            text_top_k=args.text_top_k,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            llm_model=args.llm_model,
            embedding_model=args.embedding_model,
            use_reranker=args.use_reranker,
            reranker_model=args.reranker_model,
            rerank_top_k=args.rerank_top_k,
            fallback_to_ocr=not args.no_ocr_fallback,
        )
        print("[INFO] Pipeline initialized successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Single query mode
    if args.query:
        print(f"\n[QUERY] {args.query}")
        print("-" * 80)
        try:
            answer = pipeline.query(args.query)
            print(f"[ANSWER]\n{answer}")
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")
            import traceback
            traceback.print_exc()

    # Interactive mode
    else:
        print("\n[INFO] Entering interactive mode. Type '/exit' to quit.\n")
        while True:
            try:
                query = input("Query: ").strip()

                if not query:
                    continue

                if query == "/exit":
                    print("Exiting...")
                    break

                if query == "/help":
                    print("\nCommands:")
                    print("  /exit  - Exit the program")
                    print("  /help  - Show this help message")
                    print("  /stats - Show latency statistics")
                    print()
                    continue

                if query == "/stats":
                    metrics = get_all_metrics()
                    if metrics:
                        print("\nLatency Statistics:")
                        print("-" * 80)
                        for metric_name, stats in metrics.items():
                            if stats:
                                print(f"{metric_name}:")
                                print(f"  Count: {stats['count']}")
                                print(f"  Min:   {stats['min']:.4f}s")
                                print(f"  Max:   {stats['max']:.4f}s")
                                print(f"  Mean:  {stats['mean']:.4f}s")
                                print(f"  Total: {stats['total']:.4f}s")
                        print("-" * 80)
                    else:
                        print("No metrics available yet.")
                    print()
                    continue

                # Run query
                print()
                answer = pipeline.query(query)
                print(f"Answer:\n{answer}\n")

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}\n")
                import traceback
                traceback.print_exc()

    # Export metrics
    print("\n[INFO] Exporting metrics...")
    os.makedirs(args.output_dir, exist_ok=True)

    csv_path = os.path.join(args.output_dir, "mm_rag_online_metrics.csv")
    json_path = os.path.join(args.output_dir, "mm_rag_online_metrics.json")

    export_latency(csv_path, format="csv")
    export_latency(json_path, format="json")

    print(f"[INFO] Metrics saved to:")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()