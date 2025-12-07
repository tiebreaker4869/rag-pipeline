#!/usr/bin/env python3
"""
Universal inference script for RAG pipelines.

Accepts a YAML configuration file to initialize pipeline type and parameters,
then runs inference on a dataset. Each document gets its own pipeline instance.
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

# Add src to path to import rag_pipeline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag_pipeline.pipelines import (
    TextRAGPipeline,
    MultimodalRAGPipeline,
    MultimodalRAGOnlinePipeline,
    MultimodalRAGLLMRerankPipeline,
    RAGResponse,
)
from rag_pipeline.utils.profile import export_latency


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def create_pipeline(pipeline_type: str, params: Dict[str, Any]):
    """Create pipeline instance from type and parameters.

    Args:
        pipeline_type: Type of pipeline (text_rag, multimodal_rag, multimodal_rag_online, multimodal_rag_llm_rerank)
        params: Pipeline initialization parameters

    Returns:
        Initialized pipeline instance
    """
    if pipeline_type == 'text_rag':
        return TextRAGPipeline(**params)
    elif pipeline_type == 'multimodal_rag':
        return MultimodalRAGPipeline(**params)
    elif pipeline_type == 'multimodal_rag_online':
        return MultimodalRAGOnlinePipeline(**params)
    elif pipeline_type == 'multimodal_rag_llm_rerank':
        return MultimodalRAGLLMRerankPipeline(**params)
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")


def load_samples(path: str, limit: int = None) -> List[Dict[str, Any]]:
    """Load dataset samples from JSON file.

    Args:
        path: Path to samples JSON file
        limit: Optional limit on number of samples to load

    Returns:
        List of sample dictionaries
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if limit is not None and limit > 0:
        data = data[:limit]

    return data


def run_inference(
    config: Dict[str, Any],
    samples: List[Dict[str, Any]],
    output_path: str,
    doc_root: str,
):
    """Run inference on all samples and save predictions.

    Creates one pipeline instance per document.

    Args:
        config: Configuration dictionary
        samples: List of sample dictionaries with questions
        output_path: Path to save prediction results
        doc_root: Root directory containing document subdirectories
    """
    results: List[Dict[str, Any]] = []

    # Get pipeline configuration
    pipeline_config = config.get('pipeline', {})
    pipeline_type = pipeline_config.get('type')
    base_params = pipeline_config.get('params', {})

    if not pipeline_type:
        raise ValueError("Configuration must specify 'pipeline.type'")

    # Group samples by doc_id
    samples_by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for sample in samples:
        doc_id = sample.get('doc_id')
        if not doc_id:
            print(f"Warning: Sample missing doc_id, skipping: {sample.get('question', '')[:50]}")
            continue
        samples_by_doc.setdefault(doc_id, []).append(sample)

    print(f"Processing {len(samples)} samples across {len(samples_by_doc)} documents")
    print(f"Pipeline type: {pipeline_type}")

    # Process each document with its own pipeline
    for doc_idx, (doc_id, doc_samples) in enumerate(samples_by_doc.items(), start=1):
        print(f"\n[Doc {doc_idx}/{len(samples_by_doc)}] {doc_id} -> {len(doc_samples)} questions")

        # Construct document-specific directory
        doc_stem = Path(doc_id).stem
        doc_dir = Path(doc_root) / doc_stem

        if not doc_dir.exists():
            print(f"  [ERROR] Document directory not found: {doc_dir}")
            # Add failed results
            for sample in doc_samples:
                results.append({
                    'doc_id': doc_id,
                    'question': sample['question'],
                    'gold_answer': str(sample.get('answer', '')),
                    'pred_answer': '',
                    'doc_type': sample.get('doc_type', ''),
                    'answer_format': sample.get('answer_format', ''),
                    'error': f'Document directory not found: {doc_dir}',
                })
            continue

        # Create pipeline params for this document
        pipeline_params = base_params.copy()
        pipeline_params['doc_dir'] = str(doc_dir)

        print(f"  Initializing pipeline with doc_dir: {doc_dir}")

        # Create pipeline for this document
        try:
            pipeline = create_pipeline(pipeline_type, pipeline_params)
        except Exception as e:
            print(f"  [ERROR] Failed to initialize pipeline: {e}")
            import traceback
            traceback.print_exc()
            # Add failed results
            for sample in doc_samples:
                results.append({
                    'doc_id': doc_id,
                    'question': sample['question'],
                    'gold_answer': str(sample.get('answer', '')),
                    'pred_answer': '',
                    'doc_type': sample.get('doc_type', ''),
                    'answer_format': sample.get('answer_format', ''),
                    'error': f'Pipeline init failed: {str(e)}',
                })
            continue

        # Run inference on all questions for this document
        for q_idx, sample in enumerate(doc_samples, start=1):
            question = sample['question']
            print(f"  [{q_idx}/{len(doc_samples)}] {question[:60]}...")

            try:
                response = pipeline.query_with_metadata(question)
                pred = response.answer
                metadata = response.metadata
            except Exception as e:
                print(f"    [ERROR] Query failed: {e}")
                pred = ""
                metadata = {}

            # Build result with retrieval metadata
            result = {
                'doc_id': doc_id,
                'question': question,
                'gold_answer': str(sample.get('answer', '')),
                'pred_answer': pred,
                'doc_type': sample.get('doc_type', ''),
                'answer_format': sample.get('answer_format', ''),
                'evidence_pages': sample.get('evidence_pages', '[]'),
                'evidence_sources': sample.get('evidence_sources', '[]'),
            }

            # Add retrieval metadata if available
            if metadata:
                # Vision retrieved pages (only for multimodal pipelines)
                if 'vision_retrieved_pages' in metadata:
                    result['vision_retrieved_pages'] = metadata['vision_retrieved_pages']

                # Text retrieved pages (before reranking)
                if 'text_retrieved_pages' in metadata:
                    result['text_retrieved_pages'] = metadata['text_retrieved_pages']

                # Final pages sent to LLM (after reranking if enabled)
                if 'final_pages' in metadata:
                    result['final_pages'] = metadata['final_pages']

                # Number of chunks used
                if 'num_chunks' in metadata:
                    result['num_chunks'] = metadata['num_chunks']

            results.append(result)

        # Cleanup pipeline for this document
        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved {len(results)} predictions to {output_file}")

    # Export metrics if configured
    inference_config = config.get('inference', {})
    metrics_path = inference_config.get('metrics_output')
    if metrics_path:
        metrics_file = Path(metrics_path)
        metrics_file.parent.mkdir(parents=True, exist_ok=True)

        if metrics_file.suffix == '.csv':
            export_latency(str(metrics_file), format='csv')
        else:
            export_latency(str(metrics_file), format='json')

        print(f"✓ Saved metrics to {metrics_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference using a configured RAG pipeline"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file',
    )
    parser.add_argument(
        '--samples',
        type=str,
        required=True,
        help='Path to samples JSON file',
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save prediction results (JSON)',
    )
    parser.add_argument(
        '--doc_root',
        type=str,
        required=True,
        help='Root directory containing document subdirectories (one per doc)',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of samples for quick testing',
    )

    args = parser.parse_args()

    print("=" * 80)
    print("RAG Pipeline Inference")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Samples: {args.samples}")
    print(f"Output: {args.output}")
    print(f"Document root: {args.doc_root}")
    if args.limit:
        print(f"Sample limit: {args.limit}")
    print("=" * 80)

    # Load configuration
    print("\n[1/3] Loading configuration...")
    config = load_config(args.config)
    print(f"  Pipeline type: {config.get('pipeline', {}).get('type')}")

    # Load samples
    print("\n[2/3] Loading samples...")
    samples = load_samples(args.samples, args.limit)
    print(f"  Loaded {len(samples)} samples")

    # Run inference
    print("\n[3/3] Running inference...")
    run_inference(
        config=config,
        samples=samples,
        output_path=args.output,
        doc_root=args.doc_root,
    )

    print("\n" + "=" * 80)
    print("Inference completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
