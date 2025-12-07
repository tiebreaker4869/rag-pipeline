#!/usr/bin/env python3
"""
Analyze retrieval recall at page level.

Calculates recall for:
- Vision retrieval (for multimodal pipelines)
- Text retrieval (before reranking)
- Final retrieval (after reranking)

Recall = (# evidence pages retrieved) / (# total evidence pages)

Also breaks down by evidence source and page count.
"""

import argparse
import json
import ast
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def load_json(path: str) -> List[Dict[str, Any]]:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_list_field(field) -> List:
    """Parse a field that could be a list or string representation of a list."""
    if isinstance(field, list):
        return field
    if isinstance(field, str):
        try:
            return ast.literal_eval(field)
        except:
            return []
    return []


def calculate_recall(evidence_pages: List[int], retrieved_pages: List[int]) -> float:
    """
    Calculate page-level recall.

    Recall = |evidence_pages ∩ retrieved_pages| / |evidence_pages|
    """
    if not evidence_pages:
        return 0.0

    evidence_set = set(evidence_pages)
    retrieved_set = set(retrieved_pages) if retrieved_pages else set()

    overlap = evidence_set & retrieved_set
    recall = len(overlap) / len(evidence_set)

    return recall


def analyze_retrieval_recall(
    inference_results: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Analyze retrieval recall by category.

    Returns:
        {
            'overall': {stage: {metric: value}},
            'by_source': {source: {stage: {metric: value}}},
            'by_page_count': {category: {stage: {metric: value}}}
        }
    """
    gt_map = {(s['doc_id'], s['question']): s for s in ground_truth}

    # Initialize stats collectors
    overall_stats = defaultdict(lambda: {'recalls': [], 'perfect': 0, 'partial': 0, 'miss': 0, 'total': 0})
    source_stats = defaultdict(lambda: defaultdict(lambda: {'recalls': [], 'perfect': 0, 'partial': 0, 'miss': 0, 'total': 0}))
    page_count_stats = defaultdict(lambda: defaultdict(lambda: {'recalls': [], 'perfect': 0, 'partial': 0, 'miss': 0, 'total': 0}))

    for result in inference_results:
        doc_id = result.get('doc_id')
        question = result.get('question')

        gt_sample = gt_map.get((doc_id, question))
        if not gt_sample:
            continue

        # Get ground truth info
        evidence_pages = parse_list_field(gt_sample.get('evidence_pages', []))
        evidence_sources = parse_list_field(gt_sample.get('evidence_sources', []))
        answer = gt_sample.get('answer', '')

        # Skip if no evidence pages
        if not evidence_pages:
            continue

        # Get retrieval results
        vision_pages = parse_list_field(result.get('vision_retrieved_pages')) if result.get('vision_retrieved_pages') is not None else None
        text_pages = parse_list_field(result.get('text_retrieved_pages')) if result.get('text_retrieved_pages') is not None else None
        final_pages = parse_list_field(result.get('final_pages')) if result.get('final_pages') is not None else None

        # Categorize by page count
        if answer == 'Not answerable':
            page_category = 'unanswerable'
        elif len(evidence_pages) == 1:
            page_category = 'single_page'
        else:
            page_category = 'multi_page'

        # Calculate recalls for each stage
        stages = {}
        if vision_pages is not None:
            stages['vision'] = vision_pages
        if text_pages is not None:
            stages['text'] = text_pages
        if final_pages is not None:
            stages['final'] = final_pages

        for stage_name, retrieved_pages in stages.items():
            recall = calculate_recall(evidence_pages, retrieved_pages)

            # Classify as perfect/partial/miss
            if recall == 1.0:
                category = 'perfect'
            elif recall > 0.0:
                category = 'partial'
            else:
                category = 'miss'

            # Update overall stats
            overall_stats[stage_name]['recalls'].append(recall)
            overall_stats[stage_name][category] += 1
            overall_stats[stage_name]['total'] += 1

            # Update page count stats
            page_count_stats[page_category][stage_name]['recalls'].append(recall)
            page_count_stats[page_category][stage_name][category] += 1
            page_count_stats[page_category][stage_name]['total'] += 1

            # Update source stats
            for source in evidence_sources:
                source_stats[source][stage_name]['recalls'].append(recall)
                source_stats[source][stage_name][category] += 1
                source_stats[source][stage_name]['total'] += 1

    # Calculate average recalls
    def compute_averages(stats):
        result = {}
        for key, metrics in stats.items():
            if isinstance(metrics, dict) and 'recalls' in metrics:
                recalls = metrics['recalls']
                avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
                result[key] = {
                    'avg_recall': avg_recall,
                    'perfect': metrics['perfect'],
                    'partial': metrics['partial'],
                    'miss': metrics['miss'],
                    'total': metrics['total'],
                }
            else:
                # Nested dict (for source/page_count stats)
                result[key] = compute_averages(metrics)
        return result

    return {
        'overall': compute_averages(overall_stats),
        'by_source': compute_averages(source_stats),
        'by_page_count': compute_averages(page_count_stats),
    }


def print_stage_stats(stats: Dict[str, Any], stage_name: str, indent: str = "  "):
    """Print stats for a retrieval stage."""
    if stage_name not in stats:
        return

    stage_stats = stats[stage_name]
    avg_recall = stage_stats['avg_recall']
    perfect = stage_stats['perfect']
    partial = stage_stats['partial']
    miss = stage_stats['miss']
    total = stage_stats['total']

    perfect_pct = (perfect / total * 100) if total > 0 else 0.0
    partial_pct = (partial / total * 100) if total > 0 else 0.0
    miss_pct = (miss / total * 100) if total > 0 else 0.0

    stage_label = {
        'vision': 'Vision Retrieval',
        'text': 'Text Retrieval',
        'final': 'After Reranking',
    }.get(stage_name, stage_name)

    print(f"{indent}{stage_label}:")
    print(f"{indent}  Avg Recall: {avg_recall*100:.1f}%")
    print(f"{indent}  Perfect (100%): {perfect:3d} ({perfect_pct:5.1f}%)")
    print(f"{indent}  Partial (>0%):  {partial:3d} ({partial_pct:5.1f}%)")
    print(f"{indent}  Miss (0%):      {miss:3d} ({miss_pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze retrieval recall at page level"
    )
    parser.add_argument(
        '--inference_results',
        type=str,
        required=True,
        help='Path to inference results JSON (output from run_inference.py)',
    )
    parser.add_argument(
        '--ground_truth',
        type=str,
        required=True,
        help='Path to ground truth samples JSON',
    )

    args = parser.parse_args()

    # Load data
    print("Loading inference results...")
    inference_results = load_json(args.inference_results)

    print("Loading ground truth...")
    ground_truth = load_json(args.ground_truth)

    # Analyze
    print("\nAnalyzing retrieval recall...")
    results = analyze_retrieval_recall(inference_results, ground_truth)

    overall_stats = results['overall']
    by_source_stats = results['by_source']
    by_page_count_stats = results['by_page_count']

    # Print overall results
    print("\n" + "=" * 70)
    print("OVERALL RETRIEVAL RECALL")
    print("=" * 70)

    for stage in ['vision', 'text', 'final']:
        if stage in overall_stats:
            print_stage_stats(overall_stats, stage)
            print()

    # Print results by evidence source
    print("=" * 70)
    print("RETRIEVAL RECALL BY EVIDENCE SOURCE")
    print("=" * 70)

    # Sort by total count
    sorted_sources = sorted(
        by_source_stats.items(),
        key=lambda x: max(s['total'] for s in x[1].values() if isinstance(s, dict) and 'total' in s),
        reverse=True
    )

    for source, stats in sorted_sources:
        # Get total from any stage
        total = next((s['total'] for s in stats.values() if isinstance(s, dict) and 'total' in s), 0)
        print(f"\n{source}  (Total: {total} samples)")

        for stage in ['vision', 'text', 'final']:
            if stage in stats:
                print_stage_stats(stats, stage, indent="  ")

    # Print results by page count
    print("\n" + "=" * 70)
    print("RETRIEVAL RECALL BY EVIDENCE PAGE COUNT")
    print("=" * 70)

    page_categories = [
        ('single_page', 'Single Page'),
        ('multi_page', 'Multi-Page'),
        ('unanswerable', 'Unanswerable'),
    ]

    for category, label in page_categories:
        if category in by_page_count_stats:
            stats = by_page_count_stats[category]
            # Get total from any stage
            total = next((s['total'] for s in stats.values() if isinstance(s, dict) and 'total' in s), 0)
            print(f"\n{label}  (Total: {total} samples)")

            for stage in ['vision', 'text', 'final']:
                if stage in stats:
                    print_stage_stats(stats, stage, indent="  ")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
