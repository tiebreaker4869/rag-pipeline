#!/usr/bin/env python3
"""
Analyze evaluation results by evidence source and page count.

Shows accuracy breakdown by:
- Evidence source type (Chart, Table, Pure-text, Figure, etc.)
- Evidence page count (single page, multi-page, unanswerable)
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


def parse_evidence_pages(evidence_pages) -> List[int]:
    """Parse evidence_pages field to list of integers."""
    if isinstance(evidence_pages, list):
        return evidence_pages
    if isinstance(evidence_pages, str):
        try:
            return ast.literal_eval(evidence_pages)
        except:
            return []
    return []


def parse_evidence_sources(evidence_sources) -> List[str]:
    """Parse evidence_sources field to list of strings."""
    if isinstance(evidence_sources, list):
        return evidence_sources
    if isinstance(evidence_sources, str):
        try:
            return ast.literal_eval(evidence_sources)
        except:
            return []
    return []


def analyze_by_evidence_source(
    eval_results: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Analyze accuracy by evidence source type."""
    # Create mapping from (doc_id, question) to sample
    gt_map = {(s['doc_id'], s['question']): s for s in ground_truth}

    # Group by evidence source
    source_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    for result in eval_results:
        doc_id = result.get('doc_id')
        question = result.get('question')
        score = result.get('score', 0)

        # Get ground truth info
        gt_sample = gt_map.get((doc_id, question))
        if not gt_sample:
            continue

        sources = parse_evidence_sources(gt_sample.get('evidence_sources', []))

        # Count for each source type
        for source in sources:
            source_stats[source]['total'] += 1
            source_stats[source]['correct'] += score

    # Calculate accuracy
    result = {}
    for source, stats in source_stats.items():
        accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0.0
        result[source] = {
            'total': stats['total'],
            'correct': stats['correct'],
            'accuracy': accuracy,
        }

    return result


def analyze_by_page_count(
    eval_results: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Analyze accuracy by evidence page count."""
    # Create mapping from (doc_id, question) to sample
    gt_map = {(s['doc_id'], s['question']): s for s in ground_truth}

    # Group by page count category
    page_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    for result in eval_results:
        doc_id = result.get('doc_id')
        question = result.get('question')
        score = result.get('score', 0)

        # Get ground truth info
        gt_sample = gt_map.get((doc_id, question))
        if not gt_sample:
            continue

        pages = parse_evidence_pages(gt_sample.get('evidence_pages', []))
        answer = gt_sample.get('answer', '')

        # Categorize
        if answer == 'Not answerable':
            category = 'unanswerable'
        elif len(pages) == 0:
            category = 'no_pages'
        elif len(pages) == 1:
            category = 'single_page'
        else:
            category = 'multi_page'

        page_stats[category]['total'] += 1
        page_stats[category]['correct'] += score

    # Calculate accuracy
    result = {}
    for category, stats in page_stats.items():
        accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0.0
        result[category] = {
            'total': stats['total'],
            'correct': stats['correct'],
            'accuracy': accuracy,
        }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Analyze evaluation results by evidence source and page count"
    )
    parser.add_argument(
        '--eval_results',
        type=str,
        required=True,
        help='Path to evaluation results JSON (output from llm_judge.py)',
    )
    parser.add_argument(
        '--ground_truth',
        type=str,
        required=True,
        help='Path to ground truth samples JSON',
    )

    args = parser.parse_args()

    # Load data
    print("Loading evaluation results...")
    eval_results = load_json(args.eval_results)

    # Validate evaluation results format
    if eval_results and 'score' not in eval_results[0]:
        print("\n" + "=" * 70)
        print("ERROR: Invalid evaluation results format!")
        print("=" * 70)
        print("The file does not contain 'score' field.")
        print("It looks like you provided the inference results (from run_inference.py)")
        print("instead of evaluation results (from llm_judge.py).")
        print("\nPlease run llm_judge.py first:")
        print(f"  python scripts/llm_judge.py \\")
        print(f"    --predictions {args.eval_results} \\")
        print(f"    --output output/judge_results.json")
        print("\nThen use the judge results:")
        print(f"  python scripts/analyze_results.py \\")
        print(f"    --eval_results output/judge_results.json \\")
        print(f"    --ground_truth {args.ground_truth}")
        print("=" * 70)
        return

    print("Loading ground truth...")
    ground_truth = load_json(args.ground_truth)

    # Calculate overall accuracy
    scores = [r['score'] for r in eval_results if 'score' in r]
    overall_accuracy = (sum(scores) / len(scores) * 100) if scores else 0.0

    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print(f"Total samples:  {len(scores)}")
    print(f"Correct:        {sum(scores)}")
    print(f"Accuracy:       {overall_accuracy:.2f}%")

    # Analyze by evidence source
    print("\n" + "=" * 70)
    print("ACCURACY BY EVIDENCE SOURCE")
    print("=" * 70)
    source_results = analyze_by_evidence_source(eval_results, ground_truth)

    # Sort by total count (descending)
    sorted_sources = sorted(source_results.items(), key=lambda x: x[1]['total'], reverse=True)

    for source, stats in sorted_sources:
        print(f"{source:20s}  Total: {stats['total']:4d}  "
              f"Correct: {stats['correct']:4d}  "
              f"Accuracy: {stats['accuracy']:6.2f}%")

    # Analyze by page count
    print("\n" + "=" * 70)
    print("ACCURACY BY EVIDENCE PAGE COUNT")
    print("=" * 70)
    page_results = analyze_by_page_count(eval_results, ground_truth)

    # Define order
    category_order = ['single_page', 'multi_page', 'unanswerable', 'no_pages']
    category_labels = {
        'single_page': 'Single Page',
        'multi_page': 'Multi-Page',
        'unanswerable': 'Unanswerable',
        'no_pages': 'No Pages',
    }

    for category in category_order:
        if category in page_results:
            stats = page_results[category]
            label = category_labels.get(category, category)
            print(f"{label:20s}  Total: {stats['total']:4d}  "
                  f"Correct: {stats['correct']:4d}  "
                  f"Accuracy: {stats['accuracy']:6.2f}%")

    print("=" * 70)


if __name__ == '__main__':
    main()
