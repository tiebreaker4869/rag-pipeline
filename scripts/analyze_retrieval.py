#!/usr/bin/env python3
"""
Analyze retrieval and reranking failures by evidence source and page count.

Classifies errors based on retrieval stages:
- vision_retrieval_fail: Vision retrieval missed all evidence pages
- vision_partial: Vision retrieval got some but not all evidence pages
- text_retrieval_fail: Text retrieval missed all evidence pages
- text_partial: Text retrieval got some but not all evidence pages
- rerank_fail: Reranking removed all evidence pages
- rerank_partial: Reranking removed some evidence pages
- generation_fail: All evidence pages were retrieved and kept, but answer is wrong
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


def classify_retrieval_failure(
    score: int,
    evidence_pages: List[int],
    vision_pages: List[int] = None,
    text_pages: List[int] = None,
    final_pages: List[int] = None,
) -> str:
    """Classify the type of retrieval/reranking failure."""
    if score == 1:
        return 'correct'

    evidence_set = set(evidence_pages) if evidence_pages else set()

    if not evidence_set:
        return 'no_evidence'

    # Check vision retrieval (multimodal pipelines only)
    if vision_pages is not None:
        vision_set = set(vision_pages) if vision_pages else set()
        vision_overlap = evidence_set & vision_set

        if not vision_overlap:
            return 'vision_retrieval_fail'
        elif len(vision_overlap) < len(evidence_set):
            return 'vision_partial'

    # Check text retrieval
    if text_pages is not None:
        text_set = set(text_pages) if text_pages else set()
        text_overlap = evidence_set & text_set

        if not text_overlap:
            return 'text_retrieval_fail'
        elif len(text_overlap) < len(evidence_set):
            return 'text_partial'

    # Check reranking effect
    if text_pages is not None and final_pages is not None:
        text_set = set(text_pages) if text_pages else set()
        final_set = set(final_pages) if final_pages else set()

        text_overlap = evidence_set & text_set
        final_overlap = evidence_set & final_set

        if len(final_overlap) < len(text_overlap):
            if not final_overlap:
                return 'rerank_fail'
            else:
                return 'rerank_partial'

    return 'generation_fail'


def analyze_by_category(
    eval_results: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Analyze retrieval failures by evidence source and page count.

    Returns:
        {
            'by_source': {source_type: {failure_type: count}},
            'by_page_count': {page_category: {failure_type: count}},
            'overall': {failure_type: count}
        }
    """
    gt_map = {(s['doc_id'], s['question']): s for s in ground_truth}

    # Initialize stats
    stats_by_source = defaultdict(lambda: defaultdict(int))
    stats_by_page_count = defaultdict(lambda: defaultdict(int))
    stats_overall = defaultdict(int)

    for result in eval_results:
        doc_id = result.get('doc_id')
        question = result.get('question')
        score = result.get('score', 0)

        gt_sample = gt_map.get((doc_id, question))
        if not gt_sample:
            continue

        # Get ground truth info
        evidence_pages = parse_list_field(gt_sample.get('evidence_pages', []))
        evidence_sources = parse_list_field(gt_sample.get('evidence_sources', []))
        answer = gt_sample.get('answer', '')

        # Get retrieval metadata
        vision_pages = result.get('vision_retrieved_pages')
        text_pages = result.get('text_retrieved_pages')
        final_pages = result.get('final_pages')

        vision_pages = parse_list_field(vision_pages) if vision_pages is not None else None
        text_pages = parse_list_field(text_pages) if text_pages is not None else None
        final_pages = parse_list_field(final_pages) if final_pages is not None else None

        # Classify failure
        failure_type = classify_retrieval_failure(
            score=score,
            evidence_pages=evidence_pages,
            vision_pages=vision_pages,
            text_pages=text_pages,
            final_pages=final_pages,
        )

        # Categorize by page count
        if answer == 'Not answerable':
            page_category = 'unanswerable'
        elif len(evidence_pages) == 0:
            page_category = 'no_pages'
        elif len(evidence_pages) == 1:
            page_category = 'single_page'
        else:
            page_category = 'multi_page'

        # Update stats
        stats_overall[failure_type] += 1
        stats_by_page_count[page_category][failure_type] += 1

        # Update stats for each evidence source
        for source in evidence_sources:
            stats_by_source[source][failure_type] += 1

    return {
        'overall': dict(stats_overall),
        'by_source': {k: dict(v) for k, v in stats_by_source.items()},
        'by_page_count': {k: dict(v) for k, v in stats_by_page_count.items()},
    }


def print_failure_breakdown(stats: Dict[str, int], total: int, indent: str = "  "):
    """Print failure type breakdown."""
    failure_types = [
        ('vision_retrieval_fail', 'Vision Retrieval Fail'),
        ('vision_partial', 'Vision Partial'),
        ('text_retrieval_fail', 'Text Retrieval Fail'),
        ('text_partial', 'Text Partial'),
        ('rerank_fail', 'Rerank Fail'),
        ('rerank_partial', 'Rerank Partial'),
        ('generation_fail', 'Generation Fail'),
        ('no_evidence', 'No Evidence'),
    ]

    total_errors = total - stats.get('correct', 0)

    if total_errors == 0:
        print(f"{indent}No errors")
        return

    for error_type, label in failure_types:
        if error_type in stats:
            count = stats[error_type]
            percentage = (count / total_errors * 100) if total_errors > 0 else 0.0
            print(f"{indent}{label:25s}  {count:3d}  ({percentage:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze retrieval and reranking failures by evidence source and page count"
    )
    parser.add_argument(
        '--eval_results',
        type=str,
        required=True,
        help='Path to evaluation results JSON (output from llm_judge.py)',
    )
    parser.add_argument(
        '--inference_results',
        type=str,
        required=True,
        help='Path to inference results JSON (output from run_inference.py, contains retrieval metadata)',
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

    # Validate format
    if eval_results and 'score' not in eval_results[0]:
        print("\n" + "=" * 70)
        print("ERROR: Invalid evaluation results format!")
        print("=" * 70)
        print("The file does not contain 'score' field.")
        print("Please run llm_judge.py first.")
        print("=" * 70)
        return

    print("Loading inference results...")
    inference_results = load_json(args.inference_results)

    print("Loading ground truth...")
    ground_truth = load_json(args.ground_truth)

    # Merge eval scores with inference metadata
    print("Merging evaluation scores with retrieval metadata...")
    inference_map = {(r['doc_id'], r['question']): r for r in inference_results}

    merged_results = []
    for eval_item in eval_results:
        doc_id = eval_item['doc_id']
        question = eval_item['question']

        inference_item = inference_map.get((doc_id, question))
        if inference_item:
            # Merge: eval score + inference metadata
            merged = {
                'doc_id': doc_id,
                'question': question,
                'score': eval_item['score'],
                'vision_retrieved_pages': inference_item.get('vision_retrieved_pages'),
                'text_retrieved_pages': inference_item.get('text_retrieved_pages'),
                'final_pages': inference_item.get('final_pages'),
            }
            merged_results.append(merged)

    print(f"Merged {len(merged_results)} samples with retrieval metadata")

    # Analyze
    print("\nAnalyzing retrieval failures...")
    results = analyze_by_category(merged_results, ground_truth)

    overall_stats = results['overall']
    by_source_stats = results['by_source']
    by_page_count_stats = results['by_page_count']

    # Calculate overall metrics
    total_samples = sum(overall_stats.values())
    correct = overall_stats.get('correct', 0)
    total_errors = total_samples - correct
    accuracy = (correct / total_samples * 100) if total_samples > 0 else 0.0

    # Print overall results
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print(f"Total samples:     {total_samples}")
    print(f"Correct:           {correct}")
    print(f"Incorrect:         {total_errors}")
    print(f"Accuracy:          {accuracy:.2f}%")
    print("=" * 70)

    if total_errors > 0:
        print("\nOverall Error Breakdown:")
        print_failure_breakdown(overall_stats, total_samples)

    # Print results by evidence source
    print("\n" + "=" * 70)
    print("FAILURE ANALYSIS BY EVIDENCE SOURCE")
    print("=" * 70)

    # Sort by total count
    sorted_sources = sorted(
        by_source_stats.items(),
        key=lambda x: sum(x[1].values()),
        reverse=True
    )

    for source, stats in sorted_sources:
        total = sum(stats.values())
        correct_count = stats.get('correct', 0)
        errors = total - correct_count
        acc = (correct_count / total * 100) if total > 0 else 0.0

        print(f"\n{source}  (Total: {total}, Accuracy: {acc:.1f}%)")
        if errors > 0:
            print_failure_breakdown(stats, total)
        else:
            print("  No errors")

    # Print results by page count
    print("\n" + "=" * 70)
    print("FAILURE ANALYSIS BY EVIDENCE PAGE COUNT")
    print("=" * 70)

    page_categories = [
        ('single_page', 'Single Page'),
        ('multi_page', 'Multi-Page'),
        ('unanswerable', 'Unanswerable'),
        ('no_pages', 'No Pages'),
    ]

    for category, label in page_categories:
        if category in by_page_count_stats:
            stats = by_page_count_stats[category]
            total = sum(stats.values())
            correct_count = stats.get('correct', 0)
            errors = total - correct_count
            acc = (correct_count / total * 100) if total > 0 else 0.0

            print(f"\n{label}  (Total: {total}, Accuracy: {acc:.1f}%)")
            if errors > 0:
                print_failure_breakdown(stats, total)
            else:
                print("  No errors")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
