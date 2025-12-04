#!/usr/bin/env python3
"""
LLM-based judge to score prediction correctness.

Takes a JSON file produced by eval_text_rag.py (with gold_answer/pred_answer),
asks an LLM to decide if the prediction correctly answers the question.
Outputs per-sample judgments and overall accuracy.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List
from functools import partial

import joblib
from tqdm import tqdm

# Enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced evaluation prompt with error type classification
JUDGE_PROMPT = """You are evaluating a RAG system's answer.

Question: {question}
Ground Truth Answer: {gold}
Predicted Answer: {pred}

Evidence Information:
- Evidence pages (where the answer should be): {evidence_pages}
- Evidence sources (e.g., Chart, Table, Pure-text, Figure): {evidence_sources}

Retrieval Results:
- Vision retrieved pages: {vision_retrieved_pages}
- Text retrieved pages (before reranking): {text_retrieved_pages}
- Final pages sent to LLM (after reranking): {final_pages}

Task:
1. Determine if the predicted answer is correct (binary_correctness: 0 or 1)
2. Provide a brief explanation
3. If incorrect, classify the error type

Error Types:
- vision_retrieval_fail: Vision retrieval missed all evidence pages
- vision_partial_evidence: Vision retrieval got some but not all evidence pages
- text_retrieval_fail: Text retrieval/chunking failed to get key evidence
- rerank_fail: Reranking removed relevant evidence pages
- hallucination: LLM generated false information despite having correct context
- ambiguous_interpretation: LLM misunderstood the question intent
- layout_error: OCR/parsing issues for complex layouts (tables, charts, figures, formulas, multi-column text, etc.)
- Misc_XXX: Other errors (replace XXX, e.g., Misc_Math, Misc_NotAnswerable, Misc_MultiHop)

Guidelines:
- Check if evidence pages were retrieved at each stage
- Even if pages were retrieved, key chunks might be missing (text_retrieval_fail)
- Consider evidence_sources:
  * Chart/Table/Figure → likely layout_error if parsing failed
  * Pure-text → less likely layout_error
- If all retrieval succeeded but answer is wrong, likely hallucination or ambiguous_interpretation
- layout_error includes: table structure corruption, chart/figure OCR failure, formula misrecognition, multi-column text disorder

Return ONLY this JSON:
{{
  "binary_correctness": 0 or 1,
  "explanation": "brief explanation of correctness and error cause",
  "error_type": "error_type_name" or null
}}

Example (retrieval failure):
{{"binary_correctness": 0, "explanation": "Evidence pages [5, 10] were needed but vision retrieval only got [1, 2, 3]. Complete vision retrieval failure.", "error_type": "vision_retrieval_fail"}}

Example (layout error):
{{"binary_correctness": 0, "explanation": "All evidence pages were retrieved, but the answer is from a table (evidence_sources: Table). The OCR likely corrupted the table structure, causing incorrect extraction.", "error_type": "layout_error"}}

Example (generation error):
{{"binary_correctness": 0, "explanation": "All evidence pages were retrieved correctly, but the model hallucinated 30% instead of the correct 25% stated in the document.", "error_type": "hallucination"}}

Example (correct):
{{"binary_correctness": 1, "explanation": "The predicted answer correctly identifies the answer from the retrieved evidence.", "error_type": null}}
"""


def load_predictions(path: str) -> List[Dict[str, Any]]:
    """Load predictions from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def call_llm(
    model: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 1.0,
) -> str:
    """
    Universal LLM calling function supporting OpenAI and Gemini models.

    Args:
        model: Model name (e.g., 'gpt-4o', 'gemini-2.0-flash-exp')
        prompt: Prompt text
        max_tokens: Maximum tokens for response
        temperature: Sampling temperature

    Returns:
        Generated text response
    """
    # Detect provider from model name
    if model.startswith('gpt-') or model.startswith('o1-'):
        # OpenAI models
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    elif model.startswith('gemini-'):
        # Gemini models
        import google.generativeai as genai
        import os

        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        gemini_model = genai.GenerativeModel(model)

        generation_config = genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        response = gemini_model.generate_content(
            prompt,
            generation_config=generation_config,
        )
        return response.text.strip()

    else:
        raise ValueError(f"Unsupported model: {model}. Use models starting with 'gpt-' or 'gemini-'")


def evaluate_response(
    model: str,
    question: str,
    pred: str,
    gold: str,
    evidence_pages: str = "[]",
    evidence_sources: str = "[]",
    vision_retrieved_pages: str = "N/A",
    text_retrieved_pages: str = "N/A",
    final_pages: str = "N/A",
    max_tokens: int = 512,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """
    Use LLM to evaluate a predicted answer against the ground truth with error type classification.
    Returns score (0 or 1), explanation, and error_type.
    """
    try:
        prompt = JUDGE_PROMPT.format(
            question=question,
            pred=pred,
            gold=gold,
            evidence_pages=evidence_pages,
            evidence_sources=evidence_sources,
            vision_retrieved_pages=vision_retrieved_pages,
            text_retrieved_pages=text_retrieved_pages,
            final_pages=final_pages,
        )

        evaluation_text = call_llm(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        try:
            # Parse the JSON response
            evaluation_dict = json.loads(evaluation_text)
            score = evaluation_dict.get("binary_correctness", 0)
            explanation = evaluation_dict.get("explanation", evaluation_text)
            error_type = evaluation_dict.get("error_type", None)
        except json.JSONDecodeError:
            # Fallback: try to parse from text
            score = 0
            explanation = evaluation_text
            error_type = None
            if "binary_correctness" in evaluation_text:
                if '"binary_correctness": 1' in evaluation_text or '"binary_correctness":1' in evaluation_text:
                    score = 1

        return {"score": score, "explanation": explanation, "error_type": error_type}

    except Exception as e:
        logger.error(f"Error evaluating response: {e}")
        return {"score": 0, "explanation": f"Evaluation error: {str(e)}", "error_type": None}


def process_item(
    item: Dict[str, Any],
    model: str,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    """Process a single evaluation item with error type classification."""
    try:
        question = item.get("question", "")
        gold = str(item.get("gold_answer", ""))
        pred = str(item.get("pred_answer", ""))

        # Extract evidence information
        evidence_pages = item.get("evidence_pages", "[]")
        evidence_sources = item.get("evidence_sources", "[]")

        # Extract retrieval metadata
        vision_pages = item.get("vision_retrieved_pages")
        text_pages = item.get("text_retrieved_pages")
        final_pages = item.get("final_pages")

        # Convert to string for prompt (handle None and list formats)
        vision_pages_str = str(vision_pages) if vision_pages is not None else "N/A"
        text_pages_str = str(text_pages) if text_pages is not None else "N/A"
        final_pages_str = str(final_pages) if final_pages is not None else "N/A"

        # Skip if no ground truth
        if not gold:
            return {
                "doc_id": item.get("doc_id"),
                "question": question,
                "correctness": 0,
                "explanation": "No ground truth answer available",
                "error_type": None,
                "error": "No ground truth",
            }

        # Evaluate the response
        eval_result = evaluate_response(
            model=model,
            question=question,
            pred=pred,
            gold=gold,
            evidence_pages=evidence_pages,
            evidence_sources=evidence_sources,
            vision_retrieved_pages=vision_pages_str,
            text_retrieved_pages=text_pages_str,
            final_pages=final_pages_str,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return {
            "doc_id": item.get("doc_id"),
            "question": question,
            "correctness": eval_result["score"],
            "explanation": eval_result["explanation"],
            "error_type": eval_result["error_type"],
            "error": None,
        }

    except Exception as e:
        logger.error(f"Error processing item: {str(e)}")
        return {
            "doc_id": item.get("doc_id"),
            "question": item.get("question", ""),
            "correctness": 0,
            "explanation": f"Exception: {str(e)}",
            "error_type": None,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="LLM judge accuracy over RAG predictions JSON."
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="output/text_rag_eval.json",
        help="Path to JSON predictions (from eval_text_rag.py).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/llm_judge_results.json",
        help="Path to save judgments and accuracy.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="LLM model name for judging",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum tokens for LLM responses",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for LLM responses",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=4,
        help="Number of parallel jobs (-1 for all cores, 1 for sequential)",
    )

    args = parser.parse_args()

    # Load predictions
    data = load_predictions(args.predictions)
    if not data:
        raise SystemExit("No predictions loaded.")

    print(f"Loaded {len(data)} predictions for evaluation")
    print(f"Using model: {args.model}")

    # Create partial function with fixed arguments
    process_func = partial(
        process_item,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Process items in parallel or sequentially
    if args.n_jobs == 1:
        # Sequential processing
        judged = []
        for item in tqdm(data, desc="Evaluating responses"):
            result = process_func(item)
            judged.append(result)
    else:
        # Parallel processing
        print(f"Processing {len(data)} items with {args.n_jobs} parallel jobs")
        judged = joblib.Parallel(n_jobs=args.n_jobs, backend="threading")(
            joblib.delayed(process_func)(item) for item in tqdm(data, desc="Evaluating")
        )

    # Calculate accuracy
    correctness_scores = [item["correctness"] for item in judged if "correctness" in item]
    correct = sum(correctness_scores)
    total = len(correctness_scores)
    accuracy = correct / total if total > 0 else 0.0

    # Calculate error type distribution
    error_type_distribution = {}
    for item in judged:
        if item.get("correctness") == 0:  # Only count errors
            error_type = item.get("error_type")
            if error_type:
                error_type_distribution[error_type] = error_type_distribution.get(error_type, 0) + 1

    # Prepare output
    output = {
        "metadata": {
            "judge_model": args.model,
            "total_samples": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": accuracy,
            "error_type_distribution": error_type_distribution,
        },
        "results": judged,
    }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Evaluation Results (Model: {args.model})")
    print(f"{'='*60}")
    print(f"Total samples:    {total}")
    print(f"Correct:          {correct}")
    print(f"Incorrect:        {total - correct}")
    print(f"Accuracy:         {accuracy*100:.2f}%")
    print(f"{'='*60}")

    if error_type_distribution:
        print(f"Error Type Distribution:")
        print(f"{'='*60}")
        # Sort by count descending
        sorted_errors = sorted(error_type_distribution.items(), key=lambda x: x[1], reverse=True)
        for error_type, count in sorted_errors:
            percentage = (count / (total - correct) * 100) if (total - correct) > 0 else 0
            print(f"  {error_type:30s} {count:3d} ({percentage:5.1f}%)")
        print(f"{'='*60}")

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
