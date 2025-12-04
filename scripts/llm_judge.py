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

# Evaluation prompt
JUDGE_PROMPT = """Question: {question}
Predicted Answer: {pred}
Ground Truth Answer: {gold}

Please evaluate if the predicted answer is correct compared to the ground truth.
Score the answer on:
Binary correctness (0 or 1): 1 if the answer is correct, 0 if it is incorrect

Return only a string with these scores in a dictionary and can be parsed by json.loads, e.g. {{"binary_correctness": 1}}"""


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
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> int:
    """
    Use LLM to evaluate a predicted answer against the ground truth.
    Returns score (0 or 1).
    """
    try:
        prompt = JUDGE_PROMPT.format(
            question=question,
            pred=pred,
            gold=gold,
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
        except json.JSONDecodeError:
            # Fallback: try to parse from text
            score = 0
            if "binary_correctness" in evaluation_text:
                if '"binary_correctness": 1' in evaluation_text or '"binary_correctness":1' in evaluation_text:
                    score = 1

        return score

    except Exception as e:
        logger.error(f"Error evaluating response: {e}")
        return 0


def process_item(
    item: Dict[str, Any],
    model: str,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    """Process a single evaluation item."""
    try:
        question = item.get("question", "")
        gold = str(item.get("gold_answer", ""))
        pred = str(item.get("pred_answer", ""))

        # Skip if no ground truth
        if not gold:
            return {
                "doc_id": item.get("doc_id"),
                "question": question,
                "score": 0,
                "predicted_answer": pred,
                "ground_truth": gold,
                "error": "No ground truth",
            }

        # Evaluate the response
        score = evaluate_response(
            model=model,
            question=question,
            pred=pred,
            gold=gold,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return {
            "doc_id": item.get("doc_id"),
            "question": question,
            "score": score,
            "predicted_answer": pred,
            "ground_truth": gold,
            "error": None,
        }

    except Exception as e:
        logger.error(f"Error processing item: {str(e)}")
        return {
            "doc_id": item.get("doc_id"),
            "question": item.get("question", ""),
            "score": 0,
            "predicted_answer": item.get("pred_answer", ""),
            "ground_truth": item.get("gold_answer", ""),
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
        default=0.0,
        help="Temperature for LLM responses (lower for more consistent evaluations)",
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
    scores = [item["score"] for item in judged if "score" in item]
    correct = sum(scores)
    total = len(scores)
    accuracy = correct / total if total > 0 else 0.0

    # Prepare output
    output = judged

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
    print(f"Accuracy:         {accuracy*100:.2f}%")
    print(f"{'='*60}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
