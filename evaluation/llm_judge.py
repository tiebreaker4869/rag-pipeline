#!/usr/bin/env python3
"""
LLM-based judge to score prediction correctness.

Takes a JSON file produced by eval_text_rag.py (with gold_answer/pred_answer),
asks an LLM to decide if the prediction correctly answers the question.
Outputs per-sample judgments and overall accuracy.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI


JUDGE_PROMPT = """You are a strict answer judge.
Given a question, a ground truth answer, and a model prediction,
decide if the prediction is correct. Be tolerant to minor paraphrasing,
but the key information must match exactly.

Respond ONLY with a JSON object of the form:
{{
  "answer": "YES" or "NO",
  "reason": "<brief reason>"
}}

Question: {question}
Ground Truth Answer: {gold}
Model Prediction: {pred}"""


def load_predictions(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def call_openai(client: OpenAI, model: str, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


def judge(client: OpenAI, model: str, question: str, gold: str, pred: str) -> Tuple[str, str]:
    prompt = JUDGE_PROMPT.format(question=question, gold=gold, pred=pred)
    raw = call_openai(client, model, prompt)

    verdict = "NO"
    reason = raw
    try:
        parsed = json.loads(raw)
        answer = str(parsed.get("answer", "")).upper()
        reason = parsed.get("reason", "") or raw
        if answer.startswith("YES"):
            verdict = "YES"
        elif answer.startswith("NO"):
            verdict = "NO"
    except json.JSONDecodeError:
        upper = raw.upper()
        if "YES" in upper and "NO" not in upper:
            verdict = "YES"
        elif "NO" in upper and "YES" not in upper:
            verdict = "NO"
    return verdict, reason


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
        default="gpt-4.1",
        help="OpenAI chat model name for judging.",
    )

    args = parser.parse_args()

    data = load_predictions(args.predictions)
    if not data:
        raise SystemExit("No predictions loaded.")

    client = OpenAI()

    judged: List[Dict[str, Any]] = []
    correct = 0

    for idx, item in enumerate(data, start=1):
        question = item.get("question", "")
        gold = str(item.get("gold_answer", ""))
        pred = str(item.get("pred_answer", ""))

        print(f"[{idx}/{len(data)}] Judging...")
        verdict, reason = judge(client, args.model, question, gold, pred)
        is_correct = 1 if verdict == "YES" else 0
        correct += is_correct

        judged.append(
            {
                "doc_id": item.get("doc_id"),
                "question": question,
                "gold_answer": gold,
                "pred_answer": pred,
                "verdict": verdict,
                "reason": reason,
                "is_correct": is_correct,
            }
        )

    accuracy = correct / len(data)
    output = {
        "total": len(data),
        "correct": correct,
        "accuracy": accuracy,
        "details": judged,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved LLM judge results to {output_path}")
    print(f"Accuracy: {accuracy:.3f} ({correct}/{len(data)})")


if __name__ == "__main__":
    main()
