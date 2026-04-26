from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from legislation_rag.config import settings
from legislation_rag.rag.baseline_pipeline import BaselineRAGPipeline
from legislation_rag.rag.summary_pipeline import SummaryRAGPipeline

def read_json_file(file_path: Path) -> Any:
  """
  Read a JSON file from disk.
  """
  return json.loads(file_path.read_text(encoding="utf-8"))

def write_json_file(output_path: Path, data: Any) -> None:
  """
  Write JSON data to disk with indentation.
  """
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def load_questions(
  questions_file: Path,
  category: str | None = None,
  question_id: str | None = None,
  limit: int | None = None,
) -> list[dict[str, Any]]:
  """
  Load evaluation questions and optionally filter them.
  """
  questions = read_json_file(questions_file)

  if not isinstance(questions, list):
    raise ValueError(f"Expected a list of questions in {questions_file}")

  filtered = questions

  if category:
    filtered = [question for question in filtered if question.get("category") == category]

  if question_id:
    filtered = [question for question in filtered if question.get("question_id") == question_id]

  if limit is not None:
    filtered = filtered[:limit]

  return filtered

def build_where_filter(question_record: dict[str, Any]) -> dict[str, Any] | None:
  """
  Build an optional retrieval filter from the question record.

  Exact-bill control questions can use bill_id to restrict retrieval.
  """
  bill_id = question_record.get("bill_id")
  if bill_id:
    return {"bill_id": bill_id}

  return None

def serialize_pipeline_result(result: Any) -> dict[str, Any]:
  """
  Convert a pipeline result dataclass into a JSON-serializable dictionary.
  """
  return asdict(result)

def evaluate_question(
  question_record: dict[str, Any],
  baseline_pipeline: BaselineRAGPipeline,
  summary_pipeline: SummaryRAGPipeline,
  k: int,
) -> dict[str, Any]:
  """
  Run one evaluation question through both systems.
  """
  question_text = question_record["question"]
  where = build_where_filter(question_record)

  baseline_result = baseline_pipeline.answer_question(
    question=question_text,
    k=k,
    where=where,
  )

  summary_result = summary_pipeline.answer_question(
    question=question_text,
    k=k,
    where=where,
  )

  return {
    "question_id": question_record["question_id"],
    "category": question_record["category"],
    "scope": question_record["scope"],
    "question": question_text,
    "bill_id": question_record.get("bill_id"),
    "expected_points": question_record.get("expected_points", []),
    "baseline_result": serialize_pipeline_result(baseline_result),
    "summary_result": serialize_pipeline_result(summary_result),
  }


def build_output_path(output_dir: Path, run_label: str | None = None) -> Path:
  """
  Build a timestamped output filename for an evaluation run.
  """
  timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  suffix = f"_{run_label}" if run_label else ""
  return output_dir / f"evaluation_run_{timestamp}{suffix}.json"


def print_run_summary(results: list[dict[str, Any]]) -> None:
  """
  Print a compact summary of the evaluation run.
  """
  print("\n" + "=" * 80)
  print("EVALUATION RUN SUMMARY")
  print("=" * 80)
  print(f"Questions evaluated: {len(results)}")

  for record in results:
    baseline_context_count = record["baseline_result"]["generated_answer"]["context_count"]
    summary_context_count = record["summary_result"]["generated_answer"]["context_count"]

    print(
      f"- {record['question_id']} | {record['category']} | "
      f"baseline_context={baseline_context_count} | "
      f"summary_context={summary_context_count}"
    )


def main() -> None:
  """
  Run the full evaluation question set against both RAG systems.
  """
  parser = argparse.ArgumentParser(
    description="Evaluate baseline and summary-augmented RAG systems on a question set."
  )
  parser.add_argument(
    "--questions-file",
    type=Path,
    default=settings.processed_data_dir.parent / "evaluation" / "questions.json",
    help="Path to the evaluation question set JSON file.",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=settings.processed_data_dir.parent / "evaluation" / "results",
    help="Directory where evaluation results will be saved.",
  )
  parser.add_argument(
    "--category",
    type=str,
    default=None,
    help="Optional category filter (for example: corpus_wide, specific_bill_natural, exact_bill_control).",
  )
  parser.add_argument(
    "--question-id",
    type=str,
    default=None,
    help="Optional question_id filter to run only one question.",
  )
  parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Optional limit on the number of questions to run.",
  )
  parser.add_argument(
    "--k",
    type=int,
    default=5,
    help="Number of retrieved documents to use for each system.",
  )
  parser.add_argument(
    "--run-label",
    type=str,
    default=None,
    help="Optional label to append to the output filename.",
  )

  args = parser.parse_args()

  questions = load_questions(
    questions_file=args.questions_file,
    category=args.category,
    question_id=args.question_id,
    limit=args.limit,
  )

  if not questions:
    raise ValueError("No evaluation questions matched the requested filters.")

  baseline_pipeline = BaselineRAGPipeline()
  summary_pipeline = SummaryRAGPipeline()

  run_started_at = datetime.now(timezone.utc).isoformat()
  results: list[dict[str, Any]] = []

  print(f"Loaded {len(questions)} evaluation questions.")

  for index, question_record in enumerate(questions, start=1):
    print("\n" + "-" * 80)
    print(f"Running question {index}/{len(questions)}: {question_record['question_id']}")
    print(question_record["question"])

    result_record = evaluate_question(
      question_record=question_record,
      baseline_pipeline=baseline_pipeline,
      summary_pipeline=summary_pipeline,
      k=args.k,
    )
    results.append(result_record)

  run_finished_at = datetime.now(timezone.utc).isoformat()

  output_payload = {
    "run_started_at_utc": run_started_at,
    "run_finished_at_utc": run_finished_at,
    "question_count": len(results),
    "k": args.k,
    "category_filter": args.category,
    "question_id_filter": args.question_id,
    "limit": args.limit,
    "results": results,
  }

  output_path = build_output_path(
    output_dir=args.output_dir,
    run_label=args.run_label,
  )
  write_json_file(output_path, output_payload)

  print_run_summary(results)
  print(f"\nSaved evaluation results to: {output_path}")

if __name__ == "__main__":
  main()