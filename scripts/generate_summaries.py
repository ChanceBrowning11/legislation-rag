from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from legislation_rag.config import settings
from legislation_rag.summarization.generator import BillSummaryGenerator


def get_cleaned_text_files(input_dir: Path) -> list[Path]:
  """
  Return all cleaned .txt files in the input directory.
  """
  return sorted(input_dir.glob("*.txt"))


def read_text_file(file_path: Path) -> str:
  """
  Read a UTF-8 text file.
  """
  return file_path.read_text(encoding="utf-8")


def write_summary_file(output_path: Path, summary_record: dict) -> None:
  """
  Save a summary record as formatted JSON.
  """
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(json.dumps(summary_record, indent=2), encoding="utf-8")


def process_text_file(
  text_file_path: Path,
  output_dir: Path,
  generator: BillSummaryGenerator,
  overwrite: bool = False,
) -> None:
  """
  Generate and save a summary for a single cleaned bill text file.
  """
  bill_id = text_file_path.stem
  output_path = output_dir / f"{bill_id}_summary.json"

  if output_path.exists() and not overwrite:
    print(f"Skipping existing summary: {output_path.name}")
    return

  print(f"Summarizing: {text_file_path.name}")

  bill_text = read_text_file(text_file_path)
  result = generator.generate_summary(
    bill_text=bill_text,
    bill_id=bill_id,
    source_file=text_file_path.name,
  )

  write_summary_file(output_path, asdict(result))

  print(f"  Saved: {output_path}")
  print(f"  Summary length: {len(result.summary)} characters")


def main() -> None:
  """
  CLI entry point for generating bill summaries from cleaned text files.
  """
  parser = argparse.ArgumentParser(
    description="Generate one-paragraph summaries for cleaned Minnesota bill text files."
  )
  parser.add_argument(
    "--input-dir",
    type=Path,
    default=settings.processed_cleaned_dir,
    help="Directory containing cleaned bill text files.",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=settings.processed_summaries_dir,
    help="Directory to save summary JSON files.",
  )
  parser.add_argument(
    "--file",
    type=str,
    default=None,
    help="Optional single cleaned text filename to summarize.",
  )
  parser.add_argument(
    "--model",
    type=str,
    default=None,
    help="Optional model override. Defaults to CHAT_MODEL from .env.",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite existing summary files if they already exist.",
  )

  args = parser.parse_args()

  settings.ensure_directories()
  generator = BillSummaryGenerator(model_name=args.model)

  if args.file:
    text_file_path = args.input_dir / args.file
    if not text_file_path.exists():
      raise FileNotFoundError(f"Text file not found: {text_file_path}")

    process_text_file(
      text_file_path=text_file_path,
      output_dir=args.output_dir,
      generator=generator,
      overwrite=args.overwrite,
    )
    return

  text_files = get_cleaned_text_files(args.input_dir)

  if not text_files:
    print(f"No cleaned text files found in: {args.input_dir}")
    return

  print(f"Found {len(text_files)} cleaned text files.")
  for text_file_path in text_files:
    process_text_file(
      text_file_path=text_file_path,
      output_dir=args.output_dir,
      generator=generator,
      overwrite=args.overwrite,
    )

  print("Done.")


if __name__ == "__main__":
  main()