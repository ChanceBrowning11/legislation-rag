from __future__ import annotations

import argparse
from pathlib import Path

from legislation_rag.config import settings
from legislation_rag.ingestion.cleaner import clean_extracted_text


def get_text_files(input_dir: Path) -> list[Path]:
  """
  Return all .txt files in the input directory.
  """
  return sorted(input_dir.glob("*.txt"))


def read_text_file(file_path: Path) -> str:
  """
  Read a UTF-8 text file.
  """
  return file_path.read_text(encoding="utf-8")


def write_text_file(output_path: Path, text: str) -> None:
  """
  Write cleaned text to disk.
  """
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(text, encoding="utf-8")


def process_text_file(
  text_file_path: Path,
  output_dir: Path,
  remove_page_markers: bool = True,
) -> None:
  """
  Clean a single extracted text file and save the result.
  """
  print(f"Cleaning: {text_file_path.name}")

  raw_text = read_text_file(text_file_path)
  cleaned_text = clean_extracted_text(
    raw_text,
    remove_page_markers_flag=remove_page_markers,
  )

  output_path = output_dir / text_file_path.name
  write_text_file(output_path, cleaned_text)

  print(f"  Saved: {output_path}")


def main() -> None:
  """
  CLI entry point for cleaning extracted text files.
  """
  parser = argparse.ArgumentParser(
    description="Clean extracted text files and save them to the processed/cleaned directory."
  )
  parser.add_argument(
    "--input-dir",
    type=Path,
    default=settings.processed_text_dir,
    help="Directory containing extracted text files.",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=settings.processed_cleaned_dir,
    help="Directory to save cleaned text files.",
  )
  parser.add_argument(
    "--file",
    type=str,
    default=None,
    help="Optional single text filename to clean.",
  )
  parser.add_argument(
    "--keep-page-markers",
    action="store_true",
    help="Keep page markers instead of removing them.",
  )

  args = parser.parse_args()

  settings.ensure_directories()

  remove_page_markers = not args.keep_page_markers

  if args.file:
    text_file_path = args.input_dir / args.file
    if not text_file_path.exists():
      raise FileNotFoundError(f"Text file not found: {text_file_path}")

    process_text_file(
      text_file_path=text_file_path,
      output_dir=args.output_dir,
      remove_page_markers=remove_page_markers,
    )
    return

  text_files = get_text_files(args.input_dir)

  if not text_files:
    print(f"No text files found in: {args.input_dir}")
    return

  print(f"Found {len(text_files)} text files.")
  for text_file_path in text_files:
    process_text_file(
        text_file_path=text_file_path,
        output_dir=args.output_dir,
        remove_page_markers=remove_page_markers,
    )

  print("Done.")


if __name__ == "__main__":
  main()