from __future__ import annotations

import argparse
from pathlib import Path

from legislation_rag.config import settings
from legislation_rag.ingestion.pdf_parser import extract_text_from_pdf


def get_pdf_files(input_dir: Path) -> list[Path]:
  """
  Finds all PDF files in the given input directory.

  Args:
    input_dir: Directory to search for PDF files.

  Returns:
    A sorted list of PDF file paths.
  """
  return sorted(input_dir.glob("*.pdf"))


def write_text_output(output_path: Path, text: str) -> None:
  """
  Writes the given text to the specified output path.

  Args:
    output_path: Path to save the text file.
    text: Text content to write.
  """

  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(text, encoding="utf-8")


def process_pdf(pdf_path: Path, output_dir: Path) -> None:
  """
  Processes a single PDF file: extracts text and writes it to the output directory.

  Args:
    pdf_path: Path to the PDF file.
    output_dir: Directory to save the extracted text file.

  """
  print(f"Processing: {pdf_path.name}")
  text = extract_text_from_pdf(pdf_path)

  if not text.strip():
    print(f"  Warning: No extractable text found in {pdf_path.name}")
    return

  output_path = output_dir / f"{pdf_path.stem}.txt"
  write_text_output(output_path, text)
  print(f"  Saved: {output_path}")


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Extract raw text from PDF files into .txt files."
  )
  parser.add_argument(
    "--input-dir",
    type=Path,
    default=settings.raw_pdf_dir,
    help="Directory containing source PDF files.",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=settings.processed_text_dir,
    help="Directory to save extracted text files.",
  )
  parser.add_argument(
    "--file",
    type=str,
    default=None,
    help="Optional single PDF filename to process.",
  )

  args = parser.parse_args()

  settings.ensure_directories()

  if args.file:
    pdf_path = args.input_dir / args.file
    process_pdf(pdf_path, args.output_dir)
    return

  pdf_files = get_pdf_files(args.input_dir)

  if not pdf_files:
    print(f"No PDF files found in: {args.input_dir}")
    return

  print(f"Found {len(pdf_files)} PDF files.")
  for pdf_path in pdf_files:
    process_pdf(pdf_path, args.output_dir)

  print("Done.")


if __name__ == "__main__":
  main()