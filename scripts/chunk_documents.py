from __future__ import annotations

import argparse
import json
from pathlib import Path


from legislation_rag.config import settings


DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "; ", " "]


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


def write_chunks_file(output_path: Path, chunks: list[dict]) -> None:
  """
  Save chunk records as formatted JSON.
  """
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")


def find_split_point(
  text: str,
  start: int,
  proposed_end: int,
  min_chunk_size: int,
  separators: list[str],
) -> int:
  """
  Try to split near the end of the proposed chunk, preferring paragraph
  and sentence boundaries over hard character cuts.
  """
  search_start = min(len(text), start + min_chunk_size)
  best_split = -1

  for separator in separators:
    index = text.rfind(separator, search_start, proposed_end)
    if index != -1:
      split_point = index + len(separator)
      if split_point > best_split:
          best_split = split_point

  return best_split if best_split != -1 else proposed_end


def chunk_text(
  text: str,
  chunk_size: int = 1500,
  chunk_overlap: int = 200,
  min_chunk_size: int = 500,
  separators: list[str] | None = None,
) -> list[dict[str, int | str]]:
  """
  Split text into overlapping chunks.

  The algorithm prefers to split on paragraph or sentence boundaries
  when possible, while still honoring the target chunk size.
  """
  if chunk_overlap >= chunk_size:
    raise ValueError("chunk_overlap must be smaller than chunk_size")

  if min_chunk_size > chunk_size:
    raise ValueError("min_chunk_size must be smaller than or equal to chunk_size")

  separators = separators or DEFAULT_SEPARATORS
  chunks: list[dict[str, int | str]] = []

  text_length = len(text)
  start = 0

  while start < text_length:
    proposed_end = min(start + chunk_size, text_length)

    if proposed_end == text_length:
      end = text_length
    else:
        end = find_split_point(
          text=text,
          start=start,
          proposed_end=proposed_end,
          min_chunk_size=min_chunk_size,
          separators=separators,
        )

    if end <= start:
        end = min(start + chunk_size, text_length)

    chunk_content = text[start:end].strip()

    if chunk_content:
        chunks.append(
          {
            "text": chunk_content,
            "char_start": start,
            "char_end": end,
            "char_count": len(chunk_content),
          }
        )

    if end >= text_length:
      break

    next_start = max(end - chunk_overlap, start + 1)

    while next_start < text_length and text[next_start].isspace():
      next_start += 1

    start = next_start

  return chunks


def build_chunk_records(
  source_file: Path,
  text: str,
  chunk_size: int,
  chunk_overlap: int,
  min_chunk_size: int,
) -> list[dict]:
  """
  Create structured chunk records for one document.
  """
  bill_id = source_file.stem
  raw_chunks = chunk_text(
    text=text,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    min_chunk_size=min_chunk_size,
  )

  chunk_records: list[dict] = []
  for index, chunk in enumerate(raw_chunks):
    chunk_records.append(
      {
        "chunk_id": f"{bill_id}_chunk_{index:03d}",
        "bill_id": bill_id,
        "source_file": source_file.name,
        "chunk_index": index,
        "char_start": chunk["char_start"],
        "char_end": chunk["char_end"],
        "char_count": chunk["char_count"],
        "text": chunk["text"],
      }
    )

  return chunk_records


def process_text_file(
  text_file_path: Path,
  output_dir: Path,
  chunk_size: int,
  chunk_overlap: int,
  min_chunk_size: int,
) -> None:
  """
  Chunk a single cleaned text file and save its chunk records as JSON.
  """
  print(f"Chunking: {text_file_path.name}")

  text = read_text_file(text_file_path)
  chunk_records = build_chunk_records(
    source_file=text_file_path,
    text=text,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    min_chunk_size=min_chunk_size,
  )

  output_path = output_dir / f"{text_file_path.stem}_chunks.json"
  write_chunks_file(output_path, chunk_records)

  print(f"  Saved: {output_path}")
  print(f"  Chunks created: {len(chunk_records)}")


def main() -> None:
  """
  CLI entry point for chunking cleaned text files.
  """
  parser = argparse.ArgumentParser(
    description="Chunk cleaned text files into retrieval-ready JSON records."
  )
  parser.add_argument(
    "--input-dir",
    type=Path,
    default=settings.processed_cleaned_dir,
    help="Directory containing cleaned text files.",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=settings.processed_chunks_dir,
    help="Directory to save chunk JSON files.",
  )
  parser.add_argument(
    "--file",
    type=str,
    default=None,
    help="Optional single text filename to chunk.",
  )
  parser.add_argument(
    "--chunk-size",
    type=int,
    default=1500,
    help="Target number of characters per chunk.",
  )
  parser.add_argument(
    "--chunk-overlap",
    type=int,
    default=200,
    help="Number of overlapping characters between adjacent chunks.",
  )
  parser.add_argument(
    "--min-chunk-size",
    type=int,
    default=500,
    help="Minimum preferred chunk size before searching for a split point.",
  )

  args = parser.parse_args()

  settings.ensure_directories()

  if args.file:
    text_file_path = args.input_dir / args.file
    if not text_file_path.exists():
      raise FileNotFoundError(f"Text file not found: {text_file_path}")

    process_text_file(
      text_file_path=text_file_path,
      output_dir=args.output_dir,
      chunk_size=args.chunk_size,
      chunk_overlap=args.chunk_overlap,
      min_chunk_size=args.min_chunk_size,
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
      chunk_size=args.chunk_size,
      chunk_overlap=args.chunk_overlap,
      min_chunk_size=args.min_chunk_size,
    )

  print("Done.")


if __name__ == "__main__":
  main()