from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from legislation_rag.config import settings
from legislation_rag.retrieval.embedder import OpenAIEmbedder
from legislation_rag.retrieval.vector_store import ChromaVectorStore

DEFAULT_BASELINE_COLLECTION = "bill_chunks"
DEFAULT_AUGMENTED_COLLECTION = "bill_chunks_plus_summaries"


def get_chunk_files(chunks_dir: Path) -> list[Path]:
  """
  Return all chunk JSON files.
  """
  return sorted(chunks_dir.glob("*_chunks.json"))


def get_summary_files(summaries_dir: Path) -> list[Path]:
  """
  Return all summary JSON files.
  """
  return sorted(summaries_dir.glob("*_summary.json"))


def read_json_file(file_path: Path) -> Any:
  """
  Read a JSON file from disk.
  """
  return json.loads(file_path.read_text(encoding="utf-8"))


def load_chunk_records(chunks_dir: Path) -> list[dict[str, Any]]:
  """
  Load all chunk records and normalize them for indexing.
  """
  chunk_files = get_chunk_files(chunks_dir)
  records: list[dict[str, Any]] = []

  for file_path in chunk_files:
    chunk_data = read_json_file(file_path)

    if not isinstance(chunk_data, list):
      raise ValueError(f"Expected list of chunk records in {file_path}")

    for chunk in chunk_data:
      records.append(
        {
          "id": chunk["chunk_id"],
          "text": chunk["text"],
          "bill_id": chunk["bill_id"],
          "source_file": chunk["source_file"],
          "chunk_index": chunk["chunk_index"],
          "char_start": chunk["char_start"],
          "char_end": chunk["char_end"],
          "char_count": chunk["char_count"],
          "doc_type": "chunk",
        }
      )

  return records


def load_summary_records(summaries_dir: Path) -> list[dict[str, Any]]:
  """
  Load all summary records and normalize them for indexing.
  """
  summary_files = get_summary_files(summaries_dir)
  records: list[dict[str, Any]] = []

  for file_path in summary_files:
    summary_data = read_json_file(file_path)

    if not isinstance(summary_data, dict):
      raise ValueError(f"Expected summary JSON object in {file_path}")

    bill_id = summary_data["bill_id"]

    records.append(
      {
        "id": f"{bill_id}_summary",
        "text": summary_data["summary"],
        "bill_id": bill_id,
        "source_file": summary_data["source_file"],
        "model_name": summary_data["model_name"],
        "generated_at_utc": summary_data["generated_at_utc"],
        "source_text_char_count": summary_data["source_text_char_count"],
        "doc_type": "summary",
      }
    )

  return records


def main() -> None:
  """
  CLI entry point for building the baseline and augmented Chroma indexes.
  """
  parser = argparse.ArgumentParser(
    description="Build Chroma indexes from bill chunks and generated bill summaries."
  )
  parser.add_argument(
    "--chunks-dir",
    type=Path,
    default=settings.processed_chunks_dir,
    help="Directory containing chunk JSON files.",
  )
  parser.add_argument(
    "--summaries-dir",
    type=Path,
    default=settings.processed_summaries_dir,
    help="Directory containing summary JSON files.",
  )
  parser.add_argument(
    "--persist-dir",
    type=Path,
    default=settings.vector_db_dir,
    help="Directory where Chroma will persist collections.",
  )
  parser.add_argument(
    "--baseline-collection",
    type=str,
    default=DEFAULT_BASELINE_COLLECTION,
    help="Collection name for original bill chunks only.",
  )
  parser.add_argument(
    "--augmented-collection",
    type=str,
    default=DEFAULT_AUGMENTED_COLLECTION,
    help="Collection name for bill chunks plus summaries.",
  )
  parser.add_argument(
    "--batch-size",
    type=int,
    default=100,
    help="Batch size for embedding/upserting records.",
  )
  parser.add_argument(
    "--reset",
    action="store_true",
    help="Delete and rebuild both collections from scratch.",
  )
  parser.add_argument(
    "--baseline-only",
    action="store_true",
    help="Build only the baseline bill_chunks collection; skip summaries and augmented collection.",
  )

  args = parser.parse_args()

  settings.ensure_directories()

  chunk_records = load_chunk_records(args.chunks_dir)

  if not chunk_records:
    raise ValueError(
      f"No chunk records found in {args.chunks_dir}. Run chunk_documents.py first."
    )

  print(f"Loaded {len(chunk_records)} chunk records.")

  embedder = OpenAIEmbedder()
  store = ChromaVectorStore(args.persist_dir)

  if args.reset:
    print(f"Resetting collection: {args.baseline_collection}")
    store.reset_collection(args.baseline_collection)

    if not args.baseline_only:
      print(f"Resetting collection: {args.augmented_collection}")
      store.reset_collection(args.augmented_collection)

  print(f"Building baseline collection: {args.baseline_collection}")
  store.upsert_records(
    collection_name=args.baseline_collection,
    records=chunk_records,
    embedder=embedder,
    batch_size=args.batch_size,
  )

  if not args.baseline_only:
    summary_records = load_summary_records(args.summaries_dir)
    print(f"Loaded {len(summary_records)} summary records.")

    augmented_records = chunk_records + summary_records

    print(f"Building augmented collection: {args.augmented_collection}")
    store.upsert_records(
      collection_name=args.augmented_collection,
      records=augmented_records,
      embedder=embedder,
      batch_size=args.batch_size,
    )

  print("Done.")

if __name__ == "__main__":
  main()