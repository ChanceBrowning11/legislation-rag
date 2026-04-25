from __future__ import annotations

import argparse

from legislation_rag.retrieval.retriever import BillRetriever


def main() -> None:
  """
  Run a simple retrieval smoke test against a built Chroma collection.
  """
  parser = argparse.ArgumentParser(
    description="Run a quick retrieval smoke test against a Chroma collection."
  )
  parser.add_argument(
    "--query",
    type=str,
    required=True,
    help="Query text to search for.",
  )
  parser.add_argument(
    "--collection",
    type=str,
    default="bill_chunks_plus_summaries",
    help="Collection name to query.",
  )
  parser.add_argument(
    "--k",
    type=int,
    default=5,
    help="Number of results to return.",
  )
  parser.add_argument(
    "--preview-chars",
    type=int,
    default=300,
    help="Number of characters to show from each result.",
  )

  args = parser.parse_args()

  retriever = BillRetriever()
  results = retriever.retrieve(
    query=args.query,
    collection_name=args.collection,
    k=args.k,
  )

  if not results:
    print("No retrieval results found.")
    return

  print(f"Query: {args.query}")
  print(f"Collection: {args.collection}")
  print(f"Results returned: {len(results)}")

  for index, result in enumerate(results, start=1):
    doc_type = result.metadata.get("doc_type", "unknown")
    bill_id = result.metadata.get("bill_id", "unknown")
    source_file = result.metadata.get("source_file", "unknown")

    preview = result.text[: args.preview_chars].strip()
    if len(result.text) > args.preview_chars:
      preview += "..."

    print("\n" + "=" * 80)
    print(f"Result #{index}")
    print(f"Document ID: {result.document_id}")
    print(f"Document Type: {doc_type}")
    print(f"Bill ID: {bill_id}")
    print(f"Source File: {source_file}")
    print(f"Distance: {result.distance:.4f}")
    print("Preview:")
    print(preview)

  print("\nDone.")


if __name__ == "__main__":
  main()