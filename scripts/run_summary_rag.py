from __future__ import annotations

import argparse

from legislation_rag.rag.summary_pipeline import SummaryRAGPipeline

def print_retrieved_documents(retrieved_documents: list, preview_chars: int) -> None:
  """
  Print retrieved context records in a readable format.
  """
  if not retrieved_documents:
    print("\nNo retrieved documents.")
    return

  print("\nRetrieved Context:")
  for index, document in enumerate(retrieved_documents, start=1):
    doc_type = document.metadata.get("doc_type", "unknown")
    bill_id = document.metadata.get("bill_id", "unknown")
    source_file = document.metadata.get("source_file", "unknown")

    preview = document.text[:preview_chars].strip()
    if len(document.text) > preview_chars:
      preview += "..."

    print("\n" + "=" * 80)
    print(f"Result #{index}")
    print(f"Document ID: {document.document_id}")
    print(f"Document Type: {doc_type}")
    print(f"Bill ID: {bill_id}")
    print(f"Source File: {source_file}")
    print(f"Distance: {document.distance:.4f}")
    print("Preview:")
    print(preview)

def main() -> None:
  """
  Run the summary-augmented RAG pipeline for a user question.
  """
  parser = argparse.ArgumentParser(
    description="Run the summary-augmented RAG pipeline using bill chunks plus summaries."
  )
  parser.add_argument(
    "--question",
    type=str,
    required=True,
    help="Question to ask the summary-augmented RAG system.",
  )
  parser.add_argument(
    "--k",
    type=int,
    default=5,
    help="Number of retrieved context documents to use.",
  )
  parser.add_argument(
    "--show-context",
    action="store_true",
    help="Print retrieved context records along with the answer.",
  )
  parser.add_argument(
    "--preview-chars",
    type=int,
    default=300,
    help="Number of characters to show from each retrieved result preview.",
  )
  parser.add_argument(
    "--bill-id",
    type=str,
    default=None,
    help="Optional bill_id filter to restrict retrieval to a single bill.",
  )

  args = parser.parse_args()

  pipeline = SummaryRAGPipeline()

  where = None
  if args.bill_id:
    where = {"bill_id": args.bill_id}

  result = pipeline.answer_question(
    question=args.question,
    k=args.k,
    where=where,
  )

  print("\n" + "=" * 80)
  print("SUMMARY-AUGMENTED RAG ANSWER")
  print("=" * 80)
  print(f"Question: {result.question}")
  print(f"Collection: {result.collection_name}")
  print(f"Context Count: {result.generated_answer.context_count}")
  print("\nAnswer:")
  print(result.generated_answer.answer)

  if args.show_context:
    print_retrieved_documents(
      retrieved_documents=result.retrieved_documents,
      preview_chars=args.preview_chars,
    )

  print("\nDone.")

if __name__ == "__main__":
  main()