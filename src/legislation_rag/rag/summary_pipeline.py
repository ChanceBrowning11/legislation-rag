from __future__ import annotations

from dataclasses import dataclass

from legislation_rag.rag.answer_generator import AnswerGenerator, GeneratedAnswer
from legislation_rag.retrieval.retriever import BillRetriever, RetrievedDocument


DEFAULT_SUMMARY_COLLECTION = "bill_chunks_plus_summaries"


@dataclass
class SummaryRAGResult:
  """
  Structured output for a summary-augmented RAG run.
  """
  question: str
  collection_name: str
  retrieved_documents: list[RetrievedDocument]
  generated_answer: GeneratedAnswer


class SummaryRAGPipeline:
  """
  Summary-augmented RAG pipeline that retrieves from original bill chunks
  plus generated bill summaries.
  """

  def __init__(
    self,
    retriever: BillRetriever | None = None,
    answer_generator: AnswerGenerator | None = None,
    collection_name: str = DEFAULT_SUMMARY_COLLECTION,
    default_k: int = 5,
  ) -> None:
    """
    Initialize the summary-augmented pipeline.

    Args:
      retriever: Optional retriever override.
      answer_generator: Optional answer generator override.
      collection_name: Chroma collection to query.
      default_k: Default number of retrieved results to use.
    """
    if default_k < 1:
      raise ValueError("default_k must be at least 1")

    self.retriever = retriever or BillRetriever()
    self.answer_generator = answer_generator or AnswerGenerator()
    self.collection_name = collection_name
    self.default_k = default_k

  def retrieve_context(
    self,
    question: str,
    k: int | None = None,
    where: dict | None = None,
  ) -> list[RetrievedDocument]:
    """
    Retrieve the top matching records for a question from the
    summary-augmented collection.

    Args:
      question: User question.
      k: Optional override for number of retrieved results.
      where: Optional metadata filter.

    Returns:
      Retrieved context documents.
    """
    retrieval_k = k or self.default_k

    return self.retriever.retrieve(
      query=question,
      collection_name=self.collection_name,
      k=retrieval_k,
      where=where,
    )

  def answer_question(
    self,
    question: str,
    k: int | None = None,
    where: dict | None = None,
  ) -> SummaryRAGResult:
    """
    Run the full summary-augmented RAG pipeline:
    1. retrieve context from chunks + summaries
    2. generate a grounded answer from that context

    Args:
      question: User question.
      k: Optional override for number of retrieved results.
      where: Optional metadata filter.

    Returns:
      A SummaryRAGResult containing the answer and retrieved context.
    """
    if not question.strip():
      raise ValueError("Question cannot be empty.")

    retrieved_documents = self.retrieve_context(
      question=question,
      k=k,
      where=where,
    )

    if not retrieved_documents:
      raise ValueError("No retrieved documents were found for the given question.")

    generated_answer = self.answer_generator.generate_answer(
      question=question,
      retrieved_documents=retrieved_documents,
      collection_name=self.collection_name,
    )

    return SummaryRAGResult(
      question=question,
      collection_name=self.collection_name,
      retrieved_documents=retrieved_documents,
      generated_answer=generated_answer,
    )