from __future__ import annotations

from dataclasses import dataclass

from legislation_rag.rag.answer_generator import AnswerGenerator, GeneratedAnswer
from legislation_rag.retrieval.retriever import BillRetriever, RetrievedDocument


DEFAULT_BASELINE_COLLECTION = "bill_chunks"


@dataclass
class BaselineRAGResult:
  """
  Structured output for a baseline RAG run.
  """
  question: str
  collection_name: str
  retrieved_documents: list[RetrievedDocument]
  generated_answer: GeneratedAnswer


class BaselineRAGPipeline:
  """
  Baseline RAG pipeline that retrieves only from original bill chunks.
  """

  def __init__(
    self,
    retriever: BillRetriever | None = None,
    answer_generator: AnswerGenerator | None = None,
    collection_name: str = DEFAULT_BASELINE_COLLECTION,
    default_k: int = 5,
  ) -> None:
    """
    Initialize the baseline pipeline.

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
    Retrieve the top matching bill chunks for a question.

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
  ) -> BaselineRAGResult:
    """
    Run the full baseline RAG pipeline:
    1. retrieve context from original bill chunks
    2. generate a grounded answer from that context

    Args:
      question: User question.
      k: Optional override for number of retrieved results.
      where: Optional metadata filter.

    Returns:
      A BaselineRAGResult containing the answer and retrieved context.
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

    return BaselineRAGResult(
      question=question,
      collection_name=self.collection_name,
      retrieved_documents=retrieved_documents,
      generated_answer=generated_answer,
    )