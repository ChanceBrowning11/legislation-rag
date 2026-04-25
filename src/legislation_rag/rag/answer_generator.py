from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from langchain_openai import ChatOpenAI

from legislation_rag.config import settings
from legislation_rag.retrieval.retriever import RetrievedDocument

ANSWER_SYSTEM_PROMPT ="""
You are a careful question-answering assistant for Minnesota legislative bills.

Your task is to answer a user's question using only the retrieved bill context provided to you.

Follow these rules:
- Use only the information in the provided context.
- Do not add outside facts, assumptions, or legal advice.
- Answer in plain English for a general reader.
- Be specific when the context supports a specific answer.
- If the context does not clearly answer the question, say so.
- Do not claim certainty when the context is incomplete or ambiguous.
- Keep the answer concise but complete.
"""

@dataclass
class GeneratedAnswer:
  """
  Structured output for a generated RAG answer.
  """
  question: str
  answer: str
  model_name: str
  generated_at_utc: str
  context_document_ids: list[str]
  context_count: int
  collection_name: str | None = None

def normalize_answer_text(text: str) -> str:
  """
  Normalize model output into readable plain text.
  """
  stripped = text.strip()

  if not stripped:
    return ""

  return "\n".join(line.rstrip() for line in stripped.splitlines()).strip()

def format_retrieved_context(documents: Iterable[RetrievedDocument]) -> str:
  """
  Format retrieved documents into a structured text block for the model.

  Each context block includes:
  - document id
  - document type
  - bill id
  - source file
  - text content
  """
  context_blocks: list[str] = []

  for index, document in enumerate(documents, start=1):
    doc_type = document.metadata.get("doc_type", "unknown")
    bill_id = document.metadata.get("bill_id", "unknown")
    source_file = document.metadata.get("source_file", "unknown")

    block = f"""[Context {index}]
document_id: {document.document_id}
doc_type: {doc_type}
bill_id: {bill_id}
source_file: {source_file}
text:
{document.text}
"""
    context_blocks.append(block)

  return "\n\n".join(context_blocks)

class AnswerGenerator:
  """
  Generate grounded answers from a user question and retrieved context.
  """

  def __init__(self, model_name: str | None = None, temperature: float = 0.0) -> None:
    """
    Initialize the answer generator.

    Args:
      model_name: Optional model override. Defaults to settings.chat_model.
      temperature: Sampling temperature. Lower is better for consistency.
    """
    if not settings.openai_api_key:
      raise ValueError(
        "OPENAI_API_KEY is not set. Add it to your .env file before generating answers."
      )

    self.model_name = model_name or settings.chat_model
    self.llm = ChatOpenAI(
      model=self.model_name,
      temperature=temperature,
      api_key=settings.openai_api_key,
    )

  def build_user_prompt(
    self,
    question: str,
    retrieved_documents: list[RetrievedDocument],
  ) -> str:
    """
    Build the user prompt for grounded answer generation.
    """
    if not question.strip():
      raise ValueError("Question cannot be empty.")

    if not retrieved_documents:
      raise ValueError("At least one retrieved document is required to generate an answer.")

    formatted_context = format_retrieved_context(retrieved_documents)

    return f"""Answer the following question using only the provided context.

Question:
{question}

Context:
{formatted_context}

Instructions:
- Answer in plain English.
- Stay grounded in the context only.
- Do not provide legal advice.
- If the context does not clearly answer the question, say that the available bill text does not clearly answer it.
"""

  def generate_answer(
    self,
    question: str,
    retrieved_documents: list[RetrievedDocument],
    collection_name: str | None = None,
  ) -> GeneratedAnswer:
    """
    Generate an answer from the supplied question and retrieved context.

    Args:
      question: User question.
      retrieved_documents: Retrieved context records.
      collection_name: Optional collection name used for retrieval.

    Returns:
      A GeneratedAnswer object containing the model answer and metadata.
    """
    user_prompt = self.build_user_prompt(
      question=question,
      retrieved_documents=retrieved_documents,
    )

    response = self.llm.invoke(
      [
        ("system", ANSWER_SYSTEM_PROMPT),
        ("human", user_prompt),
      ]
    )

    answer_text = normalize_answer_text(response.content)

    return GeneratedAnswer(
      question=question,
      answer=answer_text,
      model_name=self.model_name,
      generated_at_utc=datetime.now(timezone.utc).isoformat(),
      context_document_ids=[doc.document_id for doc in retrieved_documents],
      context_count=len(retrieved_documents),
      collection_name=collection_name,
    )