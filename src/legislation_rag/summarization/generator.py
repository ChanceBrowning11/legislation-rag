from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from langchain_openai import ChatOpenAI
from legislation_rag.config import settings
from legislation_rag.summarization.prompts import (
  SUMMARY_SYSTEM_PROMPT,
  build_bill_summary_user_prompt,
)

@dataclass
class BillSummaryResult:
  """
  Structured output for a generated bill summary.
  """
  bill_id: str
  source_file: str
  summary: str
  model_name: str
  generated_at_utc: str
  source_text_char_count: int


def normalize_summary_text(text: str) -> str:
  """
  Normalize model output into a single readable paragraph.

  This helps enforce the project requirement that summaries should be
  exactly one paragraph, even if the model returns extra line breaks.
  """
  stripped = text.strip()

  if not stripped:
    return ""

  normalized = " ".join(line.strip() for line in stripped.splitlines() if line.strip())
  return normalized


class BillSummaryGenerator:
  """
  Generate one-paragraph plain-English summaries for cleaned bill text.
  """

  def __init__(self, model_name: str | None = None, temperature: float = 0.0) -> None:
    """
    Initialize the summary generator.

    Args:
      model_name: Optional model override. Defaults to settings.chat_model.
      temperature: Sampling temperature. Lower is better for consistency.
    """
    if not settings.openai_api_key:
      raise ValueError(
        "OPENAI_API_KEY is not set. Add it to your .env file before generating summaries."
      )

    self.model_name = model_name or settings.chat_model
    self.llm = ChatOpenAI(
      model=self.model_name,
      temperature=temperature,
      api_key=settings.openai_api_key,
    )

  def generate_summary(
    self,
    bill_text: str,
    bill_id: str,
    source_file: str,
  ) -> BillSummaryResult:
    """
    Generate a one-paragraph summary for a single cleaned bill.

    Args:
      bill_text: Cleaned bill text.
      bill_id: Bill identifier, usually derived from filename.
      source_file: Source filename for traceability.

    Returns:
      A BillSummaryResult containing the generated summary and metadata.
    """
    if not bill_text.strip():
      raise ValueError(f"Cannot summarize empty bill text for {bill_id}")

    user_prompt = build_bill_summary_user_prompt(
      bill_text=bill_text,
      bill_id=bill_id,
    )

    response = self.llm.invoke(
      [
        ("system", SUMMARY_SYSTEM_PROMPT),
        ("human", user_prompt),
      ]
    )

    summary_text = normalize_summary_text(response.content)

    return BillSummaryResult(
      bill_id=bill_id,
      source_file=source_file,
      summary=summary_text,
      model_name=self.model_name,
      generated_at_utc=datetime.now(timezone.utc).isoformat(),
      source_text_char_count=len(bill_text),
    )