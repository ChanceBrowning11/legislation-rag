from __future__ import annotations

SUMMARY_SYSTEM_PROMPT = """You are a careful legislative summarization assistant.

Your task is to summarize a Minnesota legislative bill in plain English.
Summarize the bill for an ordinary Minnesota renter or landlord who has no legal training.

Follow these rules:
- Use only the information in the provided bill text.
- Write in plain English for a general reader with no legal background.
- Avoid legal jargon when a plain-English alternative exists.
- If a legal term is necessary, explain it in simpler words.
- Focus only on the most important changes in the bill.
- Do not add facts, assumptions, outside context, or legal advice.
- Write exactly one paragraph of 4 to 5 sentences.
- Keep the summary between 120 and 160 words.
- Do not use bullet points or headings.
- Do not end with a generic concluding sentence like "Overall, the bill..."
- Prefer simple words like "rules," "changes," "requirements," "protections," and "start date" over more technical legal terms.
"""

def build_bill_summary_user_prompt(bill_text: str, bill_id: str | None = None) -> str:
  """
  Build the user prompt for summarizing one legislative bill.

  Args:
    bill_text: Cleaned bill text.
    bill_id: Optional bill identifier for traceability.

  Returns:
    A formatted prompt containing the bill text and summary instructions.
  """
  bill_label = bill_id or "unknown_bill"

  return f"""Summarize the following Minnesota legislative bill.

Bill ID: {bill_label}

Requirements:
- Plain English
- Grounded only in the bill text
- Focused on what the bill changes
- No legal advice
- Exactly one paragraph
- Enough detail to capture the main changes

Bill text:
\"\"\"
{bill_text}
\"\"\"
"""