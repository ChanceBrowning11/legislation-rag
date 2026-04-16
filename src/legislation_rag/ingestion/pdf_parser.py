from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: Path) -> str:
  """
  Extract text from a PDF file using pypdf.

  Args:
    pdf_path: Path to the PDF file.

  Returns:
    Combined text from all pages in the PDF.
  """
  if not pdf_path.exists():
    raise FileNotFoundError(f"PDF not found: {pdf_path}")

  reader = PdfReader(str(pdf_path))
  page_texts: list[str] = []

  for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""
    if text.strip():
      page_texts.append(f"\n--- Page {page_number} ---\n{text}")

  return "\n".join(page_texts).strip()