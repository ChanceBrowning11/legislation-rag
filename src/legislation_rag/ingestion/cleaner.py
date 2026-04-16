from __future__ import annotations
import re

PAGE_MARKER_PATTERN = re.compile(r"^\s*--- Page \d+ ---\s*$", re.MULTILINE)
INLINE_WHITESPACE_PATTERN = re.compile(r"[^\S\n]+")
MULTI_BLANK_LINE_PATTERN = re.compile(r"\n{3,}")

def normalize_line_endings(text: str) -> str:
  """
  Normalize all line endings to Unix-style newlines.
  """
  return text.replace("\r\n", "\n").replace("\r", "\n")


def fix_hyphenated_line_breaks(text: str) -> str:
  """
  Join words that were split across lines during PDF extraction.

  Example:
    'informa-\\ntion' -> 'information'
  """
  return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def remove_page_markers(text: str) -> str:
  """
  Remove page markers added during PDF extraction, such as:
    --- Page 1 ---
  """
  return PAGE_MARKER_PATTERN.sub("", text)


def normalize_inline_whitespace(text: str) -> str:
  """
  Collapse repeated spaces and tabs inside lines while preserving newlines.
  """
  cleaned_lines = [
    INLINE_WHITESPACE_PATTERN.sub(" ", line).strip()
    for line in text.splitlines()
  ]
  return "\n".join(cleaned_lines)


def collapse_blank_lines(text: str, max_consecutive_blank_lines: int = 2) -> str:
  """
  Collapse excessive blank lines while preserving paragraph breaks.
  """
  if max_consecutive_blank_lines < 1:
    raise ValueError("max_consecutive_blank_lines must be at least 1")

  replacement = "\n" * max_consecutive_blank_lines
  return MULTI_BLANK_LINE_PATTERN.sub(replacement, text)


def clean_extracted_text(text: str, remove_page_markers_flag: bool = True) -> str:
  """
  Apply the full cleaning pipeline to extracted PDF text.

  Steps:
  1. Normalize line endings
  2. Remove page markers (optional)
  3. Fix hyphenated line breaks
  4. Normalize inline whitespace
  5. Collapse excessive blank lines
  6. Strip leading/trailing whitespace
  """
  cleaned = normalize_line_endings(text)

  if remove_page_markers_flag:
    cleaned = remove_page_markers(cleaned)

  cleaned = fix_hyphenated_line_breaks(cleaned)
  cleaned = cleaned.replace("\x0c", "\n")  # form feed characters
  cleaned = cleaned.replace("\u00a0", " ")  # non-breaking spaces

  cleaned = normalize_inline_whitespace(cleaned)
  cleaned = collapse_blank_lines(cleaned)
  cleaned = cleaned.strip()

  return cleaned