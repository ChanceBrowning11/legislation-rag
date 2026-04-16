from __future__ import annotations
import re

PAGE_MARKER_PATTERN = re.compile(r"^\s*--- Page \d+ ---\s*$", re.MULTILINE)
INLINE_WHITESPACE_PATTERN = re.compile(r"[^\S\n]+")
MULTI_BLANK_LINE_PATTERN = re.compile(r"\n{3,}")
LINE_NUMBER_PATTERN = re.compile(r"^\s*\d+\.\d+\s+", re.MULTILINE)
STANDALONE_SECTION_MARKER_PATTERN = re.compile(r"^\s*\d+\s*Sec\.\s*\d+\.\s*$", re.MULTILINE)

ARTIFACT_LINE_PATTERNS = [
  re.compile(r"^\s*REVISOR\b.*$", re.IGNORECASE),
  re.compile(r"^\s*State of Minnesota\s*$", re.IGNORECASE),
  re.compile(
    r"^\s*This Document can be made available\s*$",
    re.IGNORECASE,
  ),
  re.compile(
    r"^\s*in alternative formats upon request\s*$",
    re.IGNORECASE,
  ),
  re.compile(
    r"^\s*(HOUSE OF REPRESENTATIVES|SENATE)\s*$",
    re.IGNORECASE,
  ),
  re.compile(
    r"^\s*[HS]\.\s*F\.\s*No\.\s*\d+.*$",
    re.IGNORECASE,
  ),
  re.compile(
    r"^\s*NINETY-[A-Z-]+\s+SESSION\s*$",
    re.IGNORECASE,
  ),
  re.compile(
    r"^\s*Authored by .*$",
    re.IGNORECASE,
  ),
  re.compile(
    r"^\s*The bill was read for the first time.*$",
    re.IGNORECASE,
  ),
]

INVISIBLE_CHAR_TRANSLATION_TABLE = str.maketrans(
    "",
    "",
    "\u200b\u200c\u200d\ufeff",
)

def normalize_line_endings(text: str) -> str:
  """
  Normalize all line endings to Unix-style newlines.
  """
  return text.replace("\r\n", "\n").replace("\r", "\n")

def remove_invisible_characters(text: str) -> str:
  """
  Remove common invisible Unicode characters introduced by PDF extraction.

  Examples include:
  - zero-width space
  - zero-width non-joiner
  - zero-width joiner
  - byte order mark
  """
  text = text.translate(INVISIBLE_CHAR_TRANSLATION_TABLE)
  text = text.replace("\x0c", "\n")   # form feed
  text = text.replace("\u00a0", " ")  # non-breaking space
  return text

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

def remove_legislative_line_numbers(text: str) -> str:
  """
  Remove legislative line-number prefixes at the start of lines.

  Example:
      '1.1 A bill for an act' -> 'A bill for an act'
  """
  return LINE_NUMBER_PATTERN.sub("", text)

def remove_standalone_section_markers(text: str) -> str:
  """
  Remove standalone footer/header section markers such as:
      '1 Sec. 2.'
      '4 Sec. 6.'

  This intentionally does NOT remove real section headings like:
      'Sec. 2. Minnesota Statutes 2024, ...'
  """
  return STANDALONE_SECTION_MARKER_PATTERN.sub("", text)

def remove_artifact_lines(text: str) -> str:
  """
  Remove repeated legislative PDF artifact lines such as:
  - REVISOR headers
  - House/Senate headers
  - bill number lines
  - session lines
  - authored/referral lines
  """
  cleaned_lines: list[str] = []

  for line in text.splitlines():
      if any(pattern.match(line) for pattern in ARTIFACT_LINE_PATTERNS):
          continue
      cleaned_lines.append(line)

  return "\n".join(cleaned_lines)

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
    2. Remove invisible Unicode characters
    3. Remove page markers (optional)
    4. Fix hyphenated line breaks
    5. Remove legislative line-number prefixes
    6. Remove standalone footer/header section markers
    7. Remove repeated artifact lines
    8. Normalize inline whitespace
    9. Collapse excessive blank lines
    10. Strip leading/trailing whitespace
  """
  cleaned = normalize_line_endings(text)
  cleaned = remove_invisible_characters(cleaned)

  if remove_page_markers_flag:
      cleaned = remove_page_markers(cleaned)

  cleaned = remove_legislative_line_numbers(cleaned)
  cleaned = fix_hyphenated_line_breaks(cleaned)
  cleaned = remove_standalone_section_markers(cleaned)
  cleaned = remove_artifact_lines(cleaned)
  cleaned = normalize_inline_whitespace(cleaned)
  cleaned = collapse_blank_lines(cleaned)
  cleaned = cleaned.strip()
  return cleaned