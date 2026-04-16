from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sample_text = """--- Page 1 ---
1.1 A bill for an act\u200b
1.2 relating to housing and landlord tenant law.
1.3 This legisla-
1.4 tion modifies notice requirements.

1 Sec. 2.
REVISOR ABC/XY 25-00001 02/04/25
State of Minnesota
This Document can be made available
in alternative formats upon request
HOUSE OF REPRESENTATIVES
H. F. No. 2261
NINETY-FOURTH SESSION
Authored by Example, Author and Writer 03/12/2025
The bill was read for the first time and referred to the Committee on Housing Finance and Policy

--- Page 2 ---
2.1 Sec. 2. Minnesota Statutes 2024, section 504B.161, subdivision 1, is amended to read:
2.2 (1) the landlord must keep the premises in reasonable repair;
2.3 (2) the landlord must maintain compliance with health and safety laws;
"""

def test_clean_and_chunk_pipeline_smoke(tmp_path: Path) -> None:
  """
  Smoke test the preprocessing pipeline by:

  1. Creating a fake extracted text file with common legislative PDF artifacts
  2. Running the cleaning script
  3. Running the chunking script
  4. Verifying the cleaned and chunked outputs exist and look correct
  """
  input_dir = tmp_path / "text"
  cleaned_dir = tmp_path / "cleaned"
  chunks_dir = tmp_path / "chunks"

  input_dir.mkdir(parents=True, exist_ok=True)

  source_file = input_dir / "HF0001.txt"
  source_file.write_text(sample_text, encoding="utf-8")

  clean_result = subprocess.run(
    [
      sys.executable,
      "scripts/clean_text.py",
      "--input-dir",
      str(input_dir),
      "--output-dir",
      str(cleaned_dir),
    ],
    capture_output=True,
    text=True,
    check=True,
  )

  assert clean_result.returncode == 0

  cleaned_file = cleaned_dir / "HF0001.txt"
  assert cleaned_file.exists()

  cleaned_text = cleaned_file.read_text(encoding="utf-8")

  # Removed extraction/page artifacts
  assert "--- Page 1 ---" not in cleaned_text
  assert "--- Page 2 ---" not in cleaned_text

  # Removed legislative line numbers
  assert "1.1 A bill for an act" not in cleaned_text
  assert "2.1 Sec. 2." not in cleaned_text
  assert cleaned_text.startswith("A bill for an act")

  # Fixed hyphenated line breaks
  assert "This legisla-\ntion" not in cleaned_text
  assert "This legislation modifies notice requirements." in cleaned_text

  # Removed invisible characters
  assert "\u200b" not in cleaned_text

  # Removed standalone footer/header section marker
  assert "\n1 Sec. 2.\n" not in cleaned_text

  # Removed repeated legislative artifact lines
  assert "REVISOR ABC/XY 25-00001 02/04/25" not in cleaned_text
  assert "State of Minnesota" not in cleaned_text
  assert "This Document can be made available" not in cleaned_text
  assert "in alternative formats upon request" not in cleaned_text
  assert "HOUSE OF REPRESENTATIVES" not in cleaned_text
  assert "H. F. No. 2261" not in cleaned_text
  assert "NINETY-FOURTH SESSION" not in cleaned_text
  assert "Authored by Example, Author and Writer 03/12/2025" not in cleaned_text
  assert "The bill was read for the first time" not in cleaned_text

  # Real section heading should remain
  assert "Sec. 2. Minnesota Statutes 2024, section 504B.161, subdivision 1, is amended to read:" in cleaned_text

  chunk_result = subprocess.run(
    [
      sys.executable,
      "scripts/chunk_documents.py",
      "--input-dir",
      str(cleaned_dir),
      "--output-dir",
      str(chunks_dir),
      "--chunk-size",
      "180",
      "--chunk-overlap",
      "30",
      "--min-chunk-size",
      "80",
    ],
    capture_output=True,
    text=True,
    check=True,
  )

  assert chunk_result.returncode == 0

  chunks_file = chunks_dir / "HF0001_chunks.json"
  assert chunks_file.exists()

  chunks = json.loads(chunks_file.read_text(encoding="utf-8"))

  assert isinstance(chunks, list)
  assert len(chunks) >= 1

  first_chunk = chunks[0]
  assert "chunk_id" in first_chunk
  assert "bill_id" in first_chunk
  assert "source_file" in first_chunk
  assert "chunk_index" in first_chunk
  assert "char_start" in first_chunk
  assert "char_end" in first_chunk
  assert "char_count" in first_chunk
  assert "text" in first_chunk

  assert first_chunk["bill_id"] == "HF0001"
  assert first_chunk["source_file"] == "HF0001.txt"
  assert first_chunk["char_count"] > 0
  assert len(first_chunk["text"].strip()) > 0

  combined_chunk_text = "\n".join(chunk["text"] for chunk in chunks)
  assert "A bill for an act" in combined_chunk_text
  assert "Sec. 2. Minnesota Statutes 2024" in combined_chunk_text