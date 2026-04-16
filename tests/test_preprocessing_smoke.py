from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Sample text used for testing the preprocessing pipeline
sample_text = """--- Page 1 ---
A bill for an act relating to housing.

This legisla-
tion modifies landlord tenant law.

It creates additional notice requirements for property owners.

--- Page 2 ---
The commissioner may adopt rules.
This section applies to residential housing disputes.
"""

def test_clean_and_chunk_pipeline_smoke(tmp_path: Path) -> None:
  """
  Smoke test the preprocessing pipeline by:

  1. Creating a fake extracted text file
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
  assert "--- Page 1 ---" not in cleaned_text
  assert "--- Page 2 ---" not in cleaned_text
  assert "legislation" in cleaned_text
  assert "legisla-\ntion" not in cleaned_text

  chunk_result = subprocess.run(
    [
      sys.executable,
      "scripts/chunk_documents.py",
      "--input-dir",
      str(cleaned_dir),
      "--output-dir",
      str(chunks_dir),
      "--chunk-size",
      "120",
      "--chunk-overlap",
      "20",
      "--min-chunk-size",
      "50",
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