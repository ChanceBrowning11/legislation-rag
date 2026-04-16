from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
  openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
  embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
  chat_model: str = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
  vector_db_dir: Path = Path(os.getenv("VECTOR_DB_DIR", "./vectorstore"))
  raw_data_dir: Path = Path(os.getenv("RAW_DATA_DIR", "./data/raw"))
  processed_data_dir: Path = Path(os.getenv("PROCESSED_DATA_DIR", "./data/processed"))
  log_level: str = os.getenv("LOG_LEVEL", "INFO")

  @property
  def raw_pdf_dir(self) -> Path:
    return self.raw_data_dir / "pdfs"

  @property
  def raw_metadata_dir(self) -> Path:
    return self.raw_data_dir / "metadata"

  @property
  def processed_text_dir(self) -> Path:
    return self.processed_data_dir / "text"

  @property
  def processed_cleaned_dir(self) -> Path:
    return self.processed_data_dir / "cleaned"

  @property
  def processed_chunks_dir(self) -> Path:
    return self.processed_data_dir / "chunks"

  @property
  def processed_summaries_dir(self) -> Path:
    return self.processed_data_dir / "summaries"

  def ensure_directories(self) -> None:
    self.vector_db_dir.mkdir(parents=True, exist_ok=True)
    self.raw_pdf_dir.mkdir(parents=True, exist_ok=True)
    self.raw_metadata_dir.mkdir(parents=True, exist_ok=True)
    self.processed_text_dir.mkdir(parents=True, exist_ok=True)
    self.processed_cleaned_dir.mkdir(parents=True, exist_ok=True)
    self.processed_chunks_dir.mkdir(parents=True, exist_ok=True)
    self.processed_summaries_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()