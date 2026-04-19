from __future__ import annotations
from langchain_openai import OpenAIEmbeddings
from legislation_rag.config import settings

class OpenAIEmbedder:
  """
  Thin wrapper around the OpenAI embedding model used by the project.
  """

  def __init__(self, model_name: str | None = None) -> None:
    """
    Initialize the embedding client.

    Args:
      model_name: Optional model override. Defaults to settings.embedding_model.
    """
    if not settings.openai_api_key:
      raise ValueError(
        "OPENAI_API_KEY is not set. Add it to your .env file before building indexes."
      )

    self.model_name = model_name or settings.embedding_model
    self.client = OpenAIEmbeddings(
      model=self.model_name,
      api_key=settings.openai_api_key,
    )

  def embed_texts(self, texts: list[str]) -> list[list[float]]:
    """
    Embed multiple text strings.

    Args:
      texts: List of texts to embed.

    Returns:
      List of embedding vectors.
    """
    sanitized = [text if text.strip() else " " for text in texts]
    return self.client.embed_documents(sanitized)

  def embed_query(self, query: str) -> list[float]:
    """
    Embed a single query string.

    Args:
      query: User query text.

    Returns:
      Query embedding vector.
    """
    if not query.strip():
      raise ValueError("Query text cannot be empty.")

    return self.client.embed_query(query)