from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable
import chromadb
from legislation_rag.retrieval.embedder import OpenAIEmbedder

PrimitiveMetadataValue = str | int | float | bool


def iter_batches(items: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
  """
  Yield records in fixed-size batches.
  """
  if batch_size < 1:
    raise ValueError("batch_size must be at least 1")

  for index in range(0, len(items), batch_size):
    yield items[index:index + batch_size]


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, PrimitiveMetadataValue]:
  """
  Convert metadata into Chroma-compatible primitive values.
  """
  cleaned: dict[str, PrimitiveMetadataValue] = {}

  for key, value in metadata.items():
    if value is None:
      continue

    if isinstance(value, (str, int, float, bool)):
      cleaned[key] = value
    elif isinstance(value, Path):
      cleaned[key] = str(value)
    else:
      cleaned[key] = str(value)

  return cleaned


class ChromaVectorStore:
  """
  Wrapper around a persistent Chroma vector store.
  """

  def __init__(self, persist_dir: Path) -> None:
    """
    Initialize the persistent Chroma client.

    Args:
      persist_dir: Directory where Chroma data will be stored.
    """
    self.persist_dir = persist_dir
    self.persist_dir.mkdir(parents=True, exist_ok=True)
    self.client = chromadb.PersistentClient(path=str(self.persist_dir))

  def get_or_create_collection(self, collection_name: str):
    """
    Get an existing Chroma collection or create it if it does not exist.
    """
    return self.client.get_or_create_collection(
      name=collection_name,
      metadata={"hnsw:space": "cosine"},
    )

  def reset_collection(self, collection_name: str) -> None:
    """
    Delete and recreate a collection.

    Useful when you want to rebuild indexes from scratch.
    """
    try:
      self.client.delete_collection(name=collection_name)
    except Exception:
      pass

    self.get_or_create_collection(collection_name)

  def upsert_records(
    self,
    collection_name: str,
    records: list[dict[str, Any]],
    embedder: OpenAIEmbedder,
    id_field: str = "id",
    text_field: str = "text",
    batch_size: int = 100,
  ) -> None:
    """
    Embed and upsert records into a Chroma collection.

    Args:
      collection_name: Target Chroma collection name.
      records: Records containing at least id_field and text_field.
      embedder: Embedder used to create embeddings.
      id_field: Field used as the unique document id.
      text_field: Field containing text to embed.
      batch_size: Number of records to embed per batch.
    """
    if not records:
      print(f"No records provided for collection: {collection_name}")
      return

    collection = self.get_or_create_collection(collection_name)

    for batch in iter_batches(records, batch_size=batch_size):
      ids: list[str] = []
      documents: list[str] = []
      metadatas: list[dict[str, PrimitiveMetadataValue]] = []

      for record in batch:
        if id_field not in record:
          raise KeyError(f"Missing required id field '{id_field}' in record: {record}")
        if text_field not in record:
          raise KeyError(f"Missing required text field '{text_field}' in record: {record}")

        ids.append(str(record[id_field]))
        documents.append(str(record[text_field]))

        metadata = {
          key: value
          for key, value in record.items()
          if key not in {id_field, text_field}
        }
        metadatas.append(sanitize_metadata(metadata))

      embeddings = embedder.embed_texts(documents)

      collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
      )

  def query_collection(
    self,
    collection_name: str,
    query_embedding: list[float],
    n_results: int = 5,
    where: dict[str, Any] | None = None,
  ) -> dict[str, Any]:
    """
    Query a collection by vector similarity.

    Args:
      collection_name: Chroma collection name.
      query_embedding: Embedded query vector.
      n_results: Number of results to return.
      where: Optional metadata filter.

    Returns:
      Raw Chroma query response.
    """
    collection = self.get_or_create_collection(collection_name)

    return collection.query(
      query_embeddings=[query_embedding],
      n_results=n_results,
      where=where,
      include=["documents", "metadatas", "distances"],
    )