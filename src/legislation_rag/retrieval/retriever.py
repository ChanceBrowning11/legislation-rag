from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from legislation_rag.config import settings
from legislation_rag.retrieval.embedder import OpenAIEmbedder
from legislation_rag.retrieval.vector_store import ChromaVectorStore

@dataclass
class RetrievedDocument:
  """
  Normalized retrieval result returned by the BillRetriever.
  """
  document_id: str
  text: str
  metadata: dict[str, Any]
  distance: float


class BillRetriever:
  """
  Retrieve top matching bill records from a Chroma collection.
  """

  def __init__(
    self,
    store: ChromaVectorStore | None = None,
    embedder: OpenAIEmbedder | None = None,
  ) -> None:
    """
    Initialize the retriever.

    Args:
        store: Optional vector store override.
        embedder: Optional embedder override.
    """
    self.store = store or ChromaVectorStore(settings.vector_db_dir)
    self.embedder = embedder or OpenAIEmbedder()

  def retrieve(
    self,
    query: str,
    collection_name: str,
    k: int = 5,
    where: dict[str, Any] | None = None,
  ) -> list[RetrievedDocument]:
    """
    Retrieve top-k records for a user query.

    Args:
        query: User query text.
        collection_name: Collection to search.
        k: Number of results to return.
        where: Optional metadata filter.

    Returns:
        List of normalized retrieval results.
    """
    query_embedding = self.embedder.embed_query(query)
    raw_results = self.store.query_collection(
      collection_name=collection_name,
      query_embedding=query_embedding,
      n_results=k,
      where=where,
    )

    ids = raw_results.get("ids", [[]])[0]
    documents = raw_results.get("documents", [[]])[0]
    metadatas = raw_results.get("metadatas", [[]])[0]
    distances = raw_results.get("distances", [[]])[0]

    results: list[RetrievedDocument] = []
    for document_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
      results.append(
        RetrievedDocument(
          document_id=document_id,
          text=text,
          metadata=metadata or {},
          distance=float(distance),
        )
      )

    return results