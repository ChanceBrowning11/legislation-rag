from __future__ import annotations

import argparse
import textwrap
import webbrowser
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from legislation_rag.config import settings
from legislation_rag.retrieval.embedder import OpenAIEmbedder
from legislation_rag.retrieval.retriever import BillRetriever
from legislation_rag.retrieval.vector_store import ChromaVectorStore

DEFAULT_OUTPUT_PATH = Path("docs/visualizations/embedding_viz.html")


class PCAProjector:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        self.mean_ = embeddings.mean(axis=0, keepdims=True)
        centered = embeddings - self.mean_

        # SVD gives us principal directions without adding a heavy sklearn/numba stack.
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        self.components_ = vt[: self.n_components]
        return centered @ self.components_.T

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCAProjector must be fit before calling transform().")
        centered = embeddings - self.mean_
        return centered @ self.components_.T


def load_corpus(
    collection_name: str,
) -> tuple[list[str], np.ndarray, list[str], list[dict]]:
    store = ChromaVectorStore(settings.vector_db_dir)
    collection = store.get_or_create_collection(collection_name)
    result = collection.get(include=["embeddings", "documents", "metadatas"])

    ids: list[str] = result["ids"]
    embeddings = np.array(result["embeddings"], dtype=np.float32)
    documents: list[str] = result["documents"]
    metadatas: list[dict] = result["metadatas"] if result["metadatas"] else [{} for _ in ids]

    return ids, embeddings, documents, metadatas


def build_hover(doc_id: str, text: str, metadata: dict, wrap_width: int = 45) -> str:
    bill_id = metadata.get("bill_id", "—")
    doc_type = metadata.get("doc_type", "—")
    preview = textwrap.shorten(text, width=300, placeholder="...")
    wrapped_id = "<br>".join(textwrap.wrap(doc_id, width=wrap_width))
    wrapped_preview = "<br>".join(textwrap.wrap(preview, width=wrap_width))
    return (
        f"<b>{wrapped_id}</b><br>"
        f"Bill: {bill_id}<br>"
        f"Type: {doc_type}<br>"
        f"<br>{wrapped_preview}"
    )


def build_traces(
    coords: np.ndarray,
    labels: list[str],
    hover_texts: list[str],
    retrieved_indices: set[int],
    query_coord: np.ndarray | None,
    query_text: str | None,
    color_by: str,
    dims: int,
) -> list:
    palette = px.colors.qualitative.Plotly
    unique_labels = sorted(set(labels))
    label_color = {lbl: palette[i % len(palette)] for i, lbl in enumerate(unique_labels)}
    if color_by == "doc_type":
        label_color.update({
            "chunk": palette[0],
            "summary": palette[1],
        })

    traces = []

    # Corpus points grouped by label
    for label in unique_labels:
        mask = [i for i, l in enumerate(labels) if l == label and i not in retrieved_indices]
        if not mask:
            continue

        x = coords[mask, 0].tolist()
        y = coords[mask, 1].tolist()
        hover = [hover_texts[i] for i in mask]

        if dims == 3:
            traces.append(go.Scatter3d(
                x=x, y=y, z=coords[mask, 2].tolist(),
                mode="markers",
                name=label,
                marker=dict(size=3, color=label_color[label], opacity=0.55),
                text=hover,
                hovertemplate="%{text}<extra></extra>",
            ))
        else:
            traces.append(go.Scatter(
                x=x, y=y,
                mode="markers",
                name=label,
                marker=dict(size=5, color=label_color[label], opacity=0.55),
                text=hover,
                hovertemplate="%{text}<extra></extra>",
            ))

    # Retrieved chunks — gold diamonds
    if retrieved_indices:
        r = list(retrieved_indices)
        x = coords[r, 0].tolist()
        y = coords[r, 1].tolist()
        hover = [hover_texts[i] for i in r]

        if dims == 3:
            traces.append(go.Scatter3d(
                x=x, y=y, z=coords[r, 2].tolist(),
                mode="markers",
                name="Retrieved",
                marker=dict(
                    size=8, color="gold", symbol="diamond", opacity=1.0,
                    line=dict(color="black", width=1),
                ),
                text=hover,
                hovertemplate="%{text}<extra></extra>",
            ))
        else:
            traces.append(go.Scatter(
                x=x, y=y,
                mode="markers",
                name="Retrieved",
                marker=dict(
                    size=14, color="gold", symbol="diamond", opacity=1.0,
                    line=dict(color="black", width=1),
                ),
                text=hover,
                hovertemplate="%{text}<extra></extra>",
            ))

    # Query point — red X
    if query_coord is not None:
        label = query_text[:80] + ("..." if len(query_text) > 80 else "") if query_text else "Query"

        if dims == 3:
            traces.append(go.Scatter3d(
                x=[float(query_coord[0])],
                y=[float(query_coord[1])],
                z=[float(query_coord[2])],
                mode="markers",
                name="Query",
                marker=dict(
                    size=10, color="crimson", symbol="cross", opacity=1.0,
                    line=dict(color="darkred", width=2),
                ),
                hovertemplate=f"<b>Query</b><br>{label}<extra></extra>",
            ))
        else:
            traces.append(go.Scatter(
                x=[float(query_coord[0])],
                y=[float(query_coord[1])],
                mode="markers",
                name="Query",
                marker=dict(
                    size=16, color="crimson", symbol="x", opacity=1.0,
                    line=dict(color="darkred", width=2),
                ),
                hovertemplate=f"<b>Query</b><br>{label}<extra></extra>",
            ))

    return traces


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize bill chunk embeddings projected into PCA space."
    )
    parser.add_argument(
        "--question", type=str, default=None,
        help="Query to embed and project into the space. Also highlights top-k retrieved chunks.",
    )
    parser.add_argument(
        "--collection", type=str, default="bill_chunks",
        help="Chroma collection to visualize (default: bill_chunks, other option: bill_chunks_plus_summaries).",
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Number of retrieved chunks to highlight (requires --question).",
    )
    parser.add_argument(
        "--color-by", type=str, default="doc_type",
        help="Metadata field to color points by (default: doc_type).",
    )
    parser.add_argument(
        "--dims", type=int, choices=[2, 3], default=3,
        help="Projection output dimensions (default: 3).",
    )
    parser.add_argument(
        "--bill-id", type=str, default=None,
        help="Restrict retrieval to a specific bill_id (requires --question).",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT_PATH),
        help=f"Output HTML file path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    output_group.add_argument(
        "--output-filename", type=str, default=None,
        help=(
            "Output HTML file name inside the default output directory "
            f"({DEFAULT_OUTPUT_PATH.parent})."
        ),
    )
    parser.add_argument(
        "--no-open", action="store_true",
        help="Do not automatically open the generated HTML file.",
    )
    args = parser.parse_args()

    print(f"Loading corpus from '{args.collection}'...")
    ids, embeddings, documents, metadatas = load_corpus(args.collection)
    print(f"  {len(ids)} documents loaded.")

    if len(ids) == 0:
        print("Collection is empty — nothing to visualize.")
        return

    print(f"Fitting PCA ({args.dims}D)...")
    reducer = PCAProjector(n_components=args.dims)
    corpus_coords = reducer.fit_transform(embeddings)
    print("  PCA fit complete.")

    query_coord: np.ndarray | None = None
    retrieved_indices: set[int] = set()

    if args.question:
        print("Embedding query...")
        embedder = OpenAIEmbedder()
        query_vec = np.array([embedder.embed_query(args.question)], dtype=np.float32)
        query_coord = reducer.transform(query_vec)[0]

        print(f"Retrieving top {args.k} chunks...")
        retriever = BillRetriever()
        where = {"bill_id": args.bill_id} if args.bill_id else None
        retrieved = retriever.retrieve(
            query=args.question,
            collection_name=args.collection,
            k=args.k,
            where=where,
        )
        retrieved_id_set = {doc.document_id for doc in retrieved}
        retrieved_indices = {i for i, doc_id in enumerate(ids) if doc_id in retrieved_id_set}
        print(f"  Highlighted {len(retrieved_indices)} retrieved chunks.")

    labels = [str(meta.get(args.color_by, "unknown")) for meta in metadatas]
    hover_texts = [
        build_hover(doc_id, text, meta)
        for doc_id, text, meta in zip(ids, documents, metadatas)
    ]

    print("Building figure...")
    traces = build_traces(
        coords=corpus_coords,
        labels=labels,
        hover_texts=hover_texts,
        retrieved_indices=retrieved_indices,
        query_coord=query_coord,
        query_text=args.question,
        color_by=args.color_by,
        dims=args.dims,
    )

    fig = go.Figure(data=traces)

    subtitle = ""
    if args.question:
        q = args.question[:100] + ("..." if len(args.question) > 100 else "")
        subtitle = f"<br><sup>Query: {q}</sup>"

    hoverlabel = dict(
        bgcolor="white",
        bordercolor="#aaaaaa",
        font=dict(size=12, family="monospace", color="black"),
        align="left",
        namelength=0,
    )

    if args.dims == 3:
        fig.update_layout(
            title=f"Legislation RAG — Embedding Space (3D PCA){subtitle}",
            scene=dict(xaxis_title="PC-1", yaxis_title="PC-2", zaxis_title="PC-3"),
            legend_title=args.color_by,
            margin=dict(l=0, r=0, b=0, t=60),
            hoverlabel=hoverlabel,
        )
    else:
        fig.update_layout(
            title=f"Legislation RAG — Embedding Space (2D PCA){subtitle}",
            xaxis_title="PC-1",
            yaxis_title="PC-2",
            legend_title=args.color_by,
            margin=dict(l=40, r=40, b=40, t=80),
            hoverlabel=hoverlabel,
        )

    output_path = Path(args.output)
    if args.output_filename:
        output_filename = Path(args.output_filename)
        if output_filename.name != args.output_filename:
            parser.error("--output-filename must be a file name, not a path.")
        output_path = DEFAULT_OUTPUT_PATH.with_name(args.output_filename)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"\nSaved to: {output_path.resolve()}")

    if not args.no_open:
        opened = webbrowser.open(output_path.resolve().as_uri())
        if opened:
            print("Opened visualization in your default browser.")
        else:
            print("Could not auto-open the browser; open the saved HTML file manually.")


if __name__ == "__main__":
    main()
