"""
Batch embedding pipeline for Axon knowledge graphs.

Takes a :class:`KnowledgeGraph`, generates natural-language descriptions for
each embeddable symbol node, encodes them using *fastembed*, and returns a
list of :class:`NodeEmbedding` objects ready for storage.

Only code-level symbol nodes are embedded.  Structural nodes (Folder,
Community, Process) are deliberately skipped — they lack the semantic
richness that makes embedding worthwhile.
"""

from functools import lru_cache

from fastembed import TextEmbedding
from fastembed.common.types import NumpyArray

from axon.config.constants import MODEL_NAME
from axon.config.progress_bar import p_bar
from axon.core.embeddings.text import build_class_method_index, generate_text
from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import NodeLabel
from axon.core.storage.base import NodeEmbedding


@lru_cache(maxsize=4)
def get_model() -> TextEmbedding:
    """Return a cached TextEmbedding model instance."""
    return TextEmbedding(model_name=MODEL_NAME)


# Labels worth embedding — skip Folder, Community, Process (structural only).
EMBEDDABLE_LABELS: frozenset[NodeLabel] = frozenset(
    {
        NodeLabel.FILE,
        NodeLabel.FUNCTION,
        NodeLabel.CLASS,
        NodeLabel.METHOD,
        NodeLabel.INTERFACE,
        NodeLabel.TYPE_ALIAS,
        NodeLabel.ENUM,
    },
)


def embed_query(query: str) -> list[float] | None:
    """Embed a single query string, returning ``None`` on failure."""
    if not query or not query.strip():
        return

    try:
        model = get_model()
        return list(next(iter(model.embed([query]))))
    except (RuntimeError, ConnectionError, SystemError, OSError, TimeoutError):
        return


def embed_graph(
    graph: KnowledgeGraph,
    batch_size: int = 64,
) -> list[NodeEmbedding]:
    """
    Generate embeddings for all embeddable nodes in the graph.

    Uses fastembed's :class:`TextEmbedding` model for batch encoding.
    Each embeddable node is converted to a natural-language description
    via :func:`generate_text`, then embedded in a single batch call.

    Args:
        graph: The knowledge graph whose nodes should be embedded.
        batch_size: Number of texts to encode per batch.  Defaults to 64.

    Returns:
        A list of :class:`NodeEmbedding` instances, one per embeddable node,
        each carrying the node's ID and its embedding vector as a plain
        Python ``list[float]``.
    """
    if not (nodes := [n for n in graph.iter_nodes() if n.label in EMBEDDABLE_LABELS]):
        return []

    class_method_idx = build_class_method_index(graph)
    texts = [generate_text(node, graph, class_method_idx) for node in nodes]

    return [
        NodeEmbedding(node_id=node.id, embedding=vector.tolist())
        for node, vector in zip(nodes, _embed_texts(texts, batch_size), strict=True)
    ]


def _embed_texts(texts: list[str], batch_size: int) -> list[NumpyArray]:
    """Embed a list of texts using fastembed's TextEmbedding model."""
    model = get_model()
    vectors = []
    with p_bar(desc="Embedding", total=len(texts)) as pbar:
        for vector in model.embed(texts, batch_size=batch_size):
            vectors.append(vector)
            pbar.update()
        pbar.set_description_str("Completed")
    return vectors


def embed_nodes(
    graph: KnowledgeGraph,
    node_ids: set[str],
    batch_size: int = 64,
) -> list[NodeEmbedding]:
    """Like :func:`embed_graph`, but only for the given *node_ids*."""
    if not node_ids:
        return []

    g_nodes = [graph.get_node(nid) for nid in node_ids]
    if not (nodes := [n for n in g_nodes if n is not None and n.label in EMBEDDABLE_LABELS]):
        return []

    class_method_idx = build_class_method_index(graph)

    texts = [generate_text(n, graph, class_method_idx) for n in nodes]
    return [
        NodeEmbedding(node_id=node.id, embedding=vector.tolist())
        for node, vector in zip(nodes, _embed_texts(texts, batch_size), strict=True)
    ]
