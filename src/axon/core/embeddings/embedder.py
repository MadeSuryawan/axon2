"""
Batch embedding pipeline for Axon knowledge graphs.

Takes a :class:`KnowledgeGraph`, generates natural-language descriptions for
each embeddable symbol node, encodes them using *fastembed*, and returns a
list of :class:`NodeEmbedding` objects ready for storage.

Only code-level symbol nodes are embedded.  Structural nodes (Folder,
Community, Process) are deliberately skipped — they lack the semantic
richness that makes embedding worthwhile.
"""

from threading import Lock

from fastembed import TextEmbedding
from fastembed.common.types import NumpyArray

from axon.config.constants import MODEL_NAME
from axon.config.progress_bar import p_bar
from axon.core.embeddings.text import build_class_method_index, generate_text
from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import NodeLabel
from axon.core.storage.base import NodeEmbedding

_model_cache: dict[str, TextEmbedding] = {}
_model_lock = Lock()


def get_model() -> TextEmbedding:
    if cached := _model_cache.get(MODEL_NAME):
        return cached
    with _model_lock:
        if cached := _model_cache.get(MODEL_NAME):
            return cached

        model = TextEmbedding(model_name=MODEL_NAME)
        _model_cache[MODEL_NAME] = model
        return model


def _get_model_cache_clear() -> None:
    """Clear the model cache (used in tests)."""
    with _model_lock:
        _model_cache.clear()


get_model.cache_clear = _get_model_cache_clear  # type: ignore[attr-defined]

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
    if not (all_nodes := [n for n in graph.iter_nodes() if n.label in EMBEDDABLE_LABELS]):
        return []

    class_method_idx = build_class_method_index(graph)

    texts: list[str] = []
    nodes = []
    for node in all_nodes:
        if not (text := generate_text(node, graph, class_method_idx)) or not text.strip():
            continue
        texts.append(text)
        nodes.append(node)

    if not texts:
        return []

    model = get_model()
    return [
        NodeEmbedding(node_id=node.id, embedding=vector.tolist())
        for node, vector in zip(nodes, _embed_texts(texts, batch_size, model), strict=True)
    ]


def _embed_texts(texts: list[str], batch_size: int, model: TextEmbedding) -> list[NumpyArray]:
    """Embed a list of texts using fastembed's TextEmbedding model."""

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

    nodes = [graph.get_node(nid) for nid in node_ids]
    nodes = [n for n in nodes if n is not None and n.label in EMBEDDABLE_LABELS]

    if not nodes:
        return []

    class_method_idx = build_class_method_index(graph)

    texts: list[str] = []
    valid_nodes = []
    for node in nodes:
        if (text := generate_text(node, graph, class_method_idx)) and text.strip():
            texts.append(text)
            valid_nodes.append(node)

    if not texts:
        return []

    model = get_model()
    embeddings: list[NodeEmbedding] = []
    for node, vector in zip(valid_nodes, model.embed(texts, batch_size=batch_size), strict=True):
        embeddings.append(
            NodeEmbedding(
                node_id=node.id,
                embedding=vector.tolist(),
            ),
        )

    return embeddings
