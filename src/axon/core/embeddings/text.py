"""
Embedding text generation for graph nodes.

Converts a :class:`GraphNode` into a structured natural-language description
suitable for semantic embedding.  The description captures the node's identity,
signature, file location, and relevant graph context (callers, callees, type
references, class members, etc.).
"""

from collections.abc import Callable
from functools import cache

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import GraphNode, NodeLabel, RelType


def build_class_method_index(graph: KnowledgeGraph) -> dict[tuple[str, str], list[str]]:
    """
    Pre-build a mapping from class names to their sorted method names.

    Avoids O(classes × methods) scanning when generating text for each class.
    """
    index: dict[tuple[str, str], list[str]] = {}
    for method in graph.get_nodes_by_label(NodeLabel.METHOD):
        if method.class_name:
            key = (method.class_name, method.file_path)
            index.setdefault(key, []).append(method.name)
    for names in index.values():
        names.sort()
    return index


@cache
def _label_handlers() -> dict[NodeLabel, Callable[[GraphNode, KnowledgeGraph], str]]:
    """Build handler mapping (cached to avoid recreating dict on every call)."""
    return {
        NodeLabel.FUNCTION: _text_for_callable,
        NodeLabel.METHOD: _text_for_callable,
        NodeLabel.FILE: _text_for_file,
        NodeLabel.FOLDER: _text_for_folder,
        NodeLabel.INTERFACE: _text_for_type_definition,
        NodeLabel.TYPE_ALIAS: _text_for_type_definition,
        NodeLabel.ENUM: _text_for_type_definition,
        NodeLabel.COMMUNITY: _text_for_community,
        NodeLabel.PROCESS: _text_for_process,
    }


def generate_text(
    node: GraphNode,
    graph: KnowledgeGraph,
    class_method_index: dict[tuple[str, str], list[str]] | None = None,
) -> str:
    """
    Produce a natural-language description of *node* using graph context.

    The returned string is intended for use as input to an embedding model.
    It captures the node's identity, location, signature, and relationships
    to other nodes in *graph*.

    Args:
        node: The graph node to describe.
        graph: The knowledge graph that *node* belongs to.
        class_method_index: Optional pre-built class→method names index.
            When provided, avoids O(N) scans for class text generation.

    Returns:
        A multi-line text description of the node.
    """
    handlers = _label_handlers()
    label = node.label

    if label == NodeLabel.CLASS:
        # CLASS handler needs class_method_index, handle separately
        return _text_for_class(node, graph, class_method_index)

    if handler := handlers.get(label):
        return handler(node, graph)

    return _header(node)


def _text_for_callable(node: GraphNode, graph: KnowledgeGraph) -> str:
    """Build text for FUNCTION and METHOD nodes."""
    lines: list[str] = [_header(node)]

    if node.signature:
        lines.append(f"signature: {node.signature}")

    callee_names = _target_names(node.id, RelType.CALLS, graph)
    if callee_names:
        lines.append(f"calls: {', '.join(callee_names)}")

    caller_names = _source_names(node.id, RelType.CALLS, graph)
    if caller_names:
        lines.append(f"called by: {', '.join(caller_names)}")

    type_names = _target_names(node.id, RelType.USES_TYPE, graph)
    if type_names:
        lines.append(f"uses types: {', '.join(type_names)}")

    return "\n".join(lines)


def _text_for_class(
    node: GraphNode,
    graph: KnowledgeGraph,
    class_method_index: dict[tuple[str, str], list[str]] | None = None,
) -> str:
    """Build text for CLASS nodes."""
    lines: list[str] = [_header(node)]

    if class_method_index is not None:
        method_names = class_method_index.get((node.name, node.file_path), [])
    else:
        method_names = _class_method_names(node.name, graph)
    if method_names:
        lines.append(f"methods: {', '.join(method_names)}")

    base_names = _target_names(node.id, RelType.EXTENDS, graph)
    if base_names:
        lines.append(f"extends: {', '.join(base_names)}")

    iface_names = _target_names(node.id, RelType.IMPLEMENTS, graph)
    if iface_names:
        lines.append(f"implements: {', '.join(iface_names)}")

    return "\n".join(lines)


def _text_for_file(node: GraphNode, graph: KnowledgeGraph) -> str:
    """Build text for FILE nodes."""
    lines: list[str] = [_header(node)]

    defined_names = _target_names(node.id, RelType.DEFINES, graph)
    if defined_names:
        lines.append(f"defines: {', '.join(defined_names)}")

    import_names = _target_names(node.id, RelType.IMPORTS, graph)
    if import_names:
        lines.append(f"imports: {', '.join(import_names)}")

    return "\n".join(lines)


def _text_for_folder(node: GraphNode, graph: KnowledgeGraph) -> str:
    """Build text for FOLDER nodes."""
    lines: list[str] = [_header(node)]

    if child_names := _target_names(node.id, RelType.CONTAINS, graph):
        lines.append(f"contains: {', '.join(child_names)}")

    return "\n".join(lines)


def _text_for_type_definition(node: GraphNode, _graph: KnowledgeGraph) -> str:
    """Build text for INTERFACE, TYPE_ALIAS, and ENUM nodes."""
    lines: list[str] = [_header(node)]

    if signature := node.signature:
        lines.append(f"signature: {signature}")

    return "\n".join(lines)


def _text_for_community(node: GraphNode, graph: KnowledgeGraph) -> str:
    """Build text for COMMUNITY nodes."""
    lines: list[str] = [_header(node)]

    if member_names := _source_names(node.id, RelType.MEMBER_OF, graph):
        lines.append(f"members: {', '.join(member_names)}")

    return "\n".join(lines)


def _text_for_process(node: GraphNode, graph: KnowledgeGraph) -> str:
    """Build text for PROCESS nodes."""
    lines: list[str] = [_header(node)]

    if step_names := _source_names(node.id, RelType.STEP_IN_PROCESS, graph):
        lines.append(f"steps: {', '.join(step_names)}")

    return "\n".join(lines)


def _header(node: GraphNode) -> str:
    """Build the opening line: ``<label> <name> in <file_path>``."""
    parts: list[str] = [f"{node.label.value} {node.name}"]

    if node.label == NodeLabel.METHOD and node.class_name:
        parts.append(f"of class {node.class_name}")

    if node.file_path:
        parts.append(f"in {node.file_path}")

    return " ".join(parts)


def _target_names(
    node_id: str,
    rel_type: RelType,
    graph: KnowledgeGraph,
) -> list[str]:
    """Return sorted names of target nodes for outgoing edges of *rel_type*."""
    return sorted(
        [
            target.name
            for rel in graph.get_outgoing(node_id, rel_type=rel_type)
            if (target := graph.get_node(rel.target))
        ],
    )


def _source_names(
    node_id: str,
    rel_type: RelType,
    graph: KnowledgeGraph,
) -> list[str]:
    """Return sorted names of source nodes for incoming edges of *rel_type*."""
    return sorted(
        [
            source.name
            for rel in graph.get_incoming(node_id, rel_type=rel_type)
            if (source := graph.get_node(rel.source))
        ],
    )


def _class_method_names(class_name: str, graph: KnowledgeGraph) -> list[str]:
    """Return sorted names of METHOD nodes whose ``class_name`` matches."""
    methods = graph.get_nodes_by_label(NodeLabel.METHOD)
    return sorted(m.name for m in methods if m.class_name == class_name)
