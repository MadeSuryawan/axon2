"""Tests for the branch diff module (diff.py)."""

from dataclasses import asdict, dataclass, field
from typing import Any

from axon.core.diff import StructuralDiff, diff_graphs, format_diff
from axon.core.graph.model import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class GraphDeps:
    """
    Test helper dataclass for creating GraphNode instances with default values.

    Mirrors GraphNode fields but with sensible defaults for testing.
    """

    name: str = ""
    file_path: str = "src/app.py"
    start_line: int = 0
    end_line: int = 0
    content: str = ""
    signature: str = ""
    language: str = ""
    class_name: str = ""

    is_dead: bool = False
    is_entry_point: bool = False
    is_exported: bool = False
    properties: dict[str, Any] = field(default_factory=dict)


def _node(
    nid: str,
    deps: GraphDeps | None = None,
    label: NodeLabel = NodeLabel.FUNCTION,
) -> GraphNode:
    """Create a GraphNode with sensible defaults."""
    default_name = nid.split(":", maxsplit=1)[-1] or nid

    if deps is None:
        deps = GraphDeps(name=default_name)
    elif not deps.name:
        deps.name = default_name

    return GraphNode(id=nid, label=label, **asdict(deps))


def _rel(
    rid: str,
    rel_type: RelType = RelType.CALLS,
    source: str = "a",
    target: str = "b",
    properties: dict | None = None,
) -> GraphRelationship:
    """Create a GraphRelationship with sensible defaults."""
    return GraphRelationship(
        id=rid,
        type=rel_type,
        source=source,
        target=target,
        properties=properties or {},
    )


# ---------------------------------------------------------------------------
# Tests: diff_graphs — node detection
# ---------------------------------------------------------------------------


class TestDiffGraphsAddedNodes:
    """Nodes present in current but not base are detected as added."""

    def test_added_nodes(self) -> None:
        base = {}
        current = {"n1": _node("n1")}

        result = diff_graphs(base, current, {}, {})

        assert len(result.added_nodes) == 1
        assert result.added_nodes[0].id == "n1"
        assert result.added_nodes[0].name == "n1"
        assert result.removed_nodes == []
        assert result.modified_nodes == []


class TestDiffGraphsRemovedNodes:
    """Nodes present in base but not current are detected as removed."""

    def test_removed_nodes(self) -> None:
        base = {"n1": _node("n1")}
        current = {}

        result = diff_graphs(base, current, {}, {})

        assert len(result.removed_nodes) == 1
        assert result.removed_nodes[0].id == "n1"
        assert result.removed_nodes[0].name == "n1"
        assert result.added_nodes == []


class TestDiffGraphsModifiedContent:
    """Nodes with same ID but different content are detected as modified."""

    def test_modified_content(self) -> None:
        base = {"n1": _node("n1", GraphDeps(content="old body"))}
        current = {"n1": _node("n1", GraphDeps(content="new body"))}

        result = diff_graphs(base, current, {}, {})

        assert len(result.modified_nodes) == 1
        assert result.modified_nodes[0][0].content == "old body"
        assert result.modified_nodes[0][1].content == "new body"
        assert result.added_nodes == []
        assert result.removed_nodes == []


class TestDiffGraphsModifiedSignature:
    """Nodes with same ID but different signature are detected as modified."""

    def test_modified_signature(self) -> None:
        base = {"n1": _node("n1", GraphDeps(signature="def foo()"))}
        current = {"n1": _node("n1", GraphDeps(signature="def foo(x: int)"))}

        result = diff_graphs(base, current, {}, {})

        assert len(result.modified_nodes) == 1


class TestDiffGraphsModifiedLines:
    """Nodes with same ID but different line numbers are detected as modified."""

    def test_modified_start_line(self) -> None:
        base = {"n1": _node("n1", GraphDeps(start_line=10, end_line=20))}
        current = {"n1": _node("n1", GraphDeps(start_line=15, end_line=25))}

        result = diff_graphs(base, current, {}, {})

        assert len(result.modified_nodes) == 1


class TestDiffGraphsUnchangedNodes:
    """Identical nodes produce no diff entries."""

    def test_unchanged(self) -> None:
        n = _node("n1", GraphDeps(content="body", signature="def f()"))
        base = {"n1": n}
        current = {"n1": n}

        result = diff_graphs(base, current, {}, {})

        assert result.added_nodes == []
        assert result.removed_nodes == []
        assert result.modified_nodes == []


class TestDiffGraphsEmptyGraphs:
    """Diffing two empty graphs produces an empty diff."""

    def test_empty(self) -> None:
        result = diff_graphs({}, {}, {}, {})

        assert result == StructuralDiff()


# ---------------------------------------------------------------------------
# Tests: diff_graphs — relationship detection
# ---------------------------------------------------------------------------


class TestDiffGraphsAddedRelationships:
    """Relationships in current but not base are added."""

    def test_added_rels(self) -> None:
        base_rels: dict[str, GraphRelationship] = {}
        current_rels = {"r1": _rel("r1")}

        result = diff_graphs({}, {}, base_rels, current_rels)

        assert len(result.added_relationships) == 1
        assert result.added_relationships[0].id == "r1"


class TestDiffGraphsRemovedRelationships:
    """Relationships in base but not current are removed."""

    def test_removed_rels(self) -> None:
        base_rels = {"r1": _rel("r1")}
        current_rels: dict[str, GraphRelationship] = {}

        result = diff_graphs({}, {}, base_rels, current_rels)

        assert len(result.removed_relationships) == 1
        assert result.removed_relationships[0].id == "r1"


# ---------------------------------------------------------------------------
# Tests: diff_graphs — mixed scenarios
# ---------------------------------------------------------------------------


class TestDiffGraphsMixedChanges:
    """A realistic diff with adds, removes, and modifications."""

    def test_mixed(self) -> None:
        base_nodes = {
            "n1": _node("n1", GraphDeps(content="old")),
            "n2": _node("n2", GraphDeps(content="same")),
            "n3": _node("n3", GraphDeps(content="removed")),
        }
        current_nodes = {
            "n1": _node("n1", GraphDeps(content="new")),
            "n2": _node("n2", GraphDeps(content="same")),
            "n4": _node("n4", GraphDeps(content="added")),
        }
        base_rels = {
            "r1": _rel("r1"),
            "r2": _rel("r2"),
        }
        current_rels = {
            "r1": _rel("r1"),
            "r3": _rel("r3"),
        }

        result = diff_graphs(base_nodes, current_nodes, base_rels, current_rels)

        assert len(result.added_nodes) == 1
        assert result.added_nodes[0].id == "n4"

        assert len(result.removed_nodes) == 1
        assert result.removed_nodes[0].id == "n3"

        assert len(result.modified_nodes) == 1
        assert result.modified_nodes[0][0].id == "n1"

        assert len(result.added_relationships) == 1
        assert result.added_relationships[0].id == "r3"

        assert len(result.removed_relationships) == 1
        assert result.removed_relationships[0].id == "r2"


# ---------------------------------------------------------------------------
# Tests: format_diff
# ---------------------------------------------------------------------------


class TestFormatDiffEmpty:
    """Empty diff produces a 'no differences' message."""

    def test_empty(self) -> None:
        result = format_diff(StructuralDiff())
        assert "No structural differences" in result


class TestFormatDiffAddedNodes:
    """Added nodes appear with + prefix."""

    def test_added(self) -> None:
        diff = StructuralDiff(added_nodes=[_node("n1", GraphDeps(name="my_func"))])
        result = format_diff(diff)

        assert "+ my_func" in result
        assert "Added nodes (1)" in result
        assert "1 changes" in result


class TestFormatDiffRemovedNodes:
    """Removed nodes appear with - prefix."""

    def test_removed(self) -> None:
        diff = StructuralDiff(removed_nodes=[_node("n1", GraphDeps(name="old_func"))])
        result = format_diff(diff)

        assert "- old_func" in result
        assert "Removed nodes (1)" in result


class TestFormatDiffModifiedNodes:
    """Modified nodes appear with ~ prefix."""

    def test_modified(self) -> None:
        diff = StructuralDiff(
            modified_nodes=[
                (
                    _node("n1", GraphDeps(name="changed_func")),
                    _node("n1", GraphDeps(name="changed_func")),
                ),
            ],
        )
        result = format_diff(diff)

        assert "~ changed_func" in result
        assert "Modified nodes (1)" in result


class TestFormatDiffRelationships:
    """Relationship changes include type and source->target."""

    def test_rel_format(self) -> None:
        diff = StructuralDiff(
            added_relationships=[_rel("r1", source="func:a:f", target="func:b:g")],
            removed_relationships=[_rel("r2", source="func:c:h", target="func:d:i")],
        )
        result = format_diff(diff)

        assert "Added relationships (1)" in result
        assert "Removed relationships (1)" in result
        assert "[calls]" in result


class TestFormatDiffFullSummary:
    """The summary line shows total change count."""

    def test_summary(self) -> None:
        diff = StructuralDiff(
            added_nodes=[_node("n1")],
            removed_nodes=[_node("n2")],
            modified_nodes=[(_node("n3"), _node("n3"))],
        )
        result = format_diff(diff)

        assert "3 changes" in result
