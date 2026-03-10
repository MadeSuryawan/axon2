"""
Branch comparison for Axon.

Compares two code graphs structurally to find added, removed, and modified
nodes and relationships.  Uses git worktrees to avoid stashing or branch
switching in the user's working tree.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from subprocess import CalledProcessError, run
from tempfile import TemporaryDirectory

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import GraphNode, GraphRelationship
from axon.core.ingestion.pipeline import Pipelines

logger = getLogger(__name__)


@dataclass
class StructuralDiff:
    """Result of comparing two code graphs."""

    added_nodes: list[GraphNode] = field(default_factory=list)
    removed_nodes: list[GraphNode] = field(default_factory=list)
    modified_nodes: list[tuple[GraphNode, GraphNode]] = field(default_factory=list)
    added_relationships: list[GraphRelationship] = field(default_factory=list)
    removed_relationships: list[GraphRelationship] = field(default_factory=list)


# Fields checked to determine if a node was "modified".
_NODE_COMPARE_FIELDS = ("content", "signature", "start_line", "end_line")


def diff_graphs(
    base_nodes: dict[str, GraphNode],
    current_nodes: dict[str, GraphNode],
    base_rels: dict[str, GraphRelationship],
    current_rels: dict[str, GraphRelationship],
) -> StructuralDiff:
    """
    Diff two graph snapshots by node/relationship IDs.

    Nodes present only in *current_nodes* are added; only in *base_nodes* are
    removed.  Nodes with the same ID but different content/signature/lines are
    modified.  Relationships are compared by ID only (added/removed).

    Args:
        base_nodes: ``{node_id: GraphNode}`` from the base branch.
        current_nodes: ``{node_id: GraphNode}`` from the current branch.
        base_rels: ``{rel_id: GraphRelationship}`` from the base branch.
        current_rels: ``{rel_id: GraphRelationship}`` from the current branch.

    Returns:
        A :class:`StructuralDiff` with the comparison results.
    """
    result = StructuralDiff()

    base_ids = set(base_nodes)
    current_ids = set(current_nodes)

    for nid in current_ids - base_ids:
        result.added_nodes.append(current_nodes[nid])

    for nid in base_ids - current_ids:
        result.removed_nodes.append(base_nodes[nid])

    for nid in base_ids & current_ids:
        base_node = base_nodes[nid]
        current_node = current_nodes[nid]
        if _node_changed(base_node, current_node):
            result.modified_nodes.append((base_node, current_node))

    base_rel_ids = set(base_rels)
    current_rel_ids = set(current_rels)

    for rid in current_rel_ids - base_rel_ids:
        result.added_relationships.append(current_rels[rid])

    for rid in base_rel_ids - current_rel_ids:
        result.removed_relationships.append(base_rels[rid])

    return result


def _node_changed(base: GraphNode, current: GraphNode) -> bool:
    """Return True if the two nodes differ on any comparison field."""
    return any(getattr(base, attr) != getattr(current, attr) for attr in _NODE_COMPARE_FIELDS)


def diff_branches(
    repo_path: Path,
    branch_range: str,
) -> StructuralDiff:
    """
    Compare two branches structurally using git worktrees.

    *branch_range* should be ``"base..current"`` (e.g. ``"main..feature"``).
    If only one branch is given (no ``..``), it is treated as the base and
    the current working tree is used as the current branch.

    Steps:
        1. Parse branch range into base/current references.
        2. Create a temporary worktree for the base branch.
        3. Run the pipeline on both branches to build in-memory graphs.
        4. Diff the two graphs.
        5. Clean up the worktree.

    Args:
        repo_path: Root of the git repository.
        branch_range: Branch range string (e.g. ``"main..feature"``).

    Returns:
        A :class:`StructuralDiff` comparing the two branches.

    Raises:
        ValueError: If the branch range format is invalid.
        RuntimeError: If git operations fail.
    """

    if ".." in branch_range:
        parts = branch_range.split("..", 1)
        base_ref = parts[0].strip()
        current_ref = parts[1].strip() if parts[1].strip() else None
    else:
        base_ref = branch_range.strip()
        current_ref = None

    if not base_ref:
        details = f"Invalid branch range: {branch_range!r}"
        raise ValueError(details)

    # Build both graphs (in parallel when both need worktrees).
    if current_ref:
        with ThreadPoolExecutor(max_workers=2) as executor:
            base_future = executor.submit(_build_graph_for_ref, repo_path, base_ref)
            current_future = executor.submit(_build_graph_for_ref, repo_path, current_ref)
            base_graph = base_future.result()
            current_graph = current_future.result()
    else:
        current_graph = Pipelines(repo_path).build_graph()
        base_graph = _build_graph_for_ref(repo_path, base_ref)

    base_nodes = {n.id: n for n in base_graph.iter_nodes()}
    current_nodes = {n.id: n for n in current_graph.iter_nodes()}
    base_rels = {r.id: r for r in base_graph.iter_relationships()}
    current_rels = {r.id: r for r in current_graph.iter_relationships()}

    return diff_graphs(base_nodes, current_nodes, base_rels, current_rels)


def _build_graph_for_ref(repo_path: Path, ref: str) -> KnowledgeGraph:
    """Build an in-memory graph for a git ref using a temporary worktree."""

    with TemporaryDirectory(prefix="axon_diff_") as tmp_dir:
        worktree_path = Path(tmp_dir) / "worktree"

        _create_worktree(["git", "worktree", "add", f"{worktree_path}", ref], repo_path, ref)
        return _remove_worktree(
            ["git", "worktree", "remove", "--force", f"{worktree_path}"],
            repo_path,
            worktree_path,
        )


def _create_worktree(command: list[str], repo_path: Path, ref: str) -> None:
    """Create a worktree for the given ref."""
    try:
        run(command, cwd=repo_path, capture_output=True, text=True, check=True)
    except CalledProcessError as exc:
        details = f"Failed to create worktree for ref '{ref}': {exc.stderr.strip()}"
        raise RuntimeError(details) from exc


def _remove_worktree(command: list[str], repo_path: Path, worktree_path: Path) -> KnowledgeGraph:
    """Remove a worktree."""
    try:
        graph = Pipelines(worktree_path).build_graph()
    finally:
        try:
            run(command, cwd=repo_path, capture_output=True, text=True, check=True)
        except CalledProcessError:
            logger.warning("Failed to remove worktree at %s", worktree_path)
    return graph


def _format_node_line(node: GraphNode, prefix: str) -> str:
    """Format a single node as a diff line."""
    label = node.label.value.title()
    return f"  {prefix} {node.name} ({label}) -- {node.file_path}"


def _format_node_section(
    nodes: list[GraphNode],
    title: str,
    prefix: str,
) -> list[str]:
    """Format a section of nodes with the given title and prefix."""
    if not nodes:
        return []
    return [
        f"{title} ({len(nodes)}):",
        *[_format_node_line(n, prefix) for n in sorted(nodes, key=lambda n: n.id)],
        "",
    ]


def format_diff(diff: StructuralDiff) -> str:
    """
    Format a StructuralDiff as human-readable output.

    Args :
        diff: The structural diff to format.

    Returns :
        A multi-line string summarizing added, removed, and modified entities.
    """
    total_changes = (
        len(diff.added_nodes)
        + len(diff.removed_nodes)
        + len(diff.modified_nodes)
        + len(diff.added_relationships)
        + len(diff.removed_relationships)
    )

    if total_changes == 0:
        return "No structural differences found."

    lines: list[str] = [f"Structural diff: {total_changes} changes", ""]

    lines.extend(_format_node_section(diff.added_nodes, "Added nodes", "+"))
    lines.extend(_format_node_section(diff.removed_nodes, "Removed nodes", "-"))

    if diff.modified_nodes:
        modified = [current for _, current in diff.modified_nodes]
        lines.extend(_format_node_section(modified, "Modified nodes", "~"))

    lines.extend(
        _format_rel_section(diff.added_relationships, "Added relationships", "+"),
    )
    lines.extend(
        _format_rel_section(diff.removed_relationships, "Removed relationships", "-"),
    )

    return "\n".join(lines).rstrip()


def _format_rel_section(
    relationships: list[GraphRelationship],
    title: str,
    prefix: str,
) -> list[str]:
    """Format a section of relationships with the given title and prefix."""
    if not relationships:
        return []
    return [
        f"{title} ({len(relationships)}):",
        *[_format_rel_line(r, prefix) for r in sorted(relationships, key=lambda r: r.id)],
        "",
    ]


def _format_rel_line(rel: GraphRelationship, prefix: str) -> str:
    """Format a single relationship as a diff line."""
    return f"  {prefix} [{rel.type.value}] {rel.source} -> {rel.target}"
