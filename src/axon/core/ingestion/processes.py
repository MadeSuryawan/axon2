"""
Phase 9: Process / execution flow detection for Axon.

Detects execution flows by finding entry points and tracing call chains
via BFS.  Creates Process nodes and STEP_IN_PROCESS relationships that
represent end-to-end execution paths through the codebase.

The main entry point is :meth:`Process.process_processes`, which finds entry
points, traces execution flows, deduplicates them, and creates Process nodes
in the knowledge graph.
"""

from collections import deque
from logging import getLogger

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)

logger = getLogger(__name__)


class Processes:
    """
    Handles process/execution flow detection for the Axon knowledge graph.

    Scans for entry points (functions with no incoming calls or matching
    framework patterns), traces execution flows via BFS through CALLS
    edges, and creates ``Process`` nodes with ``STEP_IN_PROCESS`` relationships
    representing end-to-end execution paths.
    """

    # Labels for callable nodes (functions and methods)
    _CALLABLE_LABELS: tuple[NodeLabel, ...] = (
        NodeLabel.FUNCTION,
        NodeLabel.METHOD,
    )

    # Maximum number of nodes allowed in a single flow (prevents runaway traces)
    _MAX_FLOW_SIZE = 25

    # Python decorator patterns that indicate entry points
    _PYTHON_DECORATOR_PATTERNS: tuple[str, ...] = (
        "@app.route",
        "@router",
        "@click.command",
    )

    # Default maximum BFS depth for flow tracing
    _DEFAULT_MAX_DEPTH: int = 6

    # Default maximum callees to follow per node at each level
    _DEFAULT_MAX_BRANCHING: int = 3

    # TypeScript files where exports are true entry points (index/entry/app files).
    _TS_ENTRY_SUFFIXES = (
        "index.ts",
        "index.tsx",
        "index.js",
        "index.jsx",
        "main.ts",
        "main.tsx",
        "main.js",
        "app.ts",
        "app.tsx",
        "app.js",
        "server.ts",
        "server.js",
        "handler.ts",
        "handler.js",
        "route.ts",
        "route.tsx",
        "page.tsx",
        "page.ts",
        "layout.tsx",
        "layout.ts",
    )

    def __init__(self, graph: KnowledgeGraph) -> None:
        """
        Initialize the Process analyzer.

        Args:
            graph: The knowledge graph to analyze and enrich.
        """
        self._graph = graph
        self._callable_labels: tuple[NodeLabel, ...] = self._CALLABLE_LABELS

    def process_processes(self) -> int:
        """
        Detect execution flows and create Process nodes in the graph.

        This is the main entry point. It:
        1. Finds all entry points in the graph.
        2. Traces a flow from each entry point.
        3. Deduplicates similar flows.
        4. Filters out trivial flows (single step only).
        5. Creates a Process node and STEP_IN_PROCESS relationships for each flow.

        Returns:
            The number of Process nodes created.
        """
        # Step 1: Find all entry points in the graph
        entry_points = self._find_entry_points()
        logger.debug("Found %d entry points", len(entry_points))

        # Step 2: Trace execution flow from each entry point
        flows = self._trace_all_flows(entry_points)

        # Step 3: Remove duplicate/similar flows
        flows = self._deduplicate_flows(flows)

        # Step 4: Filter out trivial single-step flows (not meaningful processes)
        flows = [f for f in flows if len(f) > 1]

        # Step 5: Create Process nodes and relationships
        return self._create_process_nodes(flows)

    def _find_entry_points(self) -> list[GraphNode]:
        """
        Find functions/methods that serve as execution entry points.

        A node is an entry point if it has NO incoming CALLS relationships,
        or if it matches a recognised framework pattern:
        - Python: test_* functions, main function, decorated functions
        - TypeScript: handler, middleware, exported functions

        Each identified entry point has its ``is_entry_point`` attribute set
        to ``True``.

        Returns:
            A list of entry point :class:`GraphNode` instances.
        """
        entry_points: list[GraphNode] = []

        # Iterate through all callable nodes and check if they're entry points
        for label in self._callable_labels:
            for node in self._graph.get_nodes_by_label(label):
                if not self._is_entry_point(node):
                    continue
                node.is_entry_point = True
                entry_points.append(node)

        return entry_points

    def _is_entry_point(self, node: GraphNode) -> bool:
        """
        Determine whether *node* qualifies as an entry point.

        Framework patterns always qualify. For functions with no incoming calls,
        we require additional evidence (name heuristics, exported status) to avoid
        marking every utility function as an entry point in large codebases.

        Args:
            node: The node to check.

        Returns:
            True if the node is an entry point, False otherwise.
        """
        # Framework patterns (decorators, test_*, main) always qualify
        if self._matches_framework_pattern(node):
            return True

        # Functions with incoming calls are not entry points
        # (they are called by something else)
        if self._graph.get_incoming(node.id, RelType.CALLS):
            return False

        # Exported functions are likely entry points
        if node.is_exported:
            return True

        # Common entry point names
        if node.name in ("main", "cli", "run", "app", "handler", "entrypoint"):
            return True

        # Functions in common entry-point files
        return node.label == NodeLabel.FUNCTION and node.file_path.endswith(
            ("__main__.py", "cli.py", "main.py", "app.py"),
        )

    def _is_ts_entry_file(self, file_path: str) -> bool:
        return any(file_path.endswith(suffix) for suffix in self._TS_ENTRY_SUFFIXES)

    def _matches_framework_pattern(self, node: GraphNode) -> bool:
        """
        Check whether *node* matches a known framework entry point pattern.

        Args:
            node: The node to check.

        Returns:
            True if the node matches a framework pattern, False otherwise.
        """
        name = node.name
        language = node.language.lower() if node.language else ""
        content = node.content or ""

        # Python patterns
        if language in ("python", "py", "") or node.file_path.endswith(".py"):
            # Test functions are entry points
            if name.startswith("test_"):
                return True
            # Main function is an entry point
            if name == "main":
                return True
            # Check for decorator patterns indicating web routes/commands
            for pattern in self._PYTHON_DECORATOR_PATTERNS:
                if pattern in content:
                    return True

        # TypeScript patterns
        if language in ("typescript", "ts", "") or node.file_path.endswith(
            (".ts", ".tsx"),
        ):
            # Handler and middleware functions are entry points
            if name in ("handler", "middleware"):
                return True
            # Exported functions are likely entry points
            if node.is_exported and self._is_ts_entry_file(node.file_path):
                return True

        return False

    def _trace_all_flows(
        self,
        entry_points: list[GraphNode],
    ) -> list[list[GraphNode]]:
        """
        Trace execution flow from each entry point.

        Args:
            entry_points: List of entry point nodes to trace from.

        Returns:
            List of flows, each flow is a list of nodes.
        """
        flows: list[list[GraphNode]] = []
        for ep in entry_points:
            flow = self._trace_flow(ep)
            flows.append(flow)
        return flows

    def _trace_flow(
        self,
        entry_point: GraphNode,
        max_depth: int = _DEFAULT_MAX_DEPTH,
        max_branching: int = _DEFAULT_MAX_BRANCHING,
    ) -> list[GraphNode]:
        """
        BFS from *entry_point* through CALLS edges.

        At each level, at most *max_branching* callees are followed (those
        with higher confidence on the CALLS edge are preferred). Traversal
        stops after *max_depth* levels, when the flow reaches
        :data:`_MAX_FLOW_SIZE` nodes, or when no unvisited callees remain.

        Args:
            entry_point: The starting node for the flow.
            max_depth: Maximum BFS depth.
            max_branching: Maximum callees to follow per node at each level.

        Returns:
            An ordered list of nodes in the flow, starting with *entry_point*.
        """
        # Track visited nodes to avoid cycles and duplicates
        visited: set[str] = {entry_point.id}
        result: list[GraphNode] = [entry_point]

        # Queue stores (node_id, depth) tuples for BFS
        queue: deque[tuple[str, int]] = deque([(entry_point.id, 0)])

        while queue:
            # Safety check: prevent unbounded flow growth
            if len(result) >= self._MAX_FLOW_SIZE:
                break

            current_id, depth = queue.popleft()

            # Stop if we've reached the maximum depth
            if depth >= max_depth:
                continue

            # Get outgoing CALLS relationships and sort by confidence
            outgoing = self._graph.get_outgoing(current_id, RelType.CALLS)
            outgoing.sort(
                key=lambda r: r.properties.get("confidence", 0.0),
                reverse=True,
            )

            # Follow up to max_branching callees
            count = 0
            for rel in outgoing:
                if count >= max_branching or len(result) >= self._MAX_FLOW_SIZE:
                    break

                target_id = rel.target
                if target_id in visited:
                    continue

                # Get the target node; skip if not found
                target_node = self._graph.get_node(target_id)
                if target_node is None:
                    continue

                visited.add(target_id)
                result.append(target_node)
                queue.append((target_id, depth + 1))
                count += 1

        return result

    @staticmethod
    def _generate_process_label(steps: list[GraphNode]) -> str:
        """
        Create a human-readable label from the flow steps.

        Format: ``"EntryName -> Step2 -> Step3"`` (max 4 steps in label).
        If only one step, returns just the function name.

        Args:
            steps: Ordered list of nodes in the flow.

        Returns:
            A string label for the process.
        """
        if not steps:
            return ""

        if len(steps) == 1:
            return steps[0].name

        # Take first 4 steps maximum for readability
        names = [s.name for s in steps[:4]]
        return " \u2192 ".join(names)

    @staticmethod
    def _deduplicate_flows(
        flows: list[list[GraphNode]],
    ) -> list[list[GraphNode]]:
        """
        Remove flows that are too similar to longer ones.

        Two flows are "similar" if they share > 50% of their nodes (by ID).
        When a pair is similar, the shorter flow is discarded.

        This ensures we keep the most comprehensive flow representations
        while avoiding redundant process definitions.

        Args:
            flows: List of flows (each flow is a list of nodes).

        Returns:
            Deduplicated list of flows.
        """
        # Sort by length descending (keep longer flows first)
        flows_sorted = sorted(flows, key=len, reverse=True)

        kept: list[list[GraphNode]] = []
        kept_sets: list[set[str]] = []

        for flow in flows_sorted:
            flow_ids = {n.id for n in flow}
            is_duplicate = False

            # Check against all kept flows
            for kept_set in kept_sets:
                # Skip empty sets to avoid division by zero
                if not flow_ids or not kept_set:
                    continue

                # Calculate overlap percentage
                intersection = flow_ids & kept_set
                smaller_size = min(len(flow_ids), len(kept_set))
                overlap = len(intersection) / smaller_size

                # If overlap > 50%, consider this a duplicate
                if overlap > 0.5:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(flow)
                kept_sets.append(flow_ids)

        return kept

    def _determine_kind(self, steps: list[GraphNode]) -> str:
        """
        Determine whether a flow is intra- or cross-community.

        Checks MEMBER_OF relationships for each step node. If all belong to
        the same community: ``"intra_community"``. If they span multiple:
        ``"cross_community"``. If no communities are assigned: ``"unknown"``.

        Args:
            steps: List of nodes in the flow.

        Returns:
            The kind of flow: "intra_community", "cross_community", or "unknown".
        """
        communities: set[str] = set()
        has_any = False

        for step in steps:
            # Get communities this step belongs to
            for rel in self._graph.get_outgoing(step.id, RelType.MEMBER_OF):
                has_any = True
                communities.add(rel.target)

        # No community information available
        if not has_any:
            return "unknown"

        # Single community = intra, multiple = cross
        if len(communities) <= 1:
            return "intra_community"
        return "cross_community"

    def _create_process_nodes(
        self,
        flows: list[list[GraphNode]],
    ) -> int:
        """
        Create Process nodes and STEP_IN_PROCESS relationships for each flow.

        Args:
            flows: List of deduplicated flows (each flow is a list of nodes).

        Returns:
            The number of Process nodes created.
        """
        count = 0

        for i, steps in enumerate(flows):
            # Generate unique ID and label for the process
            process_id = generate_id(NodeLabel.PROCESS, f"process_{i}")
            label = self._generate_process_label(steps)
            kind = self._determine_kind(steps)

            # Create the Process node
            process_node = GraphNode(
                id=process_id,
                label=NodeLabel.PROCESS,
                name=label,
                properties={"step_count": len(steps), "kind": kind},
            )
            self._graph.add_node(process_node)

            # Create STEP_IN_PROCESS relationships for each step
            for step_number, step in enumerate(steps):
                rel_id = f"step:{step.id}->{process_id}:{step_number}"
                self._graph.add_relationship(
                    GraphRelationship(
                        id=rel_id,
                        type=RelType.STEP_IN_PROCESS,
                        source=step.id,
                        target=process_id,
                        properties={"step_number": step_number},
                    ),
                )

            count += 1

        logger.info("Created %d process nodes", count)
        return count
