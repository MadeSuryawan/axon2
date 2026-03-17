"""
MCP tool handler implementations for Axon.

This module provides the core functionality for handling MCP (Model Context Protocol)
tool requests. Each tool handler accepts a storage backend and tool-specific arguments,
performs the appropriate query, and returns a human-readable string suitable for
inclusion in an MCP ``TextContent`` response.

The main entry point is the :class:`Tools` class, which encapsulates all tool
handlers. Implementation details are delegated to the :class:`_Helpers` base class.
"""

from collections import deque
from json import JSONDecodeError, loads
from logging import getLogger
from pathlib import Path
from re import MULTILINE
from re import compile as re_compile
from typing import Any, cast

from axon.config.constants import MODEL_NAME, SYSTEM_EXCEPTIONS
from axon.core.cypher_guard import WRITE_KEYWORDS, sanitize_cypher
from axon.core.embeddings.embedder import embed_query
from axon.core.graph.model import GraphNode
from axon.core.ingestion.community import Community
from axon.core.ingestion.dead_code import DeadCode
from axon.core.search.hybrid import SearchDeps, hybrid_search
from axon.core.storage.base import StorageBackend
from axon.core.storage.kuzu_backend import escape_cypher as _escape_cypher
from axon.mcp.resources import get_dead_code_list

logger = getLogger(__name__)

# Maximum depth for impact analysis traversal (prevents runaway queries)
MAX_TRAVERSE_DEPTH = 10

# Regex patterns for parsing git diff output
_DIFF_FILE_PATTERN = re_compile(r"^diff --git a/(.+?) b/(.+?)$", MULTILINE)

# Regex pattern for parsing git diff hunk headers
_DIFF_HUNK_PATTERN = re_compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", MULTILINE)

# Human-readable labels for impact analysis depth levels
_DEPTH_LABELS: dict[int, str] = {
    1: "Direct callers (will break)",
    2: "Indirect (may break)",
}

# Regex pattern for validating file paths (prevents injection attacks)
_SAFE_PATH = re_compile(r"^[a-zA-Z0-9._/\-\s]+$")


class _Helpers:
    """
    Base class containing helper implementations for MCP tool handlers.

    This class encapsulates all reusable implementation details and cross-cutting
    concerns used by the public tool handlers. Methods are organized by the public
    method they support, with clear section comments mapping them to their
    corresponding handler.
    """

    # Class-level constants for maximum flow size limits
    _MAX_TRAVERSE_DEPTH: int = MAX_TRAVERSE_DEPTH
    _EMBED_MODEL_NAME: str = MODEL_NAME
    _SNIPPET_MAX_LENGTH: int = 200

    # =============================================================================
    # Shared/Cross-cutting Helpers
    # Used by multiple public methods
    # =============================================================================

    @staticmethod
    def _confidence_tag(confidence: float) -> str:
        """
        Return a visual confidence indicator for edge display.

        Args:
            confidence: Confidence value between 0.0 and 1.0.

        Returns:
            Empty string for high confidence (>= 0.9), "(~)" for medium
            confidence (>= 0.5), "(?)" for low confidence (< 0.5).
        """
        if confidence >= 0.9:
            return ""
        if confidence >= 0.5:
            return " (~)"
        return " (?)"

    @staticmethod
    def _resolve_symbol(storage: StorageBackend, symbol: str) -> list:
        """
        Resolve a symbol name to search results, preferring exact name matches.

        First tries exact_name_search if available, then falls back to FTS.

        Args:
            storage: The storage backend.
            symbol: The symbol name to search for.

        Returns:
            List of search results.
        """
        if hasattr(storage, "exact_name_search") and (
            results := storage.exact_name_search(symbol, limit=1)
        ):
            return results
        return storage.fts_search(symbol, limit=1)

    @staticmethod
    def _group_by_process(
        results: list,
        storage: StorageBackend,
    ) -> dict[str, list]:
        """
        Map search results to their parent execution processes.

        Delegates to ``storage.get_process_memberships()`` for a safe
        parameterized query, falling back to an empty dict if the backend
        does not support the method.

        Args:
            results: List of search results to group.
            storage: The storage backend.

        Returns:
            Dictionary mapping process names to lists of results.
        """
        if not results:
            return {}

        node_ids = [r.node_id for r in results]

        try:
            node_to_process = storage.get_process_memberships(node_ids)
        except (AttributeError, TypeError):
            return {}

        groups: dict[str, list] = {}
        for r in results:
            if not (pname := node_to_process.get(r.node_id)):
                continue
            groups.setdefault(pname, []).append(r)

        return groups

    @staticmethod
    def _format_query_results(results: list, groups: dict[str, list]) -> str:
        """
        Format search results with process grouping.

        Results belonging to a process appear under a labelled section.
        Remaining results appear in an "Other results" section.

        Args:
            results: List of search results.
            groups: Dictionary mapping process names to results.

        Returns:
            Formatted string output.
        """
        # Separate grouped and ungrouped results
        grouped_ids: set[str] = {r.node_id for group in groups.values() for r in group}
        ungrouped = [r for r in results if r.node_id not in grouped_ids]

        lines: list[str] = []
        counter = 1

        # Format grouped results first
        for process_name, proc_results in groups.items():
            lines.append(f"=== {process_name} ===")
            for r in proc_results:
                label = r.label.title() if r.label else "Unknown"
                lines.append(f"{counter}. {r.node_name} ({label}) -- {r.file_path}")
                if r.snippet:
                    snippet = r.snippet[: _Helpers._SNIPPET_MAX_LENGTH].replace("\n", " ").strip()
                    lines.append(f"   {snippet}")
                counter += 1
            lines.append("")

        # Format ungrouped results
        if ungrouped:
            if groups:
                lines.append("=== Other results ===")
            for r in ungrouped:
                label = r.label.title() if r.label else "Unknown"
                lines.append(f"{counter}. {r.node_name} ({label}) -- {r.file_path}")
                if r.snippet:
                    snippet = r.snippet[: _Helpers._SNIPPET_MAX_LENGTH].replace("\n", " ").strip()
                    lines.append(f"   {snippet}")
                counter += 1
            lines.append("")

        lines.append("Next: Use context() on a specific symbol for the full picture.")
        return "\n".join(lines)

    def _get_callers_with_fallback(
        self,
        storage: StorageBackend,
        node_id: str,
    ) -> list[tuple[GraphNode, float]]:
        """
        Get callers with confidence, falling back to callers without confidence.

        Args:
            storage: The storage backend.
            node_id: The node ID to get callers for.

        Returns:
            List of (caller_node, confidence) tuples.
        """
        try:
            return storage.get_callers_with_confidence(node_id)
        except (AttributeError, TypeError):
            return [(c, 1.0) for c in storage.get_callers(node_id)]

    def _get_callees_with_fallback(
        self,
        storage: StorageBackend,
        node_id: str,
    ) -> list[tuple[GraphNode, float]]:
        """
        Get callees with confidence, falling back to callees without confidence.

        Args:
            storage: The storage backend.
            node_id: The node ID to get callees for.

        Returns:
            List of (callee_node, confidence) tuples.
        """
        try:
            return storage.get_callees_with_confidence(node_id)
        except (AttributeError, TypeError):
            return [(c, 1.0) for c in storage.get_callees(node_id)]

    @staticmethod
    def _group_by_depth(
        affected_with_depth: list[tuple[GraphNode, int]],
    ) -> dict[int, list[GraphNode]]:
        """
        Group nodes by their traversal depth.

        Args:
            affected_with_depth: List of (node, depth) tuples.

        Returns:
            Dictionary mapping depth to list of nodes at that depth.
        """
        by_depth: dict[int, list[GraphNode]] = {}
        for node, d in affected_with_depth:
            by_depth.setdefault(d, []).append(node)
        return by_depth

    def _build_confidence_lookup(
        self,
        storage: StorageBackend,
        node_id: str,
    ) -> dict[str, float]:
        """
        Build a lookup dictionary for confidence values of direct callers.

        Args:
            storage: The storage backend.
            node_id: The node ID to get callers for.

        Returns:
            Dictionary mapping node IDs to confidence values.
        """
        conf_lookup: dict[str, float] = {}
        try:
            for node, conf in storage.get_callers_with_confidence(node_id):
                conf_lookup[node.id] = conf
        except (AttributeError, TypeError):
            pass
        return conf_lookup

    def _parse_diff(self, diff: str) -> dict[str, list[tuple[int, int]]]:
        """
        Parse git diff output to extract changed files and line ranges.

        Args:
            diff: Raw git diff output string.

        Returns:
            Dictionary mapping file paths to lists of (start, end) line ranges.
        """
        changed_files: dict[str, list[tuple[int, int]]] = {}
        current_file: str | None = None

        for line in diff.split("\n"):
            # Match file headers
            file_match = _DIFF_FILE_PATTERN.match(line)
            if file_match:
                current_file = file_match.group(2)
                if current_file not in changed_files:
                    changed_files[current_file] = []
                continue

            # Match hunk headers to extract line ranges
            hunk_match = _DIFF_HUNK_PATTERN.match(line)
            if hunk_match and current_file is not None:
                start = int(hunk_match.group(1))
                count = int(hunk_match.group(2) or "1")
                changed_files[current_file].append((start, start + count - 1))

        return changed_files

    # =============================================================================
    # Helpers for handle_list_repos
    # =============================================================================

    def _format_repos_list(self, repos: list[dict[str, Any]]) -> str:
        """
        Format the list of repositories into a readable string.

        Args:
            repos: List of repository metadata dictionaries.

        Returns:
            Formatted string listing all repositories.
        """
        lines = [f"Indexed repositories ({len(repos)}):"]
        lines.append("")

        for i, repo in enumerate(repos, 1):
            name = repo.get("name", "unknown")
            path = repo.get("path", "")
            stats = repo.get("stats", {})
            files = stats.get("files", "?")
            symbols = stats.get("symbols", "?")
            relationships = stats.get("relationships", "?")
            lines.append(f"  {i}. {name}")
            lines.append(f"     Path: {path}")
            lines.append(
                f"     Files: {files}  Symbols: {symbols}  Relationships: {relationships}",
            )
            lines.append("")

        return "\n".join(lines)

    # =============================================================================
    # Helpers for handle_context
    # =============================================================================

    def _resolve_symbol_to_node(
        self,
        storage: StorageBackend,
        symbol: str,
    ) -> GraphNode | None:
        """
        Resolve a symbol name to its corresponding graph node.

        This method performs a two-step resolution:
            1. First attempts exact name matching if the backend supports it
            2. Falls back to full-text search if exact match fails

        Args:
            storage: The storage backend to query.
            symbol: The symbol name to resolve.

        Returns:
            The resolved GraphNode if found, None otherwise.
        """
        # Perform full-text search to find the symbol
        results = self._resolve_symbol(storage, symbol)
        if not results:
            return None

        # Retrieve the actual node from storage using the node_id from search results
        node = storage.get_node(results[0].node_id)
        return node

    def _format_node_header(self, node: GraphNode) -> list[str]:
        """
        Format the header information for a node (name, file, lines, signature).

        Args:
            node: The graph node to format.

        Returns:
            List of formatted lines.
        """
        label_display = node.label.value.title() if node.label else "Unknown"
        lines = [f"Symbol: {node.name} ({label_display})"]
        lines.append(f"File: {node.file_path}:{node.start_line}-{node.end_line}")

        if node.signature:
            lines.append(f"Signature: {node.signature}")

        if node.is_dead:
            lines.append("Status: DEAD CODE (unreachable)")

        return lines

    def _format_callers_section(
        self,
        storage: StorageBackend,
        node: GraphNode,
    ) -> list[str]:
        """
        Format the callers section for a given node.

        Retrieves all functions/methods that call the given node and formats
        them into a readable section with confidence indicators.

        Args:
            storage: The storage backend to query.
            node: The node to get callers for.

        Returns:
            List of formatted strings representing the callers section.
        """
        lines: list[str] = []

        # Get callers with confidence scores, using fallback for older backends
        # that don't support confidence tracking
        if not (callers_raw := self._get_callers_with_fallback(storage, node.id)):
            return lines

        lines.append(f"\nCallers ({len(callers_raw)}):")
        # Format each caller with name, location, and confidence indicator
        for c, conf in callers_raw:
            # confidence_tag returns visual indicator based on confidence level
            tag = self._confidence_tag(conf)
            lines.append(f"  -> {c.name}  {c.file_path}:{c.start_line}{tag}")

        return lines

    def _format_callees_section(
        self,
        storage: StorageBackend,
        node: GraphNode,
    ) -> list[str]:
        """
        Format the callees section for a given node.

        Retrieves all functions/methods called by the given node and formats
        them into a readable section with confidence indicators.

        Args:
            storage: The storage backend to query.
            node: The node to get callees for.

        Returns:
            List of formatted strings representing the callees section.
        """
        lines: list[str] = []

        # Get callees with confidence scores, using fallback for older backends
        if not (callees_raw := self._get_callees_with_fallback(storage, node.id)):
            return lines

        lines.append(f"\nCallees ({len(callees_raw)}):")
        for c, conf in callees_raw:
            tag = self._confidence_tag(conf)
            lines.append(f"  -> {c.name}  {c.file_path}:{c.start_line}{tag}")

        return lines

    def _format_type_refs_section(
        self,
        storage: StorageBackend,
        node: GraphNode,
    ) -> list[str]:
        """
        Format the type references section for a given node.

        Retrieves all type references (usages of this symbol as a type)
        and formats them into a readable section.

        Args:
            storage: The storage backend to query.
            node: The node to get type references for.

        Returns:
            List of formatted strings representing the type references section.
        """
        lines: list[str] = []

        # Query type references directly from storage
        if not (type_refs := storage.get_type_refs(node.id)):
            return lines

        lines.append(f"\nType references ({len(type_refs)}):")
        for t in type_refs:
            lines.append(f"  -> {t.name}  {t.file_path}")

        return lines

    def _format_heritage_section(
        self,
        storage: StorageBackend,
        node: GraphNode,
    ) -> list[str]:
        """
        Format the heritage (extends/implements) section for a given node.

        Queries the graph for inheritance relationships, specifically looking for
        'extends' and 'implements' relationships that define the node's heritage.

        Args:
            storage: The storage backend to query.
            node: The node to get heritage information for.

        Returns:
            List of formatted strings representing the heritage section.
        """
        lines: list[str] = []

        # Escape the node ID to prevent Cypher injection attacks
        # This is critical when interpolating user input into queries
        escaped_id = _escape_cypher(node.id)

        # Execute raw Cypher query to find heritage relationships
        # Looking for CodeRelation edges with type 'extends' or 'implements'
        heritage_rows = (
            storage.execute_raw(
                f"MATCH (n)-[r:CodeRelation]->(parent) "
                f"WHERE n.id = '{escaped_id}' "
                f"AND r.rel_type IN ['extends', 'implements'] "
                f"RETURN parent.name, parent.file_path, r.rel_type",
            )
            or []
        )

        if heritage_rows:
            lines.append(f"\nHeritage ({len(heritage_rows)}):")
            for row in heritage_rows:
                # Extract parent information with fallback for missing values
                parent_name = row[0] or "?"
                parent_file = row[1] or "?"
                # rel_type indicates 'extends' or 'implements'
                rel = row[2] or "?"
                lines.append(f"  -> {rel}: {parent_name}  {parent_file}")

        return lines

    def _format_imported_by_section(
        self,
        storage: StorageBackend,
        node: GraphNode,
    ) -> list[str]:
        """
        Format the imported-by section for a given node.

        Finds all files that import the file containing this node by querying
        the graph for 'imports' relationships in the reverse direction.

        Args:
            storage: The storage backend to query.
            node: The node to get import information for.

        Returns:
            List of formatted strings representing the imported-by section.
        """
        lines: list[str] = []

        # Only query imports if the node has a file path
        # (some nodes may not be associated with a file)
        if not node.file_path:
            return lines

        # Escape the file path to prevent Cypher injection
        escaped_fp = _escape_cypher(node.file_path)

        # Query for files that import this file (reverse imports relationship)
        import_rows = (
            storage.execute_raw(
                f"MATCH (a:File)-[r:CodeRelation]->(b:File) "
                f"WHERE b.file_path = '{escaped_fp}' "
                f"AND r.rel_type = 'imports' "
                f"RETURN a.file_path ORDER BY a.file_path",
            )
            or []
        )

        if import_rows:
            # Extract unique importer file paths, filtering out None values
            importers = [r[0] for r in import_rows if r[0]]
            lines.append(f"\nImported by ({len(importers)}):")
            for imp in importers:
                lines.append(f"  -> {imp}")

        return lines

    # =============================================================================
    # Helpers for handle_impact
    # =============================================================================

    # Note: _resolve_symbol, _group_by_depth, and _build_confidence_lookup
    # are defined in the Shared/Cross-cutting section above

    # =============================================================================
    # Helpers for handle_detect_changes
    # =============================================================================

    # Note: _parse_diff is defined in the Shared/Cross-cutting section above

    def _format_changed_files_with_validation(
        self,
        storage: StorageBackend,
        changed_files: dict[str, list[tuple[int, int]]],
    ) -> str:
        """
        Query storage for symbols in changed files with path validation.

        Args:
            storage: The storage backend.
            changed_files: Dictionary mapping file paths to line ranges.

        Returns:
            Formatted string with affected symbols per file.
        """
        lines = [f"Changed files: {len(changed_files)}"]
        lines.append("")
        total_affected = 0

        for file_path, ranges in changed_files.items():
            # Validate path for security
            if not _SAFE_PATH.match(file_path):
                logger.warning("Skipping unsafe file path in diff: %r", file_path)
                lines.append(f"  {file_path}:")
                lines.append("    (skipped: path contains unsafe characters)")
                lines.append("")
                continue

            affected_symbols = self._query_symbols_in_file(storage, file_path, ranges)

            lines.append(f"  {file_path}:")
            if affected_symbols:
                for sym_name, label, s_line, e_line in affected_symbols:
                    lines.append(f"    - {sym_name} ({label}) lines {s_line}-{e_line}")
                    total_affected += 1
            else:
                lines.append("    (no indexed symbols in changed lines)")
            lines.append("")

        lines.append(f"Total affected symbols: {total_affected}")
        lines.append("")
        lines.append("Next: Use impact() on affected symbols to see downstream effects.")
        return "\n".join(lines)

    def _query_symbols_in_file(
        self,
        storage: StorageBackend,
        file_path: str,
        ranges: list[tuple[int, int]],
    ) -> list[tuple[str, str, int, int]]:
        """
        Query storage for symbols that overlap with the given line ranges.

        Args:
            storage: The storage backend.
            file_path: Path to the file to query.
            ranges: List of (start, end) line ranges.

        Returns:
            List of (name, label, start_line, end_line) tuples.
        """
        affected_symbols = []
        try:
            # Query for nodes in this file
            rows = storage.execute_raw(
                f"MATCH (n) WHERE n.file_path = '{_escape_cypher(file_path)}' "
                f"AND n.start_line > 0 "
                f"RETURN n.id, n.name, n.file_path, n.start_line, n.end_line",
            )
            for row in rows or []:
                node_id = row[0] or ""
                name = row[1] or ""
                start_line = row[3] or 0
                end_line = row[4] or 0
                label_prefix = node_id.split(":", 1)[0] if node_id else ""

                # Check if any of the changed ranges overlap with this symbol
                for start, end in ranges:
                    if start_line <= end and end_line >= start:
                        affected_symbols.append(
                            (name, label_prefix.title(), start_line, end_line),
                        )
                        break
        except SYSTEM_EXCEPTIONS as exc:
            logger.warning(
                "Failed to query symbols for %s: %s",
                file_path,
                exc,
                exc_info=True,
            )
            # Error is handled by the caller

        return affected_symbols

    # =============================================================================
    # Helpers for handle_coupling
    # =============================================================================

    def _validate_file_path_input(self, file_path: str) -> str | None:
        """
        Validate the file_path parameter for file context operations.

        Ensures the file_path parameter is non-empty and contains only safe characters.
        Empty or whitespace-only strings are rejected to prevent unnecessary database queries.
        Unsafe characters are rejected to prevent potential injection attacks.

        Args:
            file_path: The file path to validate.

        Returns:
            None if input is valid.
            Error message string if input is invalid.
        """
        # Check if file_path is provided and contains non-whitespace content
        # strip() ensures we catch whitespace-only strings as empty
        if not file_path or not file_path.strip():
            return "Error: 'file_path' parameter is required and cannot be empty."

        # Check for unsafe characters that could be used in injection attacks
        # The _SAFE_PATH regex validates against path traversal and injection patterns
        if not _SAFE_PATH.match(file_path):
            return "Error: file path contains unsafe characters."

    # =============================================================================
    # Helpers for handle_call_path
    # =============================================================================

    def _validate_call_path_inputs(self, from_symbol: str, to_symbol: str) -> str | None:
        """
        Validate inputs for the call path analysis.

        Ensures both symbol parameters are non-empty and contain actual content.
        Empty or whitespace-only strings are rejected to prevent unnecessary
        database queries.

        Args:
            from_symbol: The starting symbol name to validate.
            to_symbol: The target symbol name to validate.

        Returns:
            None if both inputs are valid.
            Error message string if either input is invalid.
        """
        # Check if from_symbol is provided and contains non-whitespace content
        # strip() ensures we catch whitespace-only strings as empty
        if not from_symbol or not from_symbol.strip():
            return "Error: 'from_symbol' parameter is required and cannot be empty."

        # Check if to_symbol is provided and contains non-whitespace content
        if not to_symbol or not to_symbol.strip():
            return "Error: 'to_symbol' parameter is required and cannot be empty."

        # Both inputs are valid
        return None

    def _resolve_symbols_for_call_path(
        self,
        storage: StorageBackend,
        from_symbol: str,
        to_symbol: str,
    ) -> tuple[GraphNode, GraphNode] | str:
        """
        Resolve symbol names to graph nodes for call path analysis.

        Performs two-step resolution for both source and target symbols:
            1. First attempts exact name matching if the backend supports it
            2. Falls back to full-text search if exact match fails
            3. Retrieves the actual node from storage using resolved node_id

        Args:
            storage: The storage backend to query.
            from_symbol: The source symbol name to resolve.
            to_symbol: The target symbol name to resolve.

        Returns:
            Tuple of (source_node, target_node) if both resolve successfully.
            Error message string if resolution fails for either symbol.
        """
        # Resolve the source symbol
        if not (from_results := self._resolve_symbol(storage, from_symbol)):
            return f"Source symbol '{from_symbol}' not found."

        # Resolve the target symbol
        if not (to_results := self._resolve_symbol(storage, to_symbol)):
            return f"Target symbol '{to_symbol}' not found."

        # Retrieve actual nodes from storage using resolved node IDs
        # This ensures we have the complete node data including edges
        src_node = storage.get_node(from_results[0].node_id)
        tgt_node = storage.get_node(to_results[0].node_id)

        # Verify both nodes were successfully retrieved
        # get_node can return None if the node ID is invalid or was deleted
        if not src_node or not tgt_node:
            return "Could not resolve one or both symbols."

        return (src_node, tgt_node)

    def _find_shortest_call_path_bfs(
        self,
        storage: StorageBackend,
        src_node_id: str,
        tgt_node_id: str,
        max_depth: int,
    ) -> dict[str, str] | None:
        """
        Find the shortest call path using Breadth-First Search.

        Executes BFS traversal from the source node, exploring callees at each
        level before moving deeper. This guarantees finding the shortest path
        (fewest hops) to the target if one exists.

        The algorithm maintains:
            - parent map: tracks the predecessor of each visited node
            - queue: BFS frontier containing (node_id, depth) tuples
            - visited set: prevents revisiting nodes (avoids cycles)

        Args:
            storage: The storage backend to query for callees.
            src_node_id: The starting node ID.
            tgt_node_id: The target node ID to find.
            max_depth: Maximum depth to traverse (excluded - nodes at this
                depth are explored but their children are not).

        Returns:
            Dictionary mapping each visited node_id to its parent node_id
            (the BFS parent map), if the target was found.
            None if the target was not found within max_depth.
        """
        # Initialize BFS data structures
        # parent: maps each visited node to its predecessor in the path
        # queue: BFS frontier - tuples of (node_id, current_depth)
        # visited: set of already-explored node IDs to prevent cycles
        parent: dict[str, str] = {}
        queue: deque[tuple[str, int]] = deque([(src_node_id, 0)])
        visited: set[str] = {src_node_id}

        found = False

        # BFS main loop: process nodes level by level
        while queue:
            # Dequeue the next node to explore
            # popleft ensures FIFO order - nodes are explored in insertion order
            current_id, depth = queue.popleft()

            # Skip expanding nodes at or beyond max_depth
            # We still check them for equality with target, but don't explore their callees
            if depth >= max_depth:
                continue

            # Explore all callees (functions/methods called by current node)
            for callee in storage.get_callees(current_id):
                # Skip if already visited - prevents infinite loops in cyclic graphs
                if callee.id in visited:
                    continue

                # Mark as visited and record parent
                visited.add(callee.id)
                parent[callee.id] = current_id

                # Check if this callee is our target
                if callee.id == tgt_node_id:
                    found = True
                    # Found the target - can stop searching
                    # Since BFS explores level by level, this is guaranteed
                    # to be the shortest path
                    break

                # Add callee to queue for next level exploration
                # Increment depth for tracking max depth limit
                queue.append((callee.id, depth + 1))

            # Exit outer loop if target was found
            if found:
                break

        # Return parent map if target was found, None otherwise
        return parent if found else None

    def _reconstruct_path_from_bfs(
        self,
        parent_map: dict[str, str],
        target_node_id: str,
    ) -> list[str]:
        """
        Reconstruct the node ID path from BFS parent map.

        Walks backwards from the target node using the parent map to build
        the path from source to target, then reverses it to get source-to-target order.

        Args:
            parent_map: Dictionary mapping each node_id to its parent node_id,
                as returned by _find_shortest_call_path_bfs.
            target_node_id: The ID of the target node (end of the path).

        Returns:
            List of node IDs in order from source to target.
        """
        path_ids: list[str] = []
        node_id: str | None = target_node_id

        # Walk backwards from target to source using parent map
        # parent_map.get(node_id) returns None when we reach the source
        # (which has no parent in the map)
        while node_id:
            path_ids.append(node_id)
            node_id = parent_map.get(node_id)

        # Reverse to get source-to-target order
        path_ids.reverse()

        return path_ids

    def _format_call_path_result(
        self,
        storage: StorageBackend,
        path_ids: list[str],
        src_name: str,
        tgt_name: str,
    ) -> str:
        """
        Format the call path result into a human-readable string.

        Converts the path of node IDs into formatted output showing:
            - Header with hop count and path visualization (arrow notation)
            - Each hop with symbol name, type label, file path, and line number

        Args:
            storage: The storage backend to query for node details.
            path_ids: List of node IDs in order from source to target.
            src_name: Original source symbol name (for header).
            tgt_name: Original target symbol name (for header).

        Returns:
            Formatted string representation of the call path.
        """
        # Calculate hop count: number of edges = number of nodes - 1
        hop_count = len(path_ids) - 1

        # Build list of symbol names for the header visualization
        path_names = []
        lines = []

        # Iterate through path IDs to build formatted output
        # Enumerate starting at 1 for human-readable line numbers
        for i, nid in enumerate(path_ids, 1):
            # Retrieve full node details from storage
            if node := storage.get_node(nid):
                # Format node with name, type label, file path, and line number
                label = node.label.value.title() if node.label else "Unknown"
                path_names.append(node.name)
                lines.append(f"  {i}. {node.name} ({label}) — {node.file_path}:{node.start_line}")
                continue
            # Fallback for nodes that couldn't be retrieved
            # (might happen with deleted or invalid nodes)
            path_names.append(nid)
            lines.append(f"  {i}. {nid}")

        # Build header with path visualization and hop count
        # Handle singular/plural for "hop" vs "hops"
        hop_label = f"{hop_count} hop{'s' if hop_count != 1 else ''}"
        header = f"Call path: {' → '.join(path_names)} ({hop_label})"

        # Combine header with detailed path listing
        return header + "\n\n" + "\n".join(lines)

    # =============================================================================
    # Helpers for handle_explain
    # =============================================================================

    def _validate_symbol_input(self, symbol: str) -> str | None:
        """
        Validate the symbol parameter for explain operations.

        Ensures the symbol parameter is non-empty and contains actual content.
        Empty or whitespace-only strings are rejected to prevent unnecessary
        database queries.

        Args:
            symbol: The symbol name to validate.

        Returns:
            None if input is valid.
            Error message string if input is invalid.
        """
        # Check if symbol is provided and contains non-whitespace content
        # strip() ensures we catch whitespace-only strings as empty
        if not symbol or not symbol.strip():
            return "Error: 'symbol' parameter is required and cannot be empty."
        return None

    def _resolve_symbol_for_explain(
        self,
        storage: StorageBackend,
        symbol: str,
    ) -> GraphNode | None:
        """
        Resolve a symbol name to its corresponding graph node for explain operations.

        Performs two-step resolution:
            1. First attempts to find the symbol using full-text search
            2. Retrieves the actual node from storage using resolved node_id

        Args:
            storage: The storage backend to query.
            symbol: The symbol name to resolve.

        Returns:
            The resolved GraphNode if found, None otherwise.
        """
        # Perform full-text search to find the symbol
        if not (results := self._resolve_symbol(storage, symbol)):
            return

        # Retrieve the actual node from storage using the node_id from search results
        return storage.get_node(results[0].node_id)

    def _format_explanation_header(self, node: GraphNode) -> list[str]:
        """
        Format the header section of the explanation.

        Creates the initial header with symbol name and type label,
        including a visual separator line.

        Args:
            node: The graph node to format the header for.

        Returns:
            List of formatted lines for the header section.
        """
        # Get the type label, defaulting to "Unknown" if not present
        # node.label is an enum, so we access its value and convert to title case
        label_display = node.label.value.title() if node.label else "Unknown"

        lines = [f"Explanation: {node.name} ({label_display})"]
        # Add visual separator line for better readability
        lines.append("=" * 48)
        lines.append("")

        return lines

    def _extract_and_format_roles(self, node: GraphNode) -> list[str]:
        """
        Extract and format role information for a symbol.

        Roles indicate special status flags on the symbol:
            - Entry point: Symbol is an entry point (e.g., main function)
            - Exported: Symbol is exported from its module
            - Dead code: Symbol is unreachable (not called by any other symbol)

        Args:
            node: The graph node to extract roles from.

        Returns:
            List of formatted lines for the roles section.
        """
        lines: list[str] = []

        # Collect all applicable roles based on node properties
        # Each role represents a special status in the codebase
        roles = []
        if node.is_entry_point:
            roles.append("Entry point")
        if node.is_exported:
            roles.append("Exported")
        if node.is_dead:
            roles.append("Dead code (unreachable)")

        # Only add the role line if at least one role applies
        # This keeps output clean when symbol has no special roles
        if roles:
            lines.append(f"Role: {', '.join(roles)}")

        return lines

    def _format_location(self, node: GraphNode) -> list[str]:
        """
        Format location information for a symbol.

        Includes the file path, line range, and signature if available.

        Args:
            node: The graph node to format location for.

        Returns:
            List of formatted lines for the location section.
        """
        lines: list[str] = []

        # Format location as "file_path:start_line-end_line"
        lines.append(f"Location: {node.file_path}:{node.start_line}-{node.end_line}")

        # Add signature if available - this shows the function/method signature
        # including parameters and return type if parsed
        if node.signature:
            lines.append(f"Signature: {node.signature}")

        return lines

    def _query_and_format_community(
        self,
        storage: StorageBackend,
        node: GraphNode,
    ) -> list[str]:
        """
        Query and format community membership for a symbol.

        Queries the graph to find which community (if any) the symbol belongs to.
        Communities are groups of densely connected symbols detected during indexing.

        Args:
            storage: The storage backend to query.
            node: The graph node to get community for.

        Returns:
            List of formatted lines for the community section.
        """
        lines: list[str] = []

        # Escape the node ID to prevent Cypher injection attacks
        # This is critical when interpolating user input into queries
        escaped_id = _escape_cypher(node.id)

        # Query for community membership using MEMBER_OF relationship
        # Returns community name if the symbol belongs to a detected community
        comm_rows = (
            storage.execute_raw(
                f"MATCH (n)-[:MEMBER_OF]->(c:Community) WHERE n.id = '{escaped_id}' RETURN c.name",
            )
            or []
        )

        # Add community information if found
        if comm_rows:
            # Extract community name from query result, defaulting to "?" if null
            comm_name = comm_rows[0][0] or "?"
            lines.append(f"Community: {comm_name}")

        return lines

    def _format_callers_callees(
        self,
        storage: StorageBackend,
        node: GraphNode,
    ) -> list[str]:
        """
        Format callers and callees section for a symbol.

        Queries the graph for:
            - Callers: functions/methods that call this symbol
            - Callees: functions/methods called by this symbol

        Each relationship includes a confidence score indicating the reliability
        of the call relationship detection.

        Args:
            storage: The storage backend to query.
            node: The graph node to get callers/callees for.

        Returns:
            List of formatted lines for the callers/callees section.
        """
        lines: list[str] = []

        # Add blank line to separate from previous section
        lines.append("")

        # Query both callers and callees with confidence scores
        # Confidence indicates how reliable the call relationship is
        callers = storage.get_callers_with_confidence(node.id)
        callees = storage.get_callees_with_confidence(node.id)

        # Format callers section
        # Shows up to 5 caller names with count and "+N more" suffix if needed
        if callers:
            # Take first 5 names for display brevity
            caller_names = ", ".join(c.name for c, _ in callers[:5])
            # Add suffix indicating additional callers not shown
            suffix = f" (+{len(callers) - 5} more)" if len(callers) > 5 else ""
            lines.append(f"Called by {len(callers)}: {caller_names}{suffix}")
        else:
            # No callers indicates this is a root function or unreachable (dead code)
            lines.append("Called by: nothing (root or dead)")

        # Format callees section
        # Shows up to 5 callee names with count and "+N more" suffix if needed
        if callees:
            callee_names = ", ".join(c.name for c, _ in callees[:5])
            suffix = f" (+{len(callees) - 5} more)" if len(callees) > 5 else ""
            lines.append(f"Calls {len(callees)}: {callee_names}{suffix}")
        else:
            # No callees indicates this is a leaf function (no further calls)
            lines.append("Calls: nothing (leaf)")

        return lines

    def _query_and_format_processes(
        self,
        storage: StorageBackend,
        node: GraphNode,
    ) -> list[str]:
        """
        Query and format process flow information for a symbol.

        Queries the graph to find which execution processes (if any) this symbol
        is a part of. Process flows represent execution paths that span multiple
        symbols, showing how work flows through the codebase.

        Args:
            storage: The storage backend to query.
            node: The graph node to get process flows for.

        Returns:
            List of formatted lines for the process flows section.
        """
        lines: list[str] = []

        # Escape the node ID to prevent Cypher injection attacks
        escaped_id = _escape_cypher(node.id)

        # Query for process membership using STEP_IN_PROCESS relationship
        # Returns process names that this symbol participates in
        proc_rows = (
            storage.execute_raw(
                f"MATCH (n)-[:STEP_IN_PROCESS]->(p:Process) WHERE n.id = '{escaped_id}' RETURN p.name",
            )
            or []
        )

        # Add process flows section if the symbol participates in any processes
        if proc_rows:
            # Add blank line to separate from previous section
            lines.append("")
            lines.append("Process flows through this symbol:")

            # Format each process name with bullet point
            for row in proc_rows:
                # Extract process name, defaulting to "?" if null
                proc_name = row[0] or "?"
                lines.append(f"  - {proc_name}")

        return lines

    # =============================================================================
    # Helpers for handle_review_risk
    # =============================================================================

    # Note: _parse_diff is shared with handle_detect_changes

    def _find_affected_symbols(
        self,
        storage: StorageBackend,
        changed_files: dict[str, list[tuple[int, int]]],
    ) -> list[tuple[str, str, str, int]]:
        """
        Find symbols in changed files that overlap with modified line ranges.

        For each changed file, queries the knowledge graph for symbols and filters
        to only those whose line ranges overlap with the changed lines. Also
        counts downstream dependents for each affected symbol.

        Args:
            storage: The storage backend to query.
            changed_files: Dictionary mapping file paths to lists of (start, end) line ranges.

        Returns:
            List of tuples containing:
                - name: Symbol name
                - label_prefix: Type label (e.g., Function, Class)
                - file_path: File containing the symbol
                - dependent_count: Number of downstream dependents

        Note:
            - Only considers symbols with start_line > 0 (excludes file-level nodes)
            - Validates file paths against _SAFE_PATH pattern for security
            - Traverses up to depth 2 for dependent counting
        """
        affected_symbols: list[tuple[str, str, str, int]] = []

        # Iterate through each changed file and its line ranges
        for file_path, ranges in changed_files.items():
            # Security check: validate path matches safe pattern
            # Prevents injection attacks via malicious file paths
            if not _SAFE_PATH.match(file_path):
                continue

            # Escape file path for safe inclusion in Cypher query
            escaped = _escape_cypher(file_path)

            # Query for symbols defined in this file
            # Filters to only symbols with valid line numbers (start_line > 0)
            rows = (
                storage.execute_raw(
                    f"MATCH (n) WHERE n.file_path = '{escaped}' "
                    f"AND n.start_line > 0 "
                    f"RETURN n.id, n.name, n.file_path, n.start_line, n.end_line",
                )
                or []
            )

            # Process each symbol and check for overlap with changed ranges
            for row in rows:
                node_id = row[0] or ""
                name = row[1] or ""
                start_line = row[3] or 0
                end_line = row[4] or 0

                # Extract type label from node ID prefix (e.g., "Function:..." -> "Function")
                label_prefix = node_id.split(":", 1)[0].title() if node_id else ""

                # Check if any changed range overlaps with this symbol's definition
                # Line overlap: symbol starts before range ends AND ends after range starts
                if not any(start_line <= end and end_line >= start for start, end in ranges):
                    continue

                # Retrieve full node for additional properties
                node = storage.get_node(node_id)
                dep_count = 0

                if node:
                    # Traverse to find downstream dependents (callers)
                    # Depth 2 captures direct callers and their callers
                    deps = storage.traverse_with_depth(node.id, 2, direction="callers")
                    dep_count = len(deps)

                # Add to affected symbols with all required metadata
                affected_symbols.append((name, label_prefix, file_path, dep_count))

        return affected_symbols

    def _find_missing_cochanges(
        self,
        storage: StorageBackend,
        changed_files: dict[str, list[tuple[int, int]]],
        changed_file_set: set[str],
    ) -> list[tuple[str, str, float]]:
        """
        Find files that typically change together with modified files but are not in the diff.

        Queries temporal coupling relationships to identify files that frequently change
        alongside the modified files but are missing from the current diff. This indicates
        potential incomplete changes that could break related functionality.

        Args:
            storage: The storage backend to query.
            changed_files: Dictionary mapping file paths to line ranges.
            changed_file_set: Set of changed file paths for O(1) membership testing.

        Returns:
            List of tuples containing:
                - coupled_file: Path of file that typically changes together
                - changed_file: The file it's coupled with
                - strength: Coupling strength (0.0 to 1.0)

        Note:
            - Only considers coupling relationships with strength >= 0.5
            - Filters out files already in the diff (changed_file_set)
        """
        missing_cochange: list[tuple[str, str, float]] = []

        # Iterate through each changed file to find its couplings
        for file_path in changed_files:
            # Security check: validate path matches safe pattern
            if not _SAFE_PATH.match(file_path):
                continue

            # Escape file path for safe Cypher query
            escaped = _escape_cypher(file_path)

            # Query COUPLED_WITH relationships for this file
            # Only includes couplings with strength >= 0.5 (significant coupling)
            coupling_rows = (
                storage.execute_raw(
                    f"MATCH (a:File)-[r:COUPLED_WITH]-(b:File) "
                    f"WHERE a.file_path = '{escaped}' AND r.strength >= 0.5 "
                    f"RETURN b.file_path, r.strength",
                )
                or []
            )

            # Check each coupled file to see if it's missing from the diff
            for row in coupling_rows:
                coupled_file = row[0] or ""
                strength = row[1] or 0.0

                # Only include if not already in the diff
                if coupled_file not in changed_file_set:
                    missing_cochange.append((coupled_file, file_path, strength))

        return missing_cochange

    def _find_communities_touched(
        self,
        storage: StorageBackend,
        all_affected_symbols: list[tuple[str, str, str, int]],
    ) -> set[str]:
        """
        Find which code communities are affected by the changed symbols.

        Queries the knowledge graph to determine which communities (densely connected
        groups of symbols) contain at least one affected symbol. Multiple community
        boundary crossings indicate higher risk as changes span architectural modules.

        Args:
            storage: The storage backend to query.
            all_affected_symbols: List of (name, label, file_path, deps) tuples.

        Returns:
            Set of community names that have at least one affected symbol.

        Note:
            - Uses MEMBER_OF relationship to find community membership
            - Constructs node ID from label, file_path, and name for lookup
        """
        communities_touched: set[str] = set()

        # Query community membership for each affected symbol
        for name, label, file_path, _ in all_affected_symbols:
            # Construct the node ID in the format used in the graph
            # Format: "label:file_path:name" (lowercase label)
            escaped = _escape_cypher(f"{label.lower()}:{file_path}:{name}")

            # Query for community membership via MEMBER_OF relationship
            comm_rows = (
                storage.execute_raw(
                    f"MATCH (n)-[:MEMBER_OF]->(c:Community) WHERE n.id = '{escaped}' RETURN c.name",
                )
                or []
            )

            # Add each community to the set (deduplicates automatically)
            for row in comm_rows:
                if row[0]:
                    communities_touched.add(row[0])

        return communities_touched

    def _calculate_risk_score(
        self,
        entry_points_hit: int,
        missing_cochange_count: int,
        total_dependents: int,
        communities_touched_count: int,
    ) -> int:
        """
        Calculate the PR risk score based on multiple metrics.

        Combines various risk factors into a single numeric score (0-10).
        The formula weights entry points most heavily, then missing co-changes,
        then downstream dependents, with a bonus for multi-community changes.

        Args:
            entry_points_hit: Number of entry point symbols affected.
            missing_cochange_count: Number of typically-coupled files missing from diff.
            total_dependents: Total count of downstream dependents across all symbols.
            communities_touched_count: Number of distinct communities affected.

        Returns:
            Risk score integer between 0 and 10 (capped).

        Note:
            - Base score: entry_points + missing_cochanges + (dependents // 10)
            - Bonus: +2 if changes span multiple communities
            - Final score is capped at 10
        """
        # Calculate base score from entry points and missing co-changes
        # Entry points are weighted heavily as they affect application startup/flow
        score = entry_points_hit + missing_cochange_count

        # Add scaled dependent count (divide by 10 to normalize large numbers)
        # This captures the cascade risk of breaking widely-used symbols
        score += total_dependents // 10

        # Add bonus for crossing community boundaries
        # Changes affecting multiple communities have higher architectural risk
        if communities_touched_count > 1:
            score += 2

        # Cap score at maximum of 10 to keep risk level interpretable
        return min(score, 10)

    def _determine_risk_level(self, score: int) -> str:
        """
        Determine risk level string based on numeric score.

        Args:
            score: Numeric risk score (0-10).

        Returns:
            Risk level string: "LOW" for scores <= 3, "MEDIUM" for scores 4-6,
            "HIGH" for scores >= 7.
        """
        if score <= 3:
            return "LOW"
        elif score <= 6:
            return "MEDIUM"
        else:
            return "HIGH"

    def _format_risk_assessment(
        self,
        level: str,
        score: int,
        all_affected_symbols: list[tuple[str, str, str, int]],
        missing_cochange: list[tuple[str, str, float]],
        communities_touched: set[str],
    ) -> str:
        """
        Format the complete PR risk assessment into a human-readable string.

        Constructs the output by combining all collected metrics into sections:
        header with level/score, affected symbols, missing co-changes, and
        community boundary information.

        Args:
            level: Risk level string (LOW/MEDIUM/HIGH).
            score: Numeric risk score (0-10).
            all_affected_symbols: List of affected symbol tuples.
            missing_cochange: List of missing co-change file tuples.
            communities_touched: Set of community names affected.

        Returns:
            Formatted risk assessment string suitable for MCP response.
        """
        # Build output lines starting with header
        lines = ["PR Risk Assessment"]
        lines.append("=" * 48)
        lines.append(f"Risk: {level} (score: {score}/10)")
        lines.append("")

        # Format affected symbols section
        if all_affected_symbols:
            lines.append(f"Changed symbols ({len(all_affected_symbols)}):")
            for name, label, fp, deps in all_affected_symbols:
                tags = []
                # Add downstream dependent count as a tag if greater than 0
                if deps > 0:
                    tags.append(f"{deps} downstream dependents")
                tag_str = f"  [{', '.join(tags)}]" if tags else ""
                lines.append(f"  - {name} ({label}) — {fp}{tag_str}")
        else:
            lines.append("No indexed symbols in changed lines.")

        # Format missing co-changes section if any exist
        if missing_cochange:
            lines.append("")
            lines.append("⚠️ Missing co-change files (usually change together):")
            for missing, coupled_with, strength in missing_cochange:
                lines.append(f"  - {missing} (strength: {strength:.2f} with {coupled_with})")

        # Format community boundary crossing section if multiple communities affected
        if len(communities_touched) > 1:
            lines.append("")
            lines.append(f"Community boundary crossings: {len(communities_touched)}")
            lines.append(f"  Spans: {', '.join(sorted(communities_touched))}")

        return "\n".join(lines)

    # =============================================================================
    # Helpers for handle_file_context
    # =============================================================================

    def _query_file_symbols(
        self,
        storage: StorageBackend,
        escaped_path: str,
    ) -> list[list[Any]]:
        """
        Query all symbols defined in a file from the knowledge graph.

        Retrieves all symbols (functions, classes, methods, etc.) that are defined
        in the specified file, excluding file-level nodes (those with start_line = 0).

        Args:
            storage: The storage backend to query.
            escaped_path: The escaped file path to query symbols for.

        Returns:
            List of tuples containing: (name, label, start_line, is_dead, is_entry_point, is_exported).
            Returns empty list if no symbols are found.
        """
        rows = (
            storage.execute_raw(
                f"MATCH (n) WHERE n.file_path = '{escaped_path}' AND n.start_line > 0 "
                f"RETURN n.name, label(n), n.start_line, n.is_dead, n.is_entry_point, n.is_exported "
                f"ORDER BY n.start_line",
            )
            or []
        )
        return rows

    def _query_file_imports(
        self,
        storage: StorageBackend,
        escaped_path: str,
    ) -> list[list[Any]]:
        """
        Query files that the given file imports (outgoing imports).

        Retrieves all files that have an 'imports' CodeRelation relationship
        from the given file, representing the file's direct dependencies.

        Args:
            storage: The storage backend to query.
            escaped_path: The escaped file path to query imports for.

        Returns:
            List of tuples containing the imported file paths.
            Returns empty list if no imports are found.
        """
        rows = (
            storage.execute_raw(
                f"MATCH (a:File)-[r:CodeRelation]->(b:File) "
                f"WHERE a.file_path = '{escaped_path}' AND r.rel_type = 'imports' "
                f"RETURN b.file_path ORDER BY b.file_path",
            )
            or []
        )
        return rows

    def _query_file_importers(
        self,
        storage: StorageBackend,
        escaped_path: str,
    ) -> list[list[Any]]:
        """
        Query files that import the given file (incoming importers).

        Retrieves all files that have an 'imports' CodeRelation relationship
        to the given file, representing the file's reverse dependencies
        (files that would be affected if this file changes).

        Args:
            storage: The storage backend to query.
            escaped_path: The escaped file path to query importers for.

        Returns:
            List of tuples containing the importer file paths.
            Returns empty list if no importers are found.
        """
        rows = (
            storage.execute_raw(
                f"MATCH (a:File)-[r:CodeRelation]->(b:File) "
                f"WHERE b.file_path = '{escaped_path}' AND r.rel_type = 'imports' "
                f"RETURN a.file_path ORDER BY a.file_path",
            )
            or []
        )
        return rows

    def _query_file_coupling(
        self,
        storage: StorageBackend,
        escaped_path: str,
    ) -> list[list[Any]]:
        """
        Query temporal coupling relationships for a file.

        Retrieves files that are temporally coupled with the given file,
        meaning they tend to change together. Results are limited to top 5
        by coupling strength.

        Temporal coupling is calculated based on commit history and indicates
        hidden dependencies that may not be visible through static imports.

        Args:
            storage: The storage backend to query.
            escaped_path: The escaped file path to query coupling for.

        Returns:
            List of tuples containing: (coupled_file_path, strength, co_changes).
            Returns empty list if no coupling relationships are found.
        """
        rows = (
            storage.execute_raw(
                f"MATCH (a:File)-[r:COUPLED_WITH]-(b:File) "
                f"WHERE a.file_path = '{escaped_path}' "
                f"RETURN b.file_path, r.strength, r.co_changes "
                f"ORDER BY r.strength DESC LIMIT 5",
            )
            or []
        )
        return rows

    def _query_file_dead_code(
        self,
        storage: StorageBackend,
        escaped_path: str,
    ) -> list[list[Any]]:
        """
        Query dead code (unreachable symbols) in a file.

        Retrieves all symbols in the file that are marked as dead code,
        meaning they are not called by any other symbol in the codebase.

        Args:
            storage: The storage backend to query.
            escaped_path: The escaped file path to query dead code for.

        Returns:
            List of tuples containing: (symbol_name, start_line, label).
            Returns empty list if no dead code is found.
        """
        rows = (
            storage.execute_raw(
                f"MATCH (n) WHERE n.is_dead = true AND n.file_path = '{escaped_path}' "
                f"RETURN n.name, n.start_line, label(n)",
            )
            or []
        )
        return rows

    def _query_file_communities(
        self,
        storage: StorageBackend,
        escaped_path: str,
    ) -> list[list[Any]]:
        """
        Query community membership for a file.

        Retrieves all communities that have members in this file,
        along with the count of symbols from this file in each community.

        Communities represent groups of densely connected symbols detected
        during indexing, indicating cohesive modules or packages.

        Args:
            storage: The storage backend to query.
            escaped_path: The escaped file path to query communities for.

        Returns:
            List of tuples containing: (community_name, symbol_count).
            Returns empty list if the file doesn't belong to any communities.
        """
        rows = (
            storage.execute_raw(
                f"MATCH (n)-[r:CodeRelation]->(c:Community) "
                f"WHERE n.file_path = '{escaped_path}' AND r.rel_type = 'member_of' "
                f"RETURN c.name, count(n) ORDER BY count(n) DESC",
            )
            or []
        )
        return rows

    def _format_file_context_header(self, file_path: str) -> list[str]:
        """
        Format the header section of the file context output.

        Creates the initial header with file path and a visual separator line.

        Args:
            file_path: The file path to include in the header.

        Returns:
            List of formatted lines for the header section.
        """
        lines = [f"File: {file_path}"]
        lines.append("=" * 48)
        return lines

    def _format_file_symbols_section(self, sym_rows: list[list[Any]]) -> list[str]:
        """
        Format the symbols section of the file context output.

        Takes the raw database rows from symbol queries and formats them into
        a readable section showing all definitions in the file.

        Args:
            sym_rows: List of tuples from _query_file_symbols containing:
                (name, label, start_line, is_dead, is_entry_point, is_exported)

        Returns:
            List of formatted lines for the symbols section.
            Empty list if no symbols are present.
        """
        lines: list[str] = []

        # Only add section if symbols were found in the file
        if not sym_rows:
            return lines

        lines.append("")
        lines.append(f"Symbols ({len(sym_rows)}):")

        # Format each symbol with its name, type, line number, and applicable tags
        for row in sym_rows:
            # Extract symbol data from the row tuple with fallbacks for None values
            # Using index-based access with safe defaults
            name = row[0] or "?"
            label = row[1] or "Unknown"
            start_line = row[2] or 0

            # Extract boolean flags for tags - these may not exist in older data
            # Using conditional length check to handle schema variations
            is_dead = row[3] if len(row) > 3 else False
            is_entry = row[4] if len(row) > 4 else False
            is_exported = row[5] if len(row) > 5 else False

            # Build tags list based on symbol properties
            # These indicate special roles or status in the codebase
            tags = []
            if is_entry:
                tags.append("entry point")
            if is_exported:
                tags.append("exported")
            if is_dead:
                tags.append("dead")

            # Format tags as bracketed suffix if any tags are present
            tag_str = f"  [{', '.join(tags)}]" if tags else ""

            # Format individual symbol line with name, type, location, and tags
            lines.append(f"  - {name} ({label}) line {start_line}{tag_str}")

        return lines

    def _format_file_imports_section(
        self,
        imports_out: list[list[Any]],
        imports_in: list[list[Any]],
    ) -> list[str]:
        """
        Format the imports and importers section of the file context output.

        Takes the raw database rows from import queries and formats them into
        a readable section showing both outgoing imports and incoming importers.

        Args:
            imports_out: List of tuples from _query_file_imports (files this file imports).
            imports_in: List of tuples from _query_file_importers (files that import this file).

        Returns:
            List of formatted lines for the imports section.
            Empty list if neither imports nor importers exist.
        """
        lines: list[str] = []

        # Only add section if there are either outgoing imports or incoming importers
        if not imports_out and not imports_in:
            return lines

        # Format outgoing imports: files that this file depends on
        if imports_out:
            # Extract file paths from query results, filtering out None values
            out_paths = [r[0] for r in imports_out if r[0]]
            lines.append("")
            lines.append(f"Imports ({len(out_paths)}): {', '.join(out_paths)}")

        # Format incoming importers: files that depend on this file
        if imports_in:
            # Extract file paths from query results, filtering out None values
            in_paths = [r[0] for r in imports_in if r[0]]
            lines.append(f"Imported by ({len(in_paths)}): {', '.join(in_paths)}")

        return lines

    def _format_file_coupling_section(
        self,
        coupling_rows: list[list[Any]],
    ) -> list[str]:
        """
        Format the temporal coupling section of the file context output.

        Takes the raw database rows from coupling queries and formats them into
        a readable section showing files that typically change together with
        this file.

        Args:
            coupling_rows: List of tuples from _query_file_coupling containing:
                (coupled_file_path, strength, co_changes)

        Returns:
            List of formatted lines for the coupling section.
            Empty list if no coupling relationships exist.
        """
        lines: list[str] = []

        # Only add section if coupling relationships were found
        if not coupling_rows:
            return lines

        lines.append("")
        lines.append(f"Coupled files ({len(coupling_rows)}):")

        # Format each coupled file with its coupling strength and co-change count
        for row in coupling_rows:
            # Extract coupling data from the row tuple
            coupled_path = row[0] or "?"
            strength = row[1] or 0.0  # Default to 0.0 if strength is None
            co_changes = row[2] or 0  # Default to 0 if co_changes is None

            # Format line with coupled file path, strength (0.0-1.0), and change frequency
            lines.append(
                f"  - {coupled_path}  strength: {strength:.2f}  co_changes: {co_changes}",
            )

        return lines

    def _format_file_dead_code_section(
        self,
        dead_rows: list[list[Any]],
    ) -> list[str]:
        """
        Format the dead code section of the file context output.

        Takes the raw database rows from dead code queries and formats them into
        a readable section showing unreachable symbols in the file.

        Args:
            dead_rows: List of tuples from _query_file_dead_code containing:
                (symbol_name, start_line, label)

        Returns:
            List of formatted lines for the dead code section.
            Empty list if no dead code exists.
        """
        lines: list[str] = []

        # Only add section if dead code was found in the file
        if not dead_rows:
            return lines

        lines.append("")
        lines.append(f"Dead code ({len(dead_rows)}):")

        # Format each dead code symbol with its name, type, and location
        for row in dead_rows:
            # Extract symbol data from the row tuple
            name = row[0] or "?"
            start_line = row[1] or 0
            label = row[2] or "Unknown"

            # Format line with symbol name, type, and line number
            lines.append(f"  - {name} ({label}) line {start_line}")

        return lines

    def _format_file_communities_section(
        self,
        comm_rows: list[list[Any]],
    ) -> list[str]:
        """
        Format the communities section of the file context output.

        Takes the raw database rows from community queries and formats them into
        a readable section showing which code communities this file belongs to.

        Args:
            comm_rows: List of tuples from _query_file_communities containing:
                (community_name, symbol_count)

        Returns:
            List of formatted lines for the communities section.
            Empty list if the file doesn't belong to any communities.
        """
        lines: list[str] = []

        # Only add section if the file belongs to any communities
        if not comm_rows:
            return lines

        lines.append("")

        # Format each community with its name and symbol count from this file
        # Filter out None community names before formatting
        comm_parts = [f"{r[0]} ({r[1]} symbols)" for r in comm_rows if r[0]]
        lines.append(f"Communities: {', '.join(comm_parts)}")

        return lines

    # =============================================================================
    # Helpers for handle_test_impact
    # =============================================================================

    def _extract_changed_symbols(
        self,
        storage: StorageBackend,
        diff: str,
        symbols: list[str] | None,
    ) -> list[tuple[str, str]]:
        """
        Extract changed symbols from either a diff or a symbol list.

        This method handles both input sources:
            - Diff: parses the diff to find changed files and line ranges,
              then queries storage for symbols in those ranges
            - Symbols: directly resolves the provided symbol names to graph nodes

        Args:
            storage: The storage backend to query.
            diff: Raw git diff output string (if provided).
            symbols: Optional list of symbol names to resolve directly.

        Returns:
            List of tuples containing (node_id, symbol_name) for all changed symbols.
            Returns empty list if no symbols are found.
        """
        changed_symbol_ids: list[tuple[str, str]] = []

        # Check if diff is provided and non-empty
        # Diff parsing takes precedence over symbol list
        if diff and diff.strip():
            # Parse the diff to extract changed files and their line ranges
            changed_files = self._parse_diff(diff)

            # Iterate through each changed file to find overlapping symbols
            for file_path, ranges in changed_files.items():
                # Security check: validate path matches safe pattern
                # Prevents injection attacks via malicious file paths
                if not _SAFE_PATH.match(file_path):
                    continue

                # Escape file path for safe Cypher query
                escaped = _escape_cypher(file_path)

                # Query for symbols defined in this file
                # Filters to only symbols with valid line numbers (start_line > 0)
                rows = (
                    storage.execute_raw(
                        f"MATCH (n) WHERE n.file_path = '{escaped}' "
                        f"AND n.start_line > 0 "
                        f"RETURN n.id, n.name, n.start_line, n.end_line",
                    )
                    or []
                )

                # Process each symbol and check for overlap with changed line ranges
                for row in rows:
                    node_id = row[0] or ""
                    name = row[1] or ""
                    start_line = row[2] or 0
                    end_line = row[3] or 0

                    # Check if any changed range overlaps with this symbol's definition
                    # Line overlap: symbol starts before range ends AND ends after range starts
                    if not any(start_line <= end and end_line >= start for start, end in ranges):
                        continue

                    changed_symbol_ids.append((node_id, name))

        # If no diff provided, try resolving symbols from the symbol list
        elif symbols:
            for sym_name in symbols:
                # Resolve symbol name to graph node using full-text search
                if (results := self._resolve_symbol(storage, sym_name)) and (
                    node := storage.get_node(results[0].node_id)
                ):
                    changed_symbol_ids.append((node.id, node.name))

        return changed_symbol_ids

    def _find_test_files_in_call_graph(
        self,
        storage: StorageBackend,
        changed_symbol_ids: list[tuple[str, str]],
    ) -> dict[str, list[tuple[str, str, int]]]:
        """
        Find test files in the call graph of the changed symbols.

        Traverses the call graph from each changed symbol (following CALLS edges
        in reverse, i.e., finding callers) to identify test files that exercise
        the changed code. Test file detection is based on file path matching.

        Args:
            storage: The storage backend to query.
            changed_symbol_ids: List of (node_id, symbol_name) tuples to analyze.

        Returns:
            Dictionary mapping test file paths to lists of
            (test_name, changed_symbol_name, depth) tuples.

        Note:
            - Uses a maximum traversal depth of 4 to limit query scope
            - Traverses in "callers" direction to find what calls the changed symbols
            - Test file detection uses _is_test_file which checks for 'test' in path
        """

        # Dictionary to collect test hits: file_path -> list of (test_name, source_sym, depth)
        test_hits: dict[str, list[tuple[str, str, int]]] = {}

        # Traverse call graph for each changed symbol
        for sym_id, sym_name in changed_symbol_ids:
            # Get all callers up to depth 4 (finds direct and indirect callers)
            # direction="callers" finds symbols that call this symbol
            for caller, depth in storage.traverse_with_depth(sym_id, 4, direction="callers"):
                # Check if the caller is from a test file
                if not DeadCode(storage.load_graph())._is_test_file(caller.file_path):
                    continue

                # Add to test hits: track which test calls which changed symbol
                test_hits.setdefault(caller.file_path, []).append((caller.name, sym_name, depth))

        return test_hits

    def _categorize_test_hits_by_depth(
        self,
        test_hits: dict[str, list[tuple[str, str, int]]],
    ) -> tuple[dict[str, list[tuple[str, str, int]]], dict[str, list[tuple[str, str, int]]]]:
        """
        Categorize test hits by their depth in the call graph.

        Separates test hits into direct (depth <= 2) and transitive (depth > 2) categories.
        Direct tests are more likely to exercise the changed code directly, while transitive
        tests go through intermediate symbols.

        Args:
            test_hits: Dictionary mapping test file paths to lists of
                (test_name, changed_symbol_name, depth) tuples.

        Returns:
            Tuple of (direct_files, transitive_files) where each is a dictionary
            mapping test file paths to their hit lists.
        """
        # Direct files: depth <= 2 (tests that closely exercise changed code)
        # Transitive files: depth > 2 (tests that go through intermediate symbols)
        direct_files: dict[str, list[tuple[str, str, int]]] = {}
        transitive_files: dict[str, list[tuple[str, str, int]]] = {}

        for test_file, hits in test_hits.items():
            for test_name, source_sym, depth in hits:
                if depth <= 2:
                    direct_files.setdefault(test_file, []).append((test_name, source_sym, depth))
                else:
                    transitive_files.setdefault(test_file, []).append(
                        (test_name, source_sym, depth),
                    )

        return direct_files, transitive_files

    def _format_test_impact_output(
        self,
        changed_symbol_count: int,
        test_hits: dict[str, list[tuple[str, str, int]]],
        direct_files: dict[str, list[tuple[str, str, int]]],
        transitive_files: dict[str, list[tuple[str, str, int]]],
    ) -> str:
        """
        Format the test impact analysis into a human-readable string.

        Constructs the output by combining all collected data into sections:
        header with counts, direct test coverage, and transitive test coverage.

        Args:
            changed_symbol_count: Number of changed symbols analyzed.
            test_hits: Dictionary of all test hits before categorization.
            direct_files: Dictionary of direct test hits (depth <= 2).
            transitive_files: Dictionary of transitive test hits (depth > 2).

        Returns:
            Formatted test impact analysis string suitable for MCP response.
        """
        lines = ["Test Impact Analysis"]
        lines.append("=" * 48)
        lines.append(f"Changed symbols: {changed_symbol_count}")
        lines.append("")

        # Calculate total unique test cases across all files
        total_tests = sum(len(v) for v in test_hits.values())

        # Format direct test coverage section
        if direct_files:
            lines.append(f"Affected tests ({total_tests}):")
            for test_file, hits in sorted(direct_files.items()):
                lines.append(f"  {test_file}:")
                # Use seen set to deduplicate identical test entries
                seen = set()
                for test_name, source_sym, _ in hits:
                    key = (test_name, source_sym)
                    if key in seen:
                        continue

                    seen.add(key)
                    lines.append(f"    - {test_name} (calls: {source_sym})")
            lines.append("")

        # Format transitive (indirect) test coverage section
        if transitive_files:
            lines.append("Tests with indirect coverage (depth 3+):")
            for test_file, hits in sorted(transitive_files.items()):
                lines.append(f"  {test_file}:")
                # Use seen set to deduplicate identical test entries
                seen = set()
                for test_name, source_sym, _ in hits:
                    key = (test_name, source_sym)
                    if key in seen:
                        continue

                    seen.add(key)
                    lines.append(f"    - {test_name} (transitive via: {source_sym})")

        return "\n".join(lines)

    # =============================================================================
    # Helpers for handle_communities
    # =============================================================================

    def _handle_specific_community(self, storage: StorageBackend, community: str) -> str:
        """
        Handle the case when a specific community name is provided.

        Queries the storage backend for all symbols that are members of the
        specified community and formats them into a readable list with
        metadata such as symbol type, file location, and tags.

        Args:
            storage: The storage backend to query.
            community: The name of the community to retrieve members for.

        Returns:
            Formatted string with community header and member list,
            or an error message if the community is not found or has no members.
        """
        # Query community members from the graph database
        # Using MEMBER_OF relationship to connect symbols to their community
        if not (rows := self._query_community_members(storage, community)):
            # Handle case where community doesn't exist or has no members
            return f"Community '{community}' not found or has no members."

        # Format the retrieved members into readable output
        return self._format_community_members(community, rows)

    def _query_community_members(
        self,
        storage: StorageBackend,
        community: str,
    ) -> list[list[Any]]:
        """
        Query all members of a specific community from the graph.

        Executes a Cypher query to retrieve symbols that belong to the given
        community, ordered by file path and line number for easy navigation.

        Args:
            storage: The storage backend to query.
            community: The name of the community to get members for.

        Returns:
            List of tuples containing member data: (name, label, file_path,
            start_line, is_entry_point, is_exported). Returns empty list
            if no members found.
        """
        # Escape community name to prevent Cypher injection attacks
        # This is critical when interpolating user input into queries
        escaped = _escape_cypher(community)

        # Execute Cypher query to find all symbols that are members of this community
        # ORDER BY ensures consistent, predictable output for users
        rows = (
            storage.execute_raw(
                f"MATCH (n)-[:MEMBER_OF]->(c:Community) "
                f"WHERE c.name = '{escaped}' "
                f"RETURN n.name, label(n), n.file_path, n.start_line, "
                f"n.is_entry_point, n.is_exported "
                f"ORDER BY n.file_path, n.start_line",
            )
            or []
        )

        return rows

    def _format_community_members(
        self,
        community: str,
        rows: list[list[Any]],
    ) -> str:
        """
        Format community members into a human-readable string.

        Takes the raw database rows and formats them with community header,
        member count, and detailed listing of each member with their metadata.

        Args:
            community: The name of the community (used in header).
            rows: List of tuples from query containing member data.

        Returns:
            Formatted string with community name, member count, and
            enumerated list of members with their types and locations.
        """
        lines = [f"Community: {community}"]
        lines.append(f"Members ({len(rows)}):")
        lines.append("")

        # Iterate through each member row and format with metadata
        for row in rows:
            # Extract values from the row tuple with fallbacks for None
            # Using index-based access with safe defaults
            name = row[0] or "?"
            label = row[1] or "Unknown"
            file_path = row[2] or "?"
            start_line = row[3] or 0

            # Extract boolean flags for tags - these may not exist in older data
            # Using conditional length check to handle schema variations
            is_entry = row[4] if len(row) > 4 else False
            is_exported = row[5] if len(row) > 5 else False

            # Build tags list based on symbol properties
            # These indicate special roles in the codebase
            tags = []
            if is_entry:
                tags.append("entry point")
            if is_exported:
                tags.append("exported")

            # Format tags as bracketed suffix if any tags are present
            tag_str = f"  [{', '.join(tags)}]" if tags else ""

            # Format individual member line with name, type, location, and tags
            lines.append(f"  - {name} ({label}) — {file_path}:{start_line}{tag_str}")

        return "\n".join(lines)

    def _handle_all_communities(self, storage: StorageBackend) -> str:
        """
        Handle the case when no specific community is provided.

        Queries all communities from the graph along with their cohesion scores,
        symbol counts, and cross-community processes. Cross-community processes
        are execution paths that span multiple communities, indicating potential
        integration points or architectural boundaries.

        Args:
            storage: The storage backend to query.

        Returns:
            Formatted string with all communities listed by cohesion score,
            plus any cross-community processes if they exist.
        """
        # Query all communities from the database
        # Returns community name, cohesion score, and properties JSON
        if not (rows := self._query_all_communities(storage)):
            # Handle case where no communities have been detected
            # This can happen if community detection was not enabled during indexing
            return "No communities detected. Run indexing with community detection enabled."

        # Build output lines starting with header showing total count
        lines = [f"Communities ({len(rows)} detected):"]
        lines.append("")

        # Format each community with its stats
        for i, row in enumerate(rows, 1):
            # Extract community data from the row tuple
            name = row[0] or "?"
            cohesion = row[1] or 0.0
            props_raw = row[2] or "{}"

            # Parse the properties JSON to extract symbol count
            # This JSON contains metadata about the community
            symbol_count = self._parse_community_properties(props_raw)

            # Format community line with name, cohesion score, and member count
            lines.append(f"  {i}. {name}  (cohesion: {cohesion:.2f}, {symbol_count} symbols)")

        # Add cross-community processes section if any exist
        if cross_procs := self._query_cross_community_processes(storage):
            lines.append("")
            lines.append("Cross-community processes:")
            lines.extend(self._format_cross_community_processes(cross_procs))

        return "\n".join(lines)

    def _query_all_communities(self, storage: StorageBackend) -> list[list[Any]]:
        """
        Query all communities from the graph with their statistics.

        Retrieves communities ordered by cohesion score (highest first), where
        cohesion represents the density of connections within the community.

        Args:
            storage: The storage backend to query.

        Returns:
            List of tuples containing: (name, cohesion, properties_json).
            Returns empty list if no communities exist.
        """
        rows = (
            storage.execute_raw(
                "MATCH (c:Community) "
                "RETURN c.name, c.cohesion, c.properties_json "
                "ORDER BY c.cohesion DESC",
            )
            or []
        )

        return rows

    def _parse_community_properties(self, props_raw: str | dict) -> str:
        """
        Parse community properties JSON to extract symbol count.

        Safely parses the properties JSON string from the database, handling
        both string and dict inputs, and returns the symbol count as a string.

        Args:
            props_raw: Either a JSON string or dict containing community properties.

        Returns:
            String representation of symbol count, or '?' if parsing fails
            or count is not present in the properties.
        """
        # Handle case where properties might already be a dict (from other sources)
        try:
            props = loads(props_raw) if isinstance(props_raw, str) else props_raw
        except (JSONDecodeError, TypeError):
            # If JSON parsing fails, return unknown count
            props = {}

        # Extract symbol_count from parsed properties
        # Default to '?' if not present in the JSON
        return props.get("symbol_count", "?")

    def _query_cross_community_processes(self, storage: StorageBackend) -> list[list[Any]]:
        """
        Query processes that span multiple communities.

        Finds processes that have members in more than one community, indicating
        integration points or architectural boundaries in the codebase.

        Args:
            storage: The storage backend to query.

        Returns:
            List of tuples containing: (process_name, list_of_community_names).
            Returns empty list if no cross-community processes exist.
        """
        cross_procs = (
            storage.execute_raw(
                "MATCH (n)-[:STEP_IN_PROCESS]->(p:Process), (n)-[:MEMBER_OF]->(c:Community) "
                "WITH p.name AS proc, collect(DISTINCT c.name) AS comms "
                "WHERE size(comms) > 1 "
                "RETURN proc, comms",
            )
            or []
        )

        return cross_procs

    def _format_cross_community_processes(
        self,
        cross_procs: list[list[Any]],
    ) -> list[str]:
        """
        Format cross-community processes into readable lines.

        Takes the raw database results and formats each process with its
        associated communities, showing the flow across community boundaries.

        Args:
            cross_procs: List of tuples from _query_cross_community_processes.

        Returns:
            List of formatted strings, one per cross-community process.
        """
        lines = []

        for row in cross_procs:
            # Extract process name and community list from the row
            proc_name = row[0] or "?"
            comms = row[1] if len(row) > 1 else []

            # Convert community list to arrow-separated string
            # This shows the flow/transition between communities
            comm_str = " → ".join(comms) if isinstance(comms, list) else str(comms)

            # Format line with process name and the communities it spans
            lines.append(f"  - {proc_name} ({comm_str})")

        return lines


class MCPTools(_Helpers):
    """
    Handles MCP tool requests for the Axon knowledge graph.

    This class encapsulates all tool handlers, providing
    a clean interface for processing MCP tool requests. Each method handles
    a specific tool type (query, context, impact, etc.) and returns formatted
    results suitable for MCP responses.

    Implementation details are inherited from the :class:`_Helpers` base class.
    """

    def __init__(self) -> None:
        """
        Initialize the Tools handler.

        Args:
            storage: Optional storage backend. If not provided, handlers that
                require storage will need it passed as an argument.
        """

    # -------------------------------------------------------------------------
    # Public handler methods (these are the main entry points for MCP tools)
    # -------------------------------------------------------------------------

    def handle_list_repos(self, registry_dir: Path | None = None) -> str:
        """
        List indexed repositories by scanning for .axon directories.

        Scans the global registry directory (defaults to ``~/.axon/repos``) for
        project metadata files and returns a formatted summary.

        Args:
            registry_dir: Directory containing repo metadata. If ``None``,
                defaults to ``~/.axon/repos``.

        Returns:
            Formatted list of indexed repositories with stats, or a message
            indicating none were found.
        """
        use_cwd_fallback = registry_dir is None
        if registry_dir is None:
            registry_dir = Path.home() / ".axon" / "repos"

        repos: list[dict[str, Any]] = []

        # Scan registry directory for meta.json files
        if registry_dir.exists():
            for meta_file in registry_dir.glob("*/meta.json"):
                try:
                    data = loads(meta_file.read_text())
                    repos.append(data)
                except (JSONDecodeError, OSError):
                    continue

        # Fall back: scan current directory for .axon
        if not repos and use_cwd_fallback:
            cwd_axon = Path.cwd() / ".axon" / "meta.json"
            if cwd_axon.exists():
                try:
                    data = loads(cwd_axon.read_text())
                    repos.append(data)
                except (JSONDecodeError, OSError):
                    pass

        if not repos:
            return "No indexed repositories found. Run `axon index` on a project first."

        # Format the results
        return self._format_repos_list(repos)

    def handle_query(self, storage: StorageBackend, query: str, limit: int = 20) -> str:
        """
        Execute hybrid search and format results, grouped by execution process.

        Args:
            storage: The storage backend to search against.
            query: Text search query.
            limit: Maximum number of results (default 20, capped at 100).

        Returns:
            Formatted search results grouped by process, with file, name, label,
            and snippet for each result.
        """
        # Cap limit to prevent excessive results
        limit = max(1, min(limit, 100))

        # Try to get query embedding for hybrid search, fall back to FTS-only
        if not (query_embedding := embed_query(query)):
            logger.warning("embed_query returned None; falling back to FTS-only search")

        # Execute hybrid search
        results = hybrid_search(
            SearchDeps(
                query=query,
                storage=storage,
                query_embedding=query_embedding,
                limit=limit,
            ),
        )
        if not results:
            return f"No results found for '{query}'."

        # Group results by process and format output
        groups = self._group_by_process(results, storage)
        return self._format_query_results(results, groups)

    def handle_context(self, storage: StorageBackend, symbol: str) -> str:
        """
        Provide a 360-degree view of a symbol.

        This function retrieves comprehensive context information about a given symbol,
        including its callers, callees, type references, heritage (extends/implements),
        and import relationships. The information is formatted into a human-readable
        string suitable for MCP responses.

        The context retrieval follows these steps:
            1. Validate the symbol parameter is non-empty
            2. Resolve the symbol name to a graph node via full-text search
            3. Retrieve and format each context category (callers, callees, types, etc.)
            4. Combine all sections into a unified output with usage guidance

        Args:
            storage: The storage backend instance used for graph queries.
                Must implement get_node, fts_search, get_callers_with_confidence,
                get_callees_with_confidence, get_type_refs, and execute_raw methods.
            symbol: The symbol name to look up. Can be a function, class, method,
                variable, or any other code symbol stored in the knowledge graph.

        Returns:
            Formatted string containing:
                - Symbol header (name, type, file location, signature)
                - Callers section (functions/methods that call this symbol)
                - Callees section (functions/methods called by this symbol)
                - Type references (types that reference this symbol)
                - Heritage (parent classes/modules this symbol extends/implements)
                - Import relationships (which files import this symbol)
                - Next steps guidance for further analysis

        Raises:
            No explicit exceptions are raised. All database operations are wrapped
            in try-except blocks or use safe fallback mechanisms. Invalid input
            returns error messages instead of raising exceptions.

        Note:
            - If the symbol is not found, returns an error message indicating this
            - Confidence scores are included for caller/callee relationships when available
            - The function gracefully handles missing backend capabilities by falling back
              to basic queries without confidence scores
        """
        # Step 1: Validate symbol parameter - ensure it's non-empty
        # This prevents unnecessary database queries for invalid input
        if not symbol or not symbol.strip():
            return "Error: 'symbol' parameter is required and cannot be empty."

        # Step 2: Resolve symbol to a node in the graph
        # Uses full-text search to find the symbol; returns error if not found
        if not (node := self._resolve_symbol_to_node(storage, symbol)):
            return f"Symbol '{symbol}' not found."

        # Step 3: Build the context output by collecting all sections
        # Start with the symbol header information
        lines = self._format_node_header(node)

        # Add each context section if data is available
        # Callers: functions/methods that invoke this symbol
        lines.extend(self._format_callers_section(storage, node))

        # Callees: functions/methods called by this symbol
        lines.extend(self._format_callees_section(storage, node))

        # Type references: types that reference this symbol
        lines.extend(self._format_type_refs_section(storage, node))

        # Heritage: parent classes/modules (extends/implements relationships)
        lines.extend(self._format_heritage_section(storage, node))

        # Import relationships: files that import this file/symbol
        lines.extend(self._format_imported_by_section(storage, node))

        # Step 4: Add footer with guidance for next steps
        lines.append("")
        lines.append("Next: Use impact() if planning changes to this symbol.")
        return "\n".join(lines)

    def handle_impact(
        self,
        storage: StorageBackend,
        symbol: str,
        depth: int = 3,
    ) -> str:
        """
        Analyse the blast radius of changing a symbol, grouped by hop depth.

        Uses BFS traversal through CALLS edges to find all affected symbols
        up to the specified depth, then groups results by distance.

        Args:
            storage: The storage backend.
            symbol: The symbol name to analyse.
            depth: Maximum traversal depth (default 3).

        Returns:
            Formatted impact analysis with depth-grouped sections.
        """
        # Validate symbol parameter
        if not symbol or not symbol.strip():
            return "Error: 'symbol' parameter is required and cannot be empty."

        # Clamp depth to valid range
        depth = max(1, min(depth, self._MAX_TRAVERSE_DEPTH))

        # Resolve symbol to starting node
        if not (results := self._resolve_symbol(storage, symbol)):
            return f"Symbol '{symbol}' not found."

        if not (start_node := storage.get_node(results[0].node_id)):
            return f"Symbol '{symbol}' not found."

        # Traverse to find affected symbols
        affected_with_depth = storage.traverse_with_depth(
            start_node.id,
            depth,
            direction="callers",
        )
        if not affected_with_depth:
            return f"No upstream callers found for '{symbol}'."

        # Group results by depth
        by_depth = self._group_by_depth(affected_with_depth)
        total = len(affected_with_depth)
        label_display = start_node.label.value.title() if start_node.label else "Unknown"

        # Build output header
        lines = [f"Impact analysis for: {start_node.name} ({label_display})"]
        lines.append(f"Depth: {depth} | Total: {total} symbols")

        # Build confidence lookup for direct callers
        conf_lookup = self._build_confidence_lookup(storage, start_node.id)

        # Format results by depth
        counter = 1
        for d in sorted(by_depth.keys()):
            depth_label = _DEPTH_LABELS.get(d, "Transitive (review)")
            lines.append(f"\nDepth {d} — {depth_label}:")
            for node in by_depth[d]:
                label = node.label.value.title() if node.label else "Unknown"
                conf = conf_lookup.get(node.id)
                tag = f"  (confidence: {conf:.2f})" if conf is not None else ""
                lines.append(
                    f"  {counter}. {node.name} ({label}) -- "
                    f"{node.file_path}:{node.start_line}{tag}",
                )
                counter += 1

        lines.append("")
        lines.append("Tip: Review each affected symbol before making changes.")
        return "\n".join(lines)

    def handle_dead_code(self, storage: StorageBackend) -> str:
        """
        List all symbols marked as dead code.

        Delegates to :func:`~axon.mcp.resources.get_dead_code_list` for the
        shared query and formatting.

        Args:
            storage: The storage backend.

        Returns:
            Formatted list of dead code symbols grouped by file.
        """
        return get_dead_code_list(storage)

    def handle_detect_changes(self, storage: StorageBackend, diff: str) -> str:
        """
        Map git diff output to affected symbols.

        Parses the diff to find changed files and line ranges, then queries
        the storage backend to identify which symbols those lines belong to.

        Args:
            storage: The storage backend.
            diff: Raw git diff output string.

        Returns:
            Formatted list of affected symbols per changed file.
        """
        if not diff.strip():
            return "Empty diff provided."

        # Parse diff to extract changed files and line ranges
        if not (changed_files := self._parse_diff(diff)):
            return "Could not parse any changed files from the diff."

        # Query storage for each changed file with path validation
        return self._format_changed_files_with_validation(storage, changed_files)

    def handle_cypher(self, storage: StorageBackend, query: str) -> str:
        """
        Execute a raw Cypher query and return formatted results.

        Only read-only queries are allowed. Queries containing write keywords
        (DELETE, DROP, CREATE, SET, etc.) are rejected.

        Args:
            storage: The storage backend.
            query: The Cypher query string.

        Returns:
            Formatted query results, or an error message if execution fails.
        """
        # Strip comments so write keywords hidden inside comment blocks are detected.
        cleaned_query = sanitize_cypher(query)
        if WRITE_KEYWORDS.search(cleaned_query):
            return (
                "Query rejected: only read-only queries (MATCH/RETURN) are allowed. "
                "Write operations (DELETE, DROP, CREATE, SET, MERGE) are not permitted."
            )

        try:
            rows = storage.execute_raw(query)
        except SYSTEM_EXCEPTIONS as exc:
            logger.exception("Cypher query failed", exc_info=True)
            return f"Cypher query failed: {exc}"
        except Exception as exc:
            logger.exception("Unexpected error executing Cypher query", exc_info=True)
            return f"Cypher query failed: {exc}"

        if not rows:
            return "Query returned no results."

        # Format results
        lines = [f"Results ({len(rows)} rows):"]
        lines.append("")
        for i, row in enumerate(rows, 1):
            formatted_values = [str(v) for v in row]
            lines.append(f"  {i}. {' | '.join(formatted_values)}")

        return "\n".join(lines)

    def handle_coupling(
        self,
        storage: StorageBackend,
        file_path: str,
        min_strength: float = 0.3,
    ) -> str:
        """
        Query temporal coupling for a file and flag hidden dependencies.

        Temporal coupling occurs when files change together frequently.
        This tool identifies files that tend to be modified in parallel,
        highlighting both explicit imports and hidden dependencies.

        Args:
            storage: The storage backend.
            file_path: Path to the file to analyze.
            min_strength: Minimum coupling strength threshold (default 0.3).

        Returns:
            Formatted coupling analysis with hidden dependency warnings.
        """
        if not file_path or not file_path.strip():
            return "Error: 'file_path' parameter is required and cannot be empty."

        file_path = file_path.strip()
        if not _SAFE_PATH.match(file_path):
            return "Error: file path contains unsafe characters."

        escaped = _escape_cypher(file_path)
        rows = (
            storage.execute_raw(
                f"MATCH (a:File)-[r:COUPLED_WITH]-(b:File) "
                f"WHERE a.file_path = '{escaped}' "
                f"RETURN b.file_path, r.strength, r.co_changes "
                f"ORDER BY r.strength DESC",
            )
            or []
        )

        if not (rows := [r for r in rows if (r[1] or 0) >= min_strength]):
            return f"No temporal coupling found for '{file_path}' (min strength: {min_strength})."

        # Get explicit imports for comparison
        import_rows = (
            storage.execute_raw(
                f"MATCH (a:File)-[r:CodeRelation]->(b:File) "
                f"WHERE a.file_path = '{escaped}' AND r.rel_type = 'imports' "
                f"RETURN b.file_path",
            )
            or []
        )
        imported_files = {r[0] for r in import_rows}

        lines = [f"Temporal coupling for: {file_path}"]
        lines.append("=" * 48)
        lines.append("")

        for i, row in enumerate(rows, 1):
            coupled_path = row[0] or "?"
            strength = row[1] or 0.0
            co_changes = row[2] or 0
            has_import = coupled_path in imported_files
            import_flag = "imports: yes" if has_import else "imports: no ⚠️"
            lines.append(
                f"  {i}. {coupled_path}  strength: {strength:.2f}  "
                f"co_changes: {co_changes}  ({import_flag})",
            )

        lines.append("")
        if hidden := [r[0] for r in rows if r[0] not in imported_files]:
            lines.append(
                f"⚠️ {len(hidden)} file(s) have hidden dependencies (no static import).",
            )
        return "\n".join(lines)

    def handle_call_path(
        self,
        storage: StorageBackend,
        from_symbol: str,
        to_symbol: str,
        max_depth: int = 10,
    ) -> str:
        """
        Find the shortest call chain between two symbols via BFS.

        Uses breadth-first search (BFS) to find the shortest path of function calls
        from a source symbol to a target symbol. The algorithm traverses through
        CALLED_BY edges in the graph, exploring each level of call depth before
        moving to the next, ensuring the first found path is the shortest.

        The function performs the following steps:
            1. Validate input parameters (non-empty symbol names)
            2. Clamp max_depth to prevent runaway queries (1-10 range)
            3. Resolve both symbol names to graph nodes
            4. Check if source and target are the same (trivial case)
            5. Execute BFS to find shortest call path
            6. Reconstruct path from parent map if found
            7. Format and return the result

        Args:
            storage: The storage backend instance used for graph queries.
                Must implement get_node, fts_search (or exact_name_search),
                get_callees, and get_node methods.
            from_symbol: The starting symbol name to search from. Can be a
                function, method, class, or any callable symbol in the graph.
            to_symbol: The target symbol name to find a path to. Must be
                reachable from the source through call relationships.
            max_depth: Maximum traversal depth for BFS search. Defaults to 10.
                Values are clamped to the range [1, MAX_TRAVERSE_DEPTH] to
                prevent excessive queries. Higher values find longer paths
                but take more time.

        Returns:
            Formatted string containing:
                - Header with call path visualization (symbol names joined by arrows)
                - Hop count (number of edges in the path)
                - Detailed list of each hop with symbol name, type label,
                  file path, and line number

            Or an error message if:
                - Either symbol parameter is empty
                - Either symbol is not found in the graph
                - Source and target resolve to the same symbol
                - No path exists within the specified max_depth

        Raises:
            No explicit exceptions are raised. All database operations are wrapped
            in error handling that returns descriptive error messages instead.

        Note:
            - The BFS algorithm finds the shortest path by number of hops, not
              by execution time or importance
            - If multiple shortest paths exist, which one is returned is
              non-deterministic due to set iteration order
            - The function only traverses CALLED_BY edges (callees), not callers
            - Self-loops (symbol calling itself) are handled by the same-symbol check
        """
        # Step 1: Validate input parameters - both symbols must be non-empty
        # This prevents unnecessary database queries for invalid input
        if validation_error := self._validate_call_path_inputs(from_symbol, to_symbol):
            return validation_error

        # Step 2: Clamp max_depth to valid range [1, MAX_TRAVERSE_DEPTH]
        # This prevents runaway queries that could exhaust resources
        max_depth = max(1, min(max_depth, self._MAX_TRAVERSE_DEPTH))

        # Step 3: Resolve both symbols to graph nodes
        # Returns error if either symbol cannot be found in the graph
        res_result = self._resolve_symbols_for_call_path(storage, from_symbol, to_symbol)
        if isinstance(res_result, str):
            # Error occurred during resolution (returned as error string)
            return res_result

        src_node, tgt_node = res_result

        # Step 4: Check if source and target are the same symbol
        # This is a trivial case - no traversal needed
        if src_node.id == tgt_node.id:
            return f"Source and target are the same symbol: {src_node.name}"

        # Step 5: Execute BFS to find shortest call path
        # BFS guarantees the shortest path by number of hops
        bfs_result = self._find_shortest_call_path_bfs(storage, src_node.id, tgt_node.id, max_depth)

        if bfs_result is None:
            # No path found within the specified depth limit
            return (
                f"No call path found from '{src_node.name}' to '{tgt_node.name}' "
                f"within {max_depth} hops."
            )

        # Step 6: Reconstruct the path from parent map
        path_ids = self._reconstruct_path_from_bfs(bfs_result, tgt_node.id)

        # Step 7: Format and return the result
        return self._format_call_path_result(storage, path_ids, src_node.name, tgt_node.name)

    def handle_explain(self, storage: StorageBackend, symbol: str) -> str:
        """
        Produce a narrative explanation of a symbol.

        Provides a comprehensive overview including role (entry point, exported, dead),
        location, signature, community membership, callers, callees, and process flows.

        The function retrieves symbol information from the knowledge graph and formats
        it into a human-readable narrative. It performs the following steps:
            1. Validate the symbol parameter is non-empty
            2. Resolve the symbol name to a graph node
            3. Extract and format basic information (name, type, location, signature)
            4. Query and format role information (entry point, exported, dead code)
            5. Query and format community membership
            6. Query and format caller/callee relationships
            7. Query and format process flow information

        Args:
            storage: The storage backend instance used for graph queries.
                Must implement get_node, fts_search (or exact_name_search),
                get_callers_with_confidence, get_callees_with_confidence, and
                execute_raw methods.
            symbol: The symbol name to explain. Can be a function, class, method,
                variable, or any other code symbol stored in the knowledge graph.

        Returns:
            Formatted narrative explanation of the symbol containing:
                - Header with symbol name and type label
                - Role information (entry point, exported, dead code status)
                - Location (file path and line range)
                - Signature (if available)
                - Community membership (if part of a detected community)
                - Callers (functions/methods that call this symbol)
                - Callees (functions/methods called by this symbol)
                - Process flows (execution paths this symbol participates in)

        Raises:
            No explicit exceptions are raised. All database operations are wrapped
            in error handling that returns descriptive error messages instead.
            Invalid input parameters return error messages rather than raising
            exceptions.

        Note:
            - If the symbol is not found, returns an error message indicating this
            - Confidence scores are included for caller/callee relationships when available
            - The function gracefully handles missing backend capabilities
            - Callers and callees are limited to 5 displayed with "+N more" suffix
        """
        # Step 1: Validate symbol parameter - ensure it's non-empty
        # This prevents unnecessary database queries for invalid input
        if validation_error := self._validate_symbol_input(symbol):
            return validation_error

        # Step 2: Resolve symbol to a node in the graph
        # Uses full-text search to find the symbol; returns error if not found
        if not (node := self._resolve_symbol_for_explain(storage, symbol)):
            return f"Symbol '{symbol}' not found."

        # Step 3: Build the explanation output by collecting all sections
        # Start with the header (name and type label)
        lines = self._format_explanation_header(node)

        # Step 4: Extract and format role information
        # Roles indicate special status: entry point, exported, or dead code
        lines.extend(self._extract_and_format_roles(node))

        # Step 5: Format location information
        # Includes file path, line range, and signature if available
        lines.extend(self._format_location(node))

        # Step 6: Query and format community membership
        # Community represents a group of densely connected symbols
        lines.extend(self._query_and_format_community(storage, node))

        # Step 7: Query and format callers/callees relationships
        # These show the call graph connections to/from this symbol
        lines.extend(self._format_callers_callees(storage, node))

        # Step 8: Query and format process flows
        # Process flows show execution paths this symbol participates in
        lines.extend(self._query_and_format_processes(storage, node))

        return "\n".join(lines)

    def handle_review_risk(self, storage: StorageBackend, diff: str) -> str:
        """
        Assess PR risk by synthesizing multiple graph signals.

        Analyzes a git diff to assess the risk level of a pull request by considering:
            - Entry points hit: Functions that serve as application entry points
            - Total downstream dependents: Number of symbols that depend on changed symbols
            - Missing co-change files: Files that typically change together (temporal coupling)
            - Community boundary crossings: Number of distinct code communities affected

        The risk assessment follows these steps:
            1. Validate the diff parameter is non-empty
            2. Parse the diff to extract changed files and line ranges
            3. Query the knowledge graph for symbols in changed files
            4. Filter symbols that overlap with changed line ranges
            5. For each affected symbol, count downstream dependents and check if entry point
            6. Query temporal coupling to find missing co-change files
            7. Query community membership to detect boundary crossings
            8. Calculate risk score based on all collected metrics
            9. Format and return the risk assessment

        Args:
            storage: The storage backend instance used for graph queries.
                Must implement execute_raw, get_node, and traverse_with_depth methods.
            diff: Raw git diff output string containing file changes to analyze.

        Returns:
            Formatted PR risk assessment string containing:
                - Risk level (LOW/MEDIUM/HIGH) with numeric score (0-10)
                - List of affected symbols with their downstream dependent counts
                - Missing co-change files with coupling strength
                - Community boundary crossing information (if applicable)

        Raises:
            No explicit exceptions are raised. All database operations are wrapped
            in error handling that returns descriptive error messages instead.
            Invalid input parameters return error messages rather than raising
            exceptions.

        Note:
            - Risk score is calculated as: entry_points + missing_cochanges + (dependents // 10)
              plus 2 points if multiple communities are affected
            - Score is capped at 10 (maximum risk)
            - Temporal coupling threshold is 0.5 (files with strength >= 0.5 are considered)
            - Only symbols with start_line > 0 are considered (excludes file-level nodes)
        """
        # Step 1: Validate that diff is not empty
        # Prevents unnecessary processing of empty input
        if not diff.strip():
            return "Empty diff provided."

        # Step 2: Parse the diff to extract changed files and line ranges
        # Returns dict mapping file_path -> list of (start_line, end_line) tuples
        if not (changed_files := self._parse_diff(diff)):
            return "Could not parse any changed files from the diff."

        # Convert to set for efficient O(1) lookup when checking co-changes
        changed_file_set = set(changed_files.keys())

        # Step 3 & 4: Find affected symbols in changed files
        # Returns list of (name, label_prefix, file_path, dependent_count) tuples
        all_affected_symbols = self._find_affected_symbols(storage, changed_files)

        # Aggregate metrics across all affected symbols
        # entry_points_hit: count of symbols marked as entry points
        # total_dependents: sum of all downstream dependent counts
        entry_points_hit = sum(1 for _, _, _, deps in all_affected_symbols if deps > 0)
        total_dependents = sum(deps for _, _, _, deps in all_affected_symbols)

        # Step 5: Find missing co-change files based on temporal coupling
        # These are files that typically change together with modified files
        missing_cochange = self._find_missing_cochanges(storage, changed_files, changed_file_set)

        # Step 6: Find which communities are touched by the changes
        # A set of community names that have at least one affected symbol
        communities_touched = self._find_communities_touched(storage, all_affected_symbols)

        # Step 7: Calculate risk score based on collected metrics
        # Formula balances entry points, coupling gaps, and dependent count
        score = self._calculate_risk_score(
            entry_points_hit=entry_points_hit,
            missing_cochange_count=len(missing_cochange),
            total_dependents=total_dependents,
            communities_touched_count=len(communities_touched),
        )

        # Step 8: Determine risk level based on score thresholds
        level = self._determine_risk_level(score)

        # Step 9: Format and return the complete risk assessment
        return self._format_risk_assessment(
            level=level,
            score=score,
            all_affected_symbols=all_affected_symbols,
            missing_cochange=missing_cochange,
            communities_touched=communities_touched,
        )

    def handle_file_context(self, storage: StorageBackend, file_path: str) -> str:
        """
        Provide comprehensive context for a single file.

        This function retrieves and formats comprehensive context information about a given file,
        including its symbols, imports, importers (files that import it), temporal coupling
        relationships, dead code, and community membership. The information is formatted into
        a human-readable string suitable for MCP responses.

        The context retrieval follows these steps:
            1. Validate the file_path parameter is non-empty and contains safe characters
            2. Escape the file path for safe inclusion in Cypher queries
            3. Query each context category (symbols, imports, importers, coupling, dead code, communities)
            4. Combine all sections into a unified output with the file header

        Args:
            storage: The storage backend instance used for graph queries.
                Must implement execute_raw method for Cypher queries.
            file_path: The path to the file to retrieve context for. Must be a valid
                file path that has been indexed in the knowledge graph.

        Returns:
            Formatted string containing:
                - File header with path
                - Symbols section (functions, classes, etc. defined in the file)
                - Imports section (files this file imports)
                - Imported by section (files that import this file)
                - Coupled files section (files that typically change together)
                - Dead code section (unreachable symbols in the file)
                - Communities section (code communities this file belongs to)

            Or an error message if:
                - The file_path parameter is empty or contains only whitespace
                - The file_path contains unsafe characters (potential injection)
                - No data is found for the file (not indexed)

        Raises:
            No explicit exceptions are raised. All database operations are wrapped
            in error handling that returns descriptive error messages instead.
            Invalid input parameters return error messages rather than raising exceptions.

        Note:
            - If the file is not found in the index, returns an error message indicating this
            - Only symbols with start_line > 0 are included (excludes file-level nodes)
            - Coupling relationships are limited to top 5 by strength
            - Community membership shows symbol counts per community
        """
        # Step 1: Validate file_path parameter - ensure it's non-empty and safe
        # This prevents unnecessary database queries for invalid input
        if validation_error := self._validate_file_path_input(file_path):
            return validation_error

        # Clean and escape the file path for safe Cypher query usage
        file_path = file_path.strip()
        escaped = _escape_cypher(file_path)

        # Step 2: Query all context data from the storage backend
        # Each query fetches a different aspect of file context
        sym_rows = self._query_file_symbols(storage, escaped)
        imports_out = self._query_file_imports(storage, escaped)
        imports_in = self._query_file_importers(storage, escaped)
        coupling_rows = self._query_file_coupling(storage, escaped)
        dead_rows = self._query_file_dead_code(storage, escaped)
        comm_rows = self._query_file_communities(storage, escaped)

        # Step 3: Check if any data was found for this file
        # Return early with informative message if the file hasn't been indexed
        if not sym_rows and not imports_out and not imports_in:
            return f"No data found for file '{file_path}'. Is it indexed?"

        # Step 4: Build the context output by collecting all sections
        # Start with the file header
        lines = self._format_file_context_header(file_path)

        # Add each context section if data is available
        # Symbols: functions, classes, and other definitions in the file
        lines.extend(self._format_file_symbols_section(sym_rows))

        # Imports: outgoing (files this file imports) and incoming (files that import this file)
        lines.extend(self._format_file_imports_section(imports_out, imports_in))

        # Coupling: files that typically change together with this file
        lines.extend(self._format_file_coupling_section(coupling_rows))

        # Dead code: unreachable symbols in this file
        lines.extend(self._format_file_dead_code_section(dead_rows))

        # Communities: code communities this file belongs to
        lines.extend(self._format_file_communities_section(comm_rows))

        return "\n".join(lines)

    def handle_cycles(self, storage: StorageBackend, min_size: int = 2) -> str:
        """
        Detect circular dependencies using strongly connected components.

        Uses graph analysis to find strongly connected components (SCCs)
        which indicate circular dependencies between symbols.

        Args:
            storage: The storage backend.
            min_size: Minimum cycle size to report (default 2).

        Returns:
            Formatted list of detected circular dependencies.
        """
        min_size = max(2, min_size)

        try:
            graph = storage.load_graph()
        except SYSTEM_EXCEPTIONS as exc:
            return f"Error loading graph: {exc}"

        try:
            ig_graph, index_to_node_id = Community(graph).export_to_igraph()
        except ImportError:
            return "Error: export_to_igraph not available. Install igraph dependency."

        if ig_graph.vcount() == 0:
            return "No symbols in the graph to analyze."

        sccs = ig_graph.connected_components(mode="strong")

        cycles = [list(component) for component in sccs if len(component) >= min_size]
        cycles = cast(list[list[int]], cycles)
        if not cycles:
            return "No circular dependencies detected."

        cycles.sort(key=len, reverse=True)

        lines = [f"Circular Dependencies ({len(cycles)} groups)"]
        lines.append("=" * 48)

        for i, component in enumerate(cycles, 1):
            node_ids = [index_to_node_id[idx] for idx in component if idx in index_to_node_id]
            nodes = [graph.get_node(nid) for nid in node_ids]
            nodes = [n for n in nodes if n is not None]

            severity = "CRITICAL" if len(nodes) >= 5 else ""
            size_label = f" — {severity}" if severity else ""
            lines.append(f"\nCycle {i} ({len(nodes)} symbols){size_label}:")
            for node in nodes:
                label = node.label.value.title() if node.label else "Unknown"
                lines.append(f"  - {node.name} ({label}) — {node.file_path}:{node.start_line}")

        return "\n".join(lines)

    def handle_test_impact(
        self,
        storage: StorageBackend,
        diff: str = "",
        symbols: list[str] | None = None,
    ) -> str:
        """
        Find tests likely affected by code changes.

        Analyzes changed symbols (from a git diff or provided symbol list) and finds
        test files that call them through the call graph. This helps identify which
        tests need to be run after making changes to the codebase.

        The function performs the following steps:
            1. Validate that at least one input source (diff or symbols) is provided
            2. Extract changed symbols from diff (if provided) or resolve from symbol list
            3. Traverse the call graph to find test files that call the changed symbols
            4. Categorize test hits by depth (direct: depth <= 2, transitive: depth > 2)
            5. Format and return the test impact analysis

        Args:
            storage: The storage backend instance used for graph queries.
                Must implement execute_raw, get_node, resolve_symbol (via _resolve_symbol),
                and traverse_with_depth methods.
            diff: Raw git diff output string containing file changes to analyze.
                When provided, the function parses the diff to find changed files and
                line ranges, then queries the knowledge graph for symbols in those ranges.
            symbols: Optional list of symbol names to analyze directly.
                When provided, these symbols are resolved directly without parsing a diff.
                Use this when you know specific symbols that have changed.

        Returns:
            Formatted test impact analysis string containing:
                - Header with "Test Impact Analysis" title
                - Count of changed symbols analyzed
                - Direct test coverage: test files that directly call changed symbols
                - Transitive test coverage: test files that call through intermediate symbols

            Or an error message if:
                - Neither diff nor symbols parameter is provided
                - No changed symbols are found in the diff or resolution
                - No test files are found in the call graph of changed symbols

        Raises:
            No explicit exceptions are raised. All database operations are wrapped
            in error handling that returns descriptive error messages instead.
            Invalid input parameters return error messages rather than raising exceptions.

        Note:
            - If both diff and symbols are provided, diff takes precedence
            - Test file detection uses _is_test_file which checks for 'test' in path
            - Depth threshold for direct vs transitive is 2 (depth <= 2 is direct)
            - Call graph traversal uses a maximum depth of 4 to limit query scope
        """
        # Step 1: Validate inputs - at least one input source must be provided
        # This prevents unnecessary processing when no valid input is given
        if not diff and not symbols:
            return "Error: provide either 'diff' or 'symbols' parameter."

        # Step 2: Extract or resolve changed symbols from the provided input
        # Changed symbols are stored as tuples of (node_id, symbol_name)
        if not (changed_symbol_ids := self._extract_changed_symbols(storage, diff, symbols)):
            # If none found, return early with informative message
            return "No changed symbols found."

        # Step 3: Find test files in the call graph of changed symbols
        # Traverses up to depth 4 to find all callers that are test files
        test_hits = self._find_test_files_in_call_graph(storage, changed_symbol_ids)

        # Handle case where no test files were found in the call graph
        if not test_hits:
            return (
                f"No test files found in the call graph of {len(changed_symbol_ids)} "
                f"changed symbol(s). Tests may not directly call these symbols."
            )

        # Step 4: Categorize test hits by depth (direct vs transitive)
        # Direct tests (depth <= 2) are tests that closely exercise the changed code
        # Transitive tests (depth > 2) go through intermediate symbols
        direct_files, transitive_files = self._categorize_test_hits_by_depth(test_hits)

        # Step 5: Format and return the complete test impact analysis
        return self._format_test_impact_output(
            changed_symbol_count=len(changed_symbol_ids),
            test_hits=test_hits,
            direct_files=direct_files,
            transitive_files=transitive_files,
        )

    def handle_communities(self, storage: StorageBackend, community: str | None = None) -> str:
        """
        List communities or drill into a specific one.

        Communities are groups of symbols that are densely connected in the graph,
        representing cohesive modules or packages. This tool provides two modes:
            - List all detected communities with cohesion scores and member counts
            - Drill into a specific community to see its member symbols

        The function routes to different handlers based on whether a community name
        is provided:
            - With community name: queries and formats members of that specific community
            - Without community name: queries all communities with stats and cross-community processes

        Args:
            storage: The storage backend instance used for graph queries.
                Must implement execute_raw method for Cypher queries.
            community: Optional specific community name to drill into.
                When provided, shows all symbols that are members of this community.
                When None, shows all communities with summary statistics.

        Returns:
            Formatted string containing:
                - If community specified: list of member symbols with their types,
                  file locations, and tags (entry point, exported)
                - If no community: list of all communities sorted by cohesion,
                  plus any cross-community processes

        Raises:
            No explicit exceptions are raised. Database errors are caught and
            returned as error messages. Invalid input returns informative errors
            rather than raising exceptions.

        Note:
            - Community detection must be enabled during indexing to populate data
            - Cross-community processes are symbols that belong to multiple communities
            - Cohesion score ranges from 0.0 to 1.0, higher values indicate denser connections
        """
        # Route to appropriate handler based on whether a specific community is requested
        if community:
            # Handle specific community: query and format its members
            return self._handle_specific_community(storage, community)
        else:
            # Handle all communities: list all with stats and cross-community processes
            return self._handle_all_communities(storage)
