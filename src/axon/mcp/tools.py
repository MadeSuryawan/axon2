"""
MCP tool handler implementations for Axon.

This module provides the core functionality for handling MCP (Model Context Protocol)
tool requests. Each tool handler accepts a storage backend and tool-specific arguments,
performs the appropriate query, and returns a human-readable string suitable for
inclusion in an MCP ``TextContent`` response.

The main entry point is the :class:`Tools` class, which encapsulates all tool
handlers and helper methods.
"""

from json import JSONDecodeError, loads
from logging import getLogger
from pathlib import Path
from re import IGNORECASE, MULTILINE
from re import compile as re_compile
from typing import Any

from axon.config.constants import MODEL_NAME, SYSTEM_EXCEPTIONS
from axon.core.embeddings.embedder import get_model
from axon.core.graph.model import GraphNode
from axon.core.search.hybrid import SearchDeps, hybrid_search
from axon.core.storage.base import StorageBackend
from axon.mcp.resources import get_dead_code_list

logger = getLogger(__name__)

# Maximum depth for impact analysis traversal (prevents runaway queries)
MAX_TRAVERSE_DEPTH = 10


# Regex patterns for detecting write operations in Cypher queries
_WRITE_KEYWORDS = re_compile(
    r"\b(DELETE|DROP|CREATE|SET|REMOVE|MERGE|DETACH|INSTALL|LOAD|COPY|CALL)\b",
    IGNORECASE,
)

# Regex patterns for parsing git diff output
_DIFF_FILE_PATTERN = re_compile(r"^diff --git a/(.+?) b/(.+?)$", MULTILINE)
_DIFF_HUNK_PATTERN = re_compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", MULTILINE)

# Human-readable labels for impact analysis depth levels
_DEPTH_LABELS: dict[int, str] = {
    1: "Direct callers (will break)",
    2: "Indirect (may break)",
}


class Tools:
    """
    Handles MCP tool requests for the Axon knowledge graph.

    This class encapsulates all tool handlers and helper methods, providing
    a clean interface for processing MCP tool requests. Each method handles
    a specific tool type (query, context, impact, etc.) and returns formatted
    results suitable for MCP responses.
    """

    # Class-level constants for maximum flow size limits
    _MAX_TRAVERSE_DEPTH: int = MAX_TRAVERSE_DEPTH
    _EMBED_MODEL_NAME: str = MODEL_NAME
    _SNIPPET_MAX_LENGTH: int = 200

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
            limit: Maximum number of results (default 20).

        Returns:
            Formatted search results grouped by process, with file, name, label,
            and snippet for each result.
        """
        # Try to get query embedding for hybrid search, fall back to FTS-only
        query_embedding: list[float] | None = None
        try:
            model = get_model()
            query_embedding = list(next(iter(model.embed([query]))))
        except SYSTEM_EXCEPTIONS:
            logger.debug("Query embedding failed, falling back to FTS only", exc_info=True)

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
        return self.format_query_results(results, groups)

    def handle_context(self, storage: StorageBackend, symbol: str) -> str:
        """
        Provide a 360-degree view of a symbol.

        Looks up the symbol by name via full-text search, then retrieves its
        callers, callees, and type references.

        Args:
            storage: The storage backend.
            symbol: The symbol name to look up.

        Returns:
            Formatted view including callers, callees, type refs, and guidance.
        """
        # Resolve symbol to a node
        results = self._resolve_symbol(storage, symbol)
        if not results:
            return f"Symbol '{symbol}' not found."

        node = storage.get_node(results[0].node_id)
        if not node:
            return f"Symbol '{symbol}' not found."

        # Build the context output
        lines = self._format_node_header(node)

        # Add callers information
        callers_raw = self._get_callers_with_fallback(storage, node.id)
        if callers_raw:
            lines.append(f"\nCallers ({len(callers_raw)}):")
            for c, conf in callers_raw:
                tag = self.confidence_tag(conf)
                lines.append(f"  -> {c.name}  {c.file_path}:{c.start_line}{tag}")

        # Add callees information
        callees_raw = self._get_callees_with_fallback(storage, node.id)
        if callees_raw:
            lines.append(f"\nCallees ({len(callees_raw)}):")
            for c, conf in callees_raw:
                tag = self.confidence_tag(conf)
                lines.append(f"  -> {c.name}  {c.file_path}:{c.start_line}{tag}")

        # Add type references
        type_refs = storage.get_type_refs(node.id)
        if type_refs:
            lines.append(f"\nType references ({len(type_refs)}):")
            for t in type_refs:
                lines.append(f"  -> {t.name}  {t.file_path}")

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
        # Clamp depth to valid range
        depth = max(1, min(depth, self._MAX_TRAVERSE_DEPTH))

        # Resolve symbol to starting node
        results = self._resolve_symbol(storage, symbol)
        if not results:
            return f"Symbol '{symbol}' not found."

        start_node = storage.get_node(results[0].node_id)
        if not start_node:
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
        changed_files = self._parse_diff(diff)

        if not changed_files:
            return "Could not parse any changed files from the diff."

        # Query storage for each changed file and format results
        return self._format_changed_files(storage, changed_files)

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
        # Reject write operations for safety
        if _WRITE_KEYWORDS.search(query):
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

    # -------------------------------------------------------------------------
    # Public static helper methods (for testing and external use)
    # -------------------------------------------------------------------------

    @staticmethod
    def _escape_cypher(value: str) -> str:
        """
        Escape a string for safe inclusion in a Cypher string literal.

        Args:
            value: The string to escape.

        Returns:
            The escaped string safe for Cypher queries.
        """
        return value.replace("\\", "\\\\").replace("'", "\\'")

    @staticmethod
    def confidence_tag(confidence: float) -> str:
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
        if hasattr(storage, "exact_name_search"):
            results = storage.exact_name_search(symbol, limit=1)
            if results:
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
            pname = node_to_process.get(r.node_id)
            if pname:
                groups.setdefault(pname, []).append(r)

        return groups

    @staticmethod
    def format_query_results(results: list, groups: dict[str, list]) -> str:
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
                    snippet = r.snippet[: Tools._SNIPPET_MAX_LENGTH].replace("\n", " ").strip()
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
                    snippet = r.snippet[: Tools._SNIPPET_MAX_LENGTH].replace("\n", " ").strip()
                    lines.append(f"   {snippet}")
                counter += 1
            lines.append("")

        lines.append("Next: Use context() on a specific symbol for the full picture.")
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Private helper methods
    # -------------------------------------------------------------------------

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

    def _format_changed_files(
        self,
        storage: StorageBackend,
        changed_files: dict[str, list[tuple[int, int]]],
    ) -> str:
        """
        Query storage for symbols in changed files and format results.

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
                f"MATCH (n) WHERE n.file_path = '{self._escape_cypher(file_path)}' "
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
