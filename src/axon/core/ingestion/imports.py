"""
Phase 4: Import resolution for Axon.

Takes the FileParseData produced by the parsing phase and resolves import
statements to actual File nodes in the knowledge graph, creating IMPORTS
relationships between the importing file and the target file.

The main entry point is :meth:`Imports.process_imports`, which reads parse
data for multiple files, resolves their imports to actual project files,
and creates IMPORTS relationships in the knowledge graph.
"""

from concurrent.futures.thread import ThreadPoolExecutor
from logging import getLogger
from os import cpu_count
from pathlib import PurePosixPath

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)
from axon.core.ingestion.parser_phase import FileParseData
from axon.core.ingestion.resolved import ResolvedEdge
from axon.core.parsers.base import ImportInfo

logger = getLogger(__name__)


class Imports:
    """
    Handles import resolution for the Axon knowledge graph.

    Scans parsed import statements from source files and resolves them to
    actual files in the project codebase, creating ``IMPORTS`` relationships
    in the knowledge graph. Supports Python (relative/absolute), TypeScript,
    and JavaScript import styles.

    Maps language-specific import semantics to file system resolution:
    - **Python**: Handles relative imports (dots-based) and absolute
      dotted module paths.
    - **TypeScript/JavaScript**: Resolves relative path imports (``./``,
      ``../``) against the importing file's directory. Bare specifiers
      (e.g., ``express`` or ``@types/node``) are treated as external.
    """

    # Tuple of file extensions used by JavaScript/TypeScript
    _JS_TS_EXTENSIONS: tuple[str, ...] = (".ts", ".js", ".tsx", ".jsx")

    def __init__(
        self,
        graph: KnowledgeGraph,
        parse_data: list[FileParseData],
    ) -> None:
        """
        Initialize the Imports analyzer.

        Args:
            graph: The knowledge graph to analyze and enrich.
            parse_data: The parse data containing imports for each file.
            file_index: A mapping of file paths to graph node IDs for all File nodes.
        """
        self._graph = graph
        self._parse_data = parse_data
        self._file_index: dict[str, str] = self.build_file_index()
        self._workers: int = min(cpu_count() or 4, 8, len(self._parse_data))

    def process_imports(
        self,
        *,
        parallel: bool = False,
        collect: bool = False,
    ) -> list[ResolvedEdge] | None:
        """
        Resolve imports and create IMPORTS relationships in the graph.

        Main public entry point. For each file's parsed imports, resolves
        the target file and creates an ``IMPORTS`` relationship from the
        importing file node to the target file node. Duplicate edges (same
        source -> same target) are automatically skipped.

        Algorithm:
        1. Build index of all project files from the graph.
        2. Iterate through parsed data for each source file.
        3. For each import, resolve its target file.
        4. Create IMPORTS relationship if target is in project.

        """
        source_roots = self._detect_source_roots()

        if parallel:
            with ThreadPoolExecutor(max_workers=self._workers) as pool:
                all_edges = list(
                    pool.map(
                        lambda fpd: self.resolve_file_imports(fpd, source_roots),
                        self._parse_data,
                    ),
                )
        else:
            all_edges = [self.resolve_file_imports(fpd, source_roots) for fpd in self._parse_data]

        if collect:
            return [edge for file_edges in all_edges for edge in file_edges]

        self._write_import_edges(all_edges)
        logger.info("Imports resolved")

    def resolve_file_imports(
        self,
        fpd: FileParseData,
        source_roots: set[str],
    ) -> list[ResolvedEdge]:
        """
        Resolve imports for a single file — pure read, no graph mutation.

        Returns one :class:`ResolvedEdge` per unique ``(source, target)`` pair
        with per-file merged symbols.  Cross-file symbol merging happens in the
        caller.
        """
        source_file_id = generate_id(NodeLabel.FILE, fpd.file_path)
        pair_symbols: dict[str, set[str]] = {}

        for imp in fpd.parse_result.imports:
            target_id = self._resolve_import_path(fpd.file_path, imp, source_roots)
            if target_id is None:
                continue
            if target_id not in pair_symbols:
                pair_symbols[target_id] = set()
            pair_symbols[target_id].update(imp.names)

        return self._pairing_symbols(pair_symbols, source_file_id)

    def _pairing_symbols(
        self,
        pair_symbols: dict[str, set[str]],
        source_file_id: str,
    ) -> list[ResolvedEdge]:
        edges: list[ResolvedEdge] = []
        for target_id, symbols in pair_symbols.items():
            rel_id = f"imports:{source_file_id}->{target_id}"
            edges.append(
                ResolvedEdge(
                    rel_id=rel_id,
                    rel_type=RelType.IMPORTS,
                    source=source_file_id,
                    target=target_id,
                    properties={"symbols": symbols},
                ),
            )
        return edges

    def build_file_index(self) -> dict[str, str]:
        """
        Build an index mapping file paths to their graph node IDs.

        Iterates over all FILE nodes in the graph and creates a fast lookup
        dict for resolving import targets.

        Returns:
            A dict mapping file paths to their node IDs.
            Example: ``{"src/auth/validate.py": "file:src/auth/validate.py:"}``
        """
        file_nodes = self._graph.get_nodes_by_label(NodeLabel.FILE)
        return {node.file_path: node.id for node in file_nodes}

    def _write_import_edges(self, all_edges: list[list[ResolvedEdge]]) -> None:
        """Merge cross-file symbol sets and write IMPORTS edges to the graph."""
        merged: dict[str, tuple[str, str, set[str]]] = {}
        for file_edges in all_edges:
            for edge in file_edges:
                if (rel_id := edge.rel_id) in merged:
                    merged[rel_id][2].update(edge.properties["symbols"])
                    continue
                merged[rel_id] = (
                    edge.source,
                    edge.target,
                    set(edge.properties["symbols"]),
                )
        self._merged_relationship(merged)

    def _merged_relationship(self, merged: dict[str, tuple[str, str, set[str]]]) -> None:
        for rel_id, (source, target, symbols) in merged.items():
            self._graph.add_relationship(
                GraphRelationship(
                    id=rel_id,
                    type=RelType.IMPORTS,
                    source=source,
                    target=target,
                    properties={"symbols": ",".join(sorted(symbols))},
                ),
            )

    def _resolve_import_path(
        self,
        importing_file: str,
        import_info: ImportInfo,
        source_roots: set[str] | None = None,
    ) -> str | None:
        """
        Resolve an import statement to the target file's node ID.

        Uses the importing file's path, the parsed import information, and
        the index of all known project files to determine which file is
        being imported. Returns ``None`` for external/unresolvable imports.

        Strategy:
        1. Detect the language from the importing file's extension.
        2. Route to language-specific resolver (Python, TS/JS, or other).
        3. Return the resolved file node ID or None if external.

        Args:
            importing_file: Relative path of the file containing the import.
            import_info: The parsed import information.
            source_roots: Python source root directories (e.g. src/) from the file index.

        Returns:
            The node ID of the resolved target file, or ``None`` if the
            import cannot be resolved to a file in the project.
        """
        language = self._detect_language(importing_file)

        if language == "python":
            return self._resolve_python(importing_file, import_info, source_roots)
        if language in ("typescript", "javascript"):
            return self._resolve_js_ts(importing_file, import_info)

    def _detect_source_roots(self) -> set[str]:
        """
        Detect Python source root directories (e.g. ``src/``) from the file index.

        A source root is a directory whose children have ``__init__.py`` but the
        directory itself does not, indicating a ``src/`` layout.
        """
        if not self._file_index:
            return set()

        init_dirs: set[str] = set()
        for path in self._file_index:
            if not path.endswith("/__init__.py"):
                continue
            init_dirs.add(str(PurePosixPath(path).parent))

        roots: set[str] = set()
        for d in init_dirs:
            parent = str(PurePosixPath(d).parent)
            if parent != "." and parent not in init_dirs:
                roots.add(parent)
        return roots

    @staticmethod
    def _detect_language(file_path: str) -> str:
        """
        Infer the programming language from a file's extension.

        Args:
            file_path: The file path to inspect.

        Returns:
            One of: "python", "typescript", "javascript", or "" (unknown).
        """
        suffix = PurePosixPath(file_path).suffix.lower()
        if suffix == ".py":
            return "python"
        if suffix in (".ts", ".tsx"):
            return "typescript"
        if suffix in (".js", ".jsx"):
            return "javascript"
        return ""

    def _resolve_python(
        self,
        importing_file: str,
        import_info: ImportInfo,
        source_roots: set[str] | None = None,
    ) -> str | None:
        """
        Resolve a Python import to a file node ID.

        Handles both relative and absolute import styles:
        - **Relative** (``is_relative=True``): Dot-prefixed module paths.
          Resolved relative to the importing file's directory.
        - **Absolute**: Dotted module paths from project root (e.g.,
          ``mypackage.auth.utils```).

        Returns ``None`` for external packages not present in the project.

        Args:
            importing_file: Path of the file containing the import.
            import_info: The parsed import information.
            source_roots: Python source root directories (e.g. src/) from the file index.

        Returns:
            The node ID of the target file, or ``None`` if external.
        """
        if import_info.is_relative:
            return self._resolve_python_relative(importing_file, import_info)
        return self._resolve_python_absolute(import_info, source_roots)

    def _resolve_python_relative(self, importing_file: str, import_info: ImportInfo) -> str | None:
        """
        Resolve a relative Python import (``from .foo import bar``).

        The leading dots determine the relative position:
        - 1 dot (``.foo``): Same package as importing file
        - 2 dots (``..foo``): Parent package
        - N dots: N-1 levels up from importing file's parent

        Examples:
            In ``src/auth/validate.py``:
            - ``from .utils`` → ``src/auth/utils.py``
            - ``from ..models`` → ``src/models/__init__.py``

        Args:
            importing_file: Path of the importing file.
            import_info: The parsed import information.

        Returns:
            The node ID of the target file, or ``None`` if not found.
        """
        module = import_info.module
        assert module.startswith("."), f"Expected relative import, got {module!r}"

        # Count the leading dots to determine directory depth
        dot_count = 0
        for ch in module:
            if ch == ".":
                dot_count += 1
            else:
                break

        # Extract the module name (after the dots)
        remainder = module[dot_count:]

        # Start from the importing file's parent directory
        base = PurePosixPath(importing_file).parent

        # Traverse up (dot_count - 1) more levels to reach the target base
        for _ in range(dot_count - 1):
            base = base.parent

        # Construct the target directory by appending the module path
        if remainder:
            segments = remainder.split(".")
            target_dir = base / PurePosixPath(*segments)
        else:
            target_dir = base

        return self._try_python_paths(str(target_dir))

    def _resolve_python_absolute(
        self,
        import_info: ImportInfo,
        source_roots: set[str] | None = None,
    ) -> str | None:
        """Resolve an absolute Python import (``from mypackage.auth import validate``)."""
        module = import_info.module
        target_path = str(PurePosixPath(*module.split(".")))

        if result := self._try_python_paths(target_path):
            return result

        if source_roots:
            for root in source_roots:
                if not (result := self._try_python_paths(f"{root}/{target_path}")):
                    continue
                return result

    def _try_python_paths(self, base_path: str) -> str | None:
        """
        Try common Python file resolution patterns for a base path.

        Checks resolved paths in order of preference:
        1. ``base_path.py`` (module as a single .py file)
        2. ``base_path/__init__.py`` (module as a package directory)

        Args:
            base_path: The base path without extension (e.g., ``src/auth/utils``).

        Returns:
            The first matching file's node ID, or ``None`` if no match found.
        """
        if not self._file_index:
            return

        candidates = [
            f"{base_path}.py",
            f"{base_path}/__init__.py",
        ]
        for candidate in candidates:
            if candidate not in self._file_index:
                continue
            return self._file_index[candidate]

    def _resolve_js_ts(self, importing_file: str, import_info: ImportInfo) -> str | None:
        """
        Resolve a JavaScript/TypeScript import to a file node ID.

        Only resolves **relative** imports starting with ``./`` or ``../``.
        Bare specifiers (e.g., ``express``, ``@types/node``) are treated
        as external packages and return ``None``.

        Examples:
            In ``lib/index.ts``:
            - ``import { foo } from './utils'`` → ``lib/utils.ts``
            - ``import { User } from './models'`` → ``lib/models/index.ts``
            - ``import express from 'express'`` → ``None`` (external)

        Args:
            importing_file: Path of the importing file.
            import_info: The parsed import information.

        Returns:
            The node ID of the target file, or ``None`` if external or not found.
        """
        module = import_info.module

        # Only handle relative imports; bare specifiers are external
        if not module.startswith("."):
            return None

        # Resolve the relative path against the importing file's directory
        resolved = PurePosixPath(importing_file).parent / module

        # Normalize the path components
        resolved_str = str(PurePosixPath(*resolved.parts))

        return self._try_js_ts_paths(resolved_str)

    def _try_js_ts_paths(self, base_path: str) -> str | None:
        """
        Try common JavaScript/TypeScript file resolution patterns.

        Checks resolved paths in the following order:
        1. ``base_path`` with no modification (already fully qualified)
        2. ``base_path`` + each JS/TS extension (.ts, .js, .tsx, .jsx)
        3. ``base_path/index`` + each JS/TS extension (directory index)

        This matches the behavior of Node.js and TypeScript module resolution.

        Args:
            base_path: The base path (may or may not have extension).
            file_index: Mapping of file paths to node IDs.

        Returns:
            The first matching file's node ID, or ``None`` if no match found.
        """

        if not self._file_index:
            return

        # Strategy 1: Exact match (import already includes extension).
        if base_path in self._file_index:
            return self._file_index[base_path]

        # Strategy 2: Try appending each known extension.
        for ext in self._JS_TS_EXTENSIONS:
            if (candidate := f"{base_path}{ext}") in self._file_index:
                return self._file_index[candidate]

        # Strategy 3: Treat as directory and look for index file.
        for ext in self._JS_TS_EXTENSIONS:
            if (candidate := f"{base_path}/index{ext}") in self._file_index:
                return self._file_index[candidate]

        return None
