"""
Phase 5: Call tracing for Axon.

Takes FileParseData from the parser phase and resolves call expressions to
target symbol nodes, creating CALLS relationships with confidence scores.

Resolution priority:
1. Same-file exact match (confidence 1.0)
2. Import-resolved match (confidence 1.0)
3. Global fuzzy match (confidence 0.5)
4. Receiver method resolution (confidence 0.8)
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from logging import getLogger
from os import cpu_count

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)
from axon.core.ingestion.parser_phase import FileParseData
from axon.core.ingestion.resolved import ResolvedEdge
from axon.core.ingestion.symbol_lookup import (
    build_file_symbol_index,
    build_name_index,
    find_containing_symbol,
)
from axon.core.parsers.base import CallInfo, SymbolInfo

logger = getLogger(__name__)


@dataclass
class ResolveCallParams:
    """Parameters for resolving a call expression."""

    call: CallInfo
    file_path: str
    call_index: dict[str, list[str]]
    graph: KnowledgeGraph
    caller_class_name: str | None = None
    import_cache: dict[str, set[str]] | None = None


@dataclass
class ResolveReceiverMethodParams:
    """Parameters for resolving a receiver method call."""

    receiver: str
    method_name: str
    source_id: str
    file_path: str
    call_index: dict[str, list[str]]
    graph: KnowledgeGraph


class Calls:
    """
    Call tracing processor for resolving call expressions to target symbol nodes.

    This class takes FileParseData from the parser phase and resolves call
    expressions to target symbol nodes, creating CALLS relationships with
    confidence scores.

    Resolution priority:
    1. Same-file exact match (confidence 1.0) - the called symbol is defined
       in the same file as the caller.
    2. Import-resolved match (confidence 1.0) - the called name was imported
       into this file; find the symbol in the imported file.
    3. Global fuzzy match (confidence 0.5) - any symbol with this name anywhere
       in the codebase. If multiple matches exist, the one sharing the longest
       directory prefix with the caller is preferred.
    4. Receiver method resolution (confidence 0.8) - for method calls on
       receiver objects (e.g., obj.method()).
    """

    # Node labels that represent callable entities (functions, methods, classes)
    _CALLABLE_LABELS: tuple[NodeLabel, ...] = (
        NodeLabel.FUNCTION,
        NodeLabel.METHOD,
        NodeLabel.CLASS,
    )

    # Mapping from symbol kind to node label
    _KIND_TO_LABEL: dict[str, NodeLabel] = {
        "function": NodeLabel.FUNCTION,
        "method": NodeLabel.METHOD,
        "class": NodeLabel.CLASS,
    }

    # Names that should never produce CALLS edges. These are language builtins,
    # stdlib utilities, framework hooks, and common JS/TS globals whose definitions
    # do not exist in the user's codebase. Filtering them before resolution
    # prevents low-confidence global-fuzzy matches against short, common names.
    _CALL_BLOCKLIST: frozenset[str] = frozenset(
        {
            # Python builtins
            "print",
            "len",
            "range",
            "map",
            "filter",
            "sorted",
            "list",
            "dict",
            "set",
            "str",
            "int",
            "float",
            "bool",
            "type",
            "super",
            "isinstance",
            "issubclass",
            "hasattr",
            "getattr",
            "setattr",
            "open",
            "iter",
            "next",
            "zip",
            "enumerate",
            "any",
            "all",
            "min",
            "max",
            "sum",
            "abs",
            "round",
            "repr",
            "id",
            "hash",
            "dir",
            "vars",
            "input",
            "format",
            "tuple",
            "frozenset",
            "bytes",
            "bytearray",
            "memoryview",
            "object",
            "property",
            "classmethod",
            "staticmethod",
            "delattr",
            "callable",
            "compile",
            "eval",
            "exec",
            "globals",
            "locals",
            "breakpoint",
            "exit",
            "quit",
            # Python stdlib — common method names that collide with user-defined symbols
            "append",
            "extend",
            "update",
            "pop",
            "get",
            "items",
            "keys",
            "values",
            "split",
            "join",
            "strip",
            "replace",
            "startswith",
            "endswith",
            "lower",
            "upper",
            "encode",
            "decode",
            "read",
            "write",
            "close",
            # JS/TS built-in globals
            "console",
            "setTimeout",
            "setInterval",
            "clearTimeout",
            "clearInterval",
            "JSON",
            "Array",
            "Object",
            "Promise",
            "Math",
            "Date",
            "Error",
            "Symbol",
            "parseInt",
            "parseFloat",
            "isNaN",
            "isFinite",
            "encodeURIComponent",
            "decodeURIComponent",
            "fetch",
            "require",
            "exports",
            "module",
            "document",
            "window",
            "process",
            "Buffer",
            "URL",
            # JS/TS dotted method names extracted as bare call names
            "log",
            "error",
            "warn",
            "info",
            "debug",
            "parse",
            "stringify",
            "assign",
            "freeze",
            "isArray",
            "from",
            "of",
            "resolve",
            "reject",
            "race",
            "floor",
            "ceil",
            "random",
            # React hooks
            "useState",
            "useEffect",
            "useRef",
            "useCallback",
            "useMemo",
            "useContext",
            "useReducer",
            "useLayoutEffect",
            "useImperativeHandle",
            "useDebugValue",
            "useId",
            "useTransition",
            "useDeferredValue",
        },
    )

    def __init__(
        self,
        parse_data: list[FileParseData],
        graph: KnowledgeGraph,
        name_index: dict[str, list[str]],
    ) -> None:
        """
        Initialize the Calls processor.

        Args:
            parse_data: List of FileParseData objects from the parser phase.
            graph: The knowledge graph to populate with CALLS relationships.
            name_index: Optional pre-built name index; built automatically if None.
        """
        self._parse_data = parse_data
        self._graph = graph
        # Build call index from name index or create new one
        self._call_index = (
            name_index if name_index else build_name_index(graph, self._CALLABLE_LABELS)
        )
        self._file_sym_index = build_file_symbol_index(graph, self._CALLABLE_LABELS)

        # Per-file state (set during processing)
        self._file_path: str = ""
        self._import_cache: dict[str, set[str]] | None = None
        self._no_symbols: list[tuple[str, int, str]] = []
        self._workers = min(cpu_count() or 4, 8, len(self._parse_data))

    @property
    def no_symbols(self) -> list[tuple[str, int, str]]:
        """Return list of calls that had no containing symbol."""
        return self._no_symbols

    def process_calls(
        self,
        *,
        parallel: bool = False,
        collect: bool = False,
    ) -> list[ResolvedEdge] | None:
        """
        Resolve call expressions and create CALLS relationships in the graph.

        For each call expression in the parse data:

        1. Determine which symbol in the file *contains* the call (by line
           number range).
        2. Resolve the call to a target symbol node.
        3. Create a CALLS relationship from the containing symbol to the
           target, with a ``confidence`` property.

        Args:
            parallel: When True, resolve files using a thread pool.
            collect: When True, return the list of ResolvedEdge objects instead
                     of writing them to the graph.

        Returns:
            A list of ResolvedEdge when *collect* is True, otherwise None.

        Skips calls where:
        - The containing symbol cannot be determined.
        - The target cannot be resolved.
        - A relationship with the same ID already exists (deduplication).
        """
        if parallel and len(self._parse_data) > 1:
            # Use thread pool for parallel processing
            with ThreadPoolExecutor(max_workers=self._workers) as pool:
                per_file_edges = list(pool.map(self._resolve_file, self._parse_data))
        else:
            # Sequential processing
            per_file_edges = [self._resolve_file(fpd) for fpd in self._parse_data]

        deduped = self._collect_deduped_edges(per_file_edges)

        if collect:
            # Return edges without adding to graph
            return deduped

        self._add_edges_to_graph(deduped)
        logger.debug(f"calls containing no symbols -> {len(self._no_symbols)}")

    def _collect_deduped_edges(
        self,
        per_file_edges: list[list[ResolvedEdge]],
    ) -> list[ResolvedEdge]:
        """
        Collect edges from all files, deduplicating across files.

        Args:
            per_file_edges: List of lists of ResolvedEdge objects, one list per file.

        Returns:
            A single list of deduplicated ResolvedEdge objects.
        """
        # Global deduplication across all files
        # This ensures no duplicate edges are added to the graph
        seen: set[str] = set()
        deduped: list[ResolvedEdge] = []
        for file_edges in per_file_edges:
            for edge in file_edges:
                if edge.rel_id in seen:
                    continue
                seen.add(edge.rel_id)
                deduped.append(edge)
        return deduped

    def _add_edges_to_graph(self, edges: list[ResolvedEdge]) -> None:
        """
        Add a list of ResolvedEdge objects to the knowledge graph.

        Args:
            edges: List of ResolvedEdge objects to add.

        """
        for edge in edges:
            self._graph.add_relationship(
                GraphRelationship(
                    id=edge.rel_id,
                    type=edge.rel_type,
                    source=edge.source,
                    target=edge.target,
                    properties=edge.properties,
                ),
            )

    def _resolve_file(self, fpd: FileParseData) -> list[ResolvedEdge]:
        """
        Resolve all call expressions in a single file to ResolvedEdge objects.

        This is a pure function (reads from graph but does not mutate it)
        that can be called in parallel across files.

        Args:
            fpd: File parse data for the current file.

        Returns:
            List of resolved edges for this file.
        """
        edges: list[ResolvedEdge] = []
        seen: set[str] = set()

        # Set file context for this file's processing
        self._file_path = fpd.file_path
        # Build import cache once per file for efficiency
        self._import_cache = self._build_import_cache(fpd.file_path, self._graph)

        # Process each call expression in the file
        for call in fpd.parse_result.calls:
            self._process_single_call(call, fpd, seen, edges)

        # Process decorators on all symbols in the file
        for symbol in fpd.parse_result.symbols:
            self._process_symbol_decorators(symbol, fpd, seen, edges)

        return edges

    def _process_single_call(
        self,
        call: CallInfo,
        fpd: FileParseData,
        seen: set[str],
        edges: list[ResolvedEdge],
    ) -> None:
        """
        Process a single call expression and add edges to the result list.

        Handles:
        1. Blocklist filtering
        2. Finding containing symbol
        3. Main call resolution
        4. Callback arguments
        5. Receiver method calls

        Args:
            call: The call expression to process.
            fpd: File parse data containing the call.
            seen: Set of already-seen relationship IDs for deduplication.
            edges: List to append resolved edges to.
        """
        # Skip blocklisted calls (except self/this method calls)
        if call.name in self._CALL_BLOCKLIST and call.receiver not in ("self", "this"):
            return

        # Find the symbol that contains this call (the caller)
        source_id = find_containing_symbol(call.line, fpd.file_path, self._file_sym_index)
        if not source_id:
            # logger.debug(
            #     "No containing symbol for call %s at line %d in %s",
            #     call.name,
            #     call.line,
            #     fpd.file_path,
            # )
            self._no_symbols.append((call.name, call.line, fpd.file_path))
            return

        # Get caller class name for self/this method resolution
        caller_class_name: str | None = None
        if call.receiver in ("self", "this"):
            source_node = self._graph.get_node(source_id)
            if source_node is not None:
                caller_class_name = source_node.class_name

        # Resolve and add main call edge
        target_id, confidence = self._resolve_call(
            ResolveCallParams(
                call=call,
                file_path=self._file_path,
                call_index=self._call_index,
                graph=self._graph,
                caller_class_name=caller_class_name,
                import_cache=self._import_cache,
            ),
        )
        if target_id is not None:
            edge = self._make_edge(source_id, target_id, confidence, seen)
            if edge is not None:
                edges.append(edge)

        # Process callback arguments (bare identifiers passed as arguments)
        self._process_callback_arguments(call, source_id, seen, edges)

        # Process receiver-based method calls (e.g., obj.method())
        self._process_receiver_calls(call, source_id, seen, edges)

    def _process_callback_arguments(
        self,
        call: CallInfo,
        source_id: str,
        seen: set[str],
        edges: list[ResolvedEdge],
    ) -> None:
        """
        Process callback arguments passed as bare identifiers.

        Creates CALLS edges for arguments like ``map(transform, items)``
        or ``Depends(get_db)``. These are function references passed as
        arguments to other functions.

        Args:
            call: The parsed call information.
            source_id: The node ID of the containing symbol.
            seen: Set of already-seen relationship IDs for deduplication.
            edges: List to append resolved edges to.
        """
        for arg_name in call.arguments:
            if arg_name in self._CALL_BLOCKLIST:
                continue

            arg_call = CallInfo(name=arg_name, line=call.line)
            arg_id, arg_conf = self._resolve_call(
                ResolveCallParams(
                    call=arg_call,
                    file_path=self._file_path,
                    call_index=self._call_index,
                    graph=self._graph,
                    import_cache=self._import_cache,
                ),
            )
            if arg_id is not None:
                edge = self._make_edge(source_id, arg_id, arg_conf * 0.8, seen)
                if edge is not None:
                    edges.append(edge)

    def _process_receiver_calls(
        self,
        call: CallInfo,
        source_id: str,
        seen: set[str],
        edges: list[ResolvedEdge],
    ) -> None:
        """
        Process receiver-based method calls.

        Creates CALLS edges for both the receiver class and the method
        resolution on that receiver.

        For example, for ``obj.method()``:
        1. Creates a CALLS edge to the receiver class (obj)
        2. Creates a CALLS edge to the resolved method (method on obj's class)

        Args:
            call: The parsed call information containing receiver.
            source_id: The node ID of the containing symbol.
            seen: Set of already-seen relationship IDs for deduplication.
            edges: List to append resolved edges to.
        """
        receiver = call.receiver
        if not receiver or receiver in ("self", "this"):
            return

        # Link to the receiver class
        receiver_call = CallInfo(name=receiver, line=call.line)
        recv_id, recv_conf = self._resolve_call(
            ResolveCallParams(
                call=receiver_call,
                file_path=self._file_path,
                call_index=self._call_index,
                graph=self._graph,
                import_cache=self._import_cache,
            ),
        )
        if recv_id is not None:
            edge = self._make_edge(source_id, recv_id, recv_conf, seen)
            if edge is not None:
                edges.append(edge)

        # Resolve the method on the receiver class
        recv_method_edge = self._resolve_receiver_method(
            ResolveReceiverMethodParams(
                receiver=receiver,
                method_name=call.name,
                source_id=source_id,
                file_path=self._file_path,
                call_index=self._call_index,
                graph=self._graph,
            ),
        )
        if recv_method_edge is not None and recv_method_edge.rel_id not in seen:
            seen.add(recv_method_edge.rel_id)
            edges.append(recv_method_edge)

    def _process_symbol_decorators(
        self,
        symbol: SymbolInfo,
        fpd: FileParseData,
        seen: set[str],
        edges: list[ResolvedEdge],
    ) -> None:
        """
        Process all decorators on a symbol and create CALLS edges.

        Decorators are implicit calls — @cost_decorator on a function is
        equivalent to calling cost_decorator(func).

        Args:
            symbol: The symbol information containing decorators.
            fpd: File parse data for context.
            seen: Set of already-seen relationship IDs for deduplication.
            edges: List to append resolved edges to.
        """
        if not symbol.decorators:
            return

        # Generate source ID for the decorated symbol
        symbol_name = (
            f"{symbol.class_name}.{symbol.name}"
            if symbol.kind == "method" and symbol.class_name
            else symbol.name
        )
        label = self._KIND_TO_LABEL.get(symbol.kind)
        if label is None:
            return
        source_id = generate_id(label, fpd.file_path, symbol_name)

        # Resolve each decorator
        for dec_name in symbol.decorators:
            target_id, confidence = self._resolve_decorator(dec_name, symbol.start_line)
            if target_id is not None:
                edge = self._make_edge(source_id, target_id, confidence, seen)
                if edge is not None:
                    edges.append(edge)

    def _resolve_decorator(
        self,
        dec_name: str,
        line: int,
    ) -> tuple[str | None, float]:
        """
        Resolve a decorator name to its target node ID.

        Tries the base name first (e.g., "route" from "app.route"), then
        falls back to the full dotted name if available.

        Args:
            dec_name: The decorator name (may be dotted like "app.route").
            line: The line number where the decorator appears.

        Returns:
            A tuple of ``(node_id, confidence)`` or ``(None, 0.0)`` if unresolved.
        """
        # Strip the base name for dotted decorators (e.g., "app.route" → "route")
        base_name = dec_name.rsplit(".", 1)[-1] if "." in dec_name else dec_name
        call_obj = CallInfo(name=base_name, line=line)
        target_id, confidence = self._resolve_call(
            ResolveCallParams(
                call=call_obj,
                file_path=self._file_path,
                call_index=self._call_index,
                graph=self._graph,
                import_cache=self._import_cache,
            ),
        )

        # Try full dotted name as fallback
        if target_id is None and "." in dec_name:
            call_obj = CallInfo(name=dec_name, line=line)
            target_id, confidence = self._resolve_call(
                ResolveCallParams(
                    call=call_obj,
                    file_path=self._file_path,
                    call_index=self._call_index,
                    graph=self._graph,
                    import_cache=self._import_cache,
                ),
            )

        return target_id, confidence

    # -------------------------------------------------------------------------
    # Core resolution methods (mirror the standalone functions from source)
    # -------------------------------------------------------------------------

    def _resolve_call(
        self,
        params: ResolveCallParams,
    ) -> tuple[str | None, float]:
        """
        Resolve a call expression to a target node ID and confidence score.

        Resolution strategy (tried in order):

        1. **Same-file exact match** (confidence 1.0) -- the called symbol is
           defined in the same file as the caller.
        2. **Import-resolved match** (confidence 1.0) -- the called name was
           imported into this file; find the symbol in the imported file.
        3. **Global fuzzy match** (confidence 0.5) -- any symbol with this name
           anywhere in the codebase. If multiple matches exist, the one sharing
           the longest directory prefix with the caller is preferred.

        For method calls (``call.receiver`` is non-empty):
        - If the receiver is ``"self"`` or ``"this"``, look for a method with
          that name in the same class (same file, matching class_name).
        - Otherwise, try to resolve the method name globally.

        Args:
            params: ResolveCallParams dataclass containing all resolution parameters.

        Returns:
            A tuple of ``(node_id, confidence)`` or ``(None, 0.0)`` if the
            call cannot be resolved.
        """
        call = params.call
        file_path = params.file_path
        call_index = params.call_index
        graph = params.graph
        caller_class_name = params.caller_class_name
        import_cache = params.import_cache

        name = call.name
        receiver = call.receiver

        # Handle self/this method resolution first
        if receiver in ("self", "this"):
            result = self._resolve_self_method(
                name,
                file_path,
                call_index,
                graph,
                caller_class_name,
            )
            if result is not None:
                return result, 1.0

        # Get candidate IDs from the name index
        candidate_ids = call_index.get(name, [])
        if not candidate_ids:
            return None, 0.0

        # 1. Same-file exact match (highest priority - confidence 1.0)
        for nid in candidate_ids:
            node = graph.get_node(nid)
            if node is not None and node.file_path == file_path:
                return nid, 1.0

        # 2. Import-resolved match (confidence 1.0)
        effective_cache = (
            import_cache if import_cache is not None else self._build_import_cache(file_path, graph)
        )
        imported_target = self._resolve_via_imports(name, candidate_ids, graph, effective_cache)
        if imported_target is not None:
            return imported_target, 1.0

        # 3. Global fuzzy match (confidence 0.5)
        # Limit candidates to prevent noise from ambiguous names
        if len(candidate_ids) > 5:
            return None, 0.0
        return self._pick_closest(candidate_ids, graph, caller_file_path=file_path), 0.5

    def _resolve_self_method(
        self,
        method_name: str,
        file_path: str,
        call_index: dict[str, list[str]],
        graph: KnowledgeGraph,
        caller_class_name: str | None = None,
    ) -> str | None:
        """
        Find a method with *method_name* in the same file and class.

        When the receiver is ``self`` or ``this`` the target must be a Method
        node defined in the same file. If *caller_class_name* is provided,
        candidates are further filtered to the same class.

        Args:
            method_name: The name of the method to resolve.
            file_path: The file path of the caller.
            call_index: Mapping from symbol names to node IDs.
            graph: The knowledge graph.
            caller_class_name: Optional class name to filter by.

        Returns:
            The node ID of the resolved method, or None if not found.
        """
        fallback: str | None = None
        for nid in call_index.get(method_name, []):
            node = graph.get_node(nid)
            if node is not None and node.label == NodeLabel.METHOD and node.file_path == file_path:
                if caller_class_name and node.class_name == caller_class_name:
                    return nid
                if fallback is None:
                    fallback = nid
        return fallback

    def _build_import_cache(
        self,
        file_path: str,
        graph: KnowledgeGraph,
    ) -> dict[str, set[str]]:
        """
        Build {symbol_name → set of imported file_paths} for a file.

        The special key ``"*"`` contains file paths from wildcard/full-module imports.

        Args:
            file_path: The file to build import cache for.
            graph: The knowledge graph.

        Returns:
            A dictionary mapping symbol names to sets of imported file paths.
        """
        source_file_id = generate_id(NodeLabel.FILE, file_path)
        import_rels = graph.get_outgoing(source_file_id, RelType.IMPORTS)

        cache: dict[str, set[str]] = {}
        for rel in import_rels:
            target_node = graph.get_node(rel.target)
            if target_node is None:
                continue
            symbols_str = rel.properties.get("symbols", "")
            imported_names = {s.strip() for s in symbols_str.split(",") if s.strip()}
            if not imported_names:
                # Wildcard import - add to special key
                cache.setdefault("*", set()).add(target_node.file_path)
            else:
                for sym_name in imported_names:
                    cache.setdefault(sym_name, set()).add(target_node.file_path)
        return cache

    def _resolve_via_imports(
        self,
        name: str,
        candidate_ids: list[str],
        graph: KnowledgeGraph,
        import_cache: dict[str, set[str]],
    ) -> str | None:
        """
        Check if *name* was imported and resolve to the target using cached data.

        Uses the pre-built *import_cache* (from :func:`_build_import_cache`)
        to avoid re-scanning IMPORTS relationships for every call in the same file.

        Args:
            name: The symbol name to resolve.
            candidate_ids: List of candidate node IDs from the name index.
            graph: The knowledge graph.
            import_cache: Pre-built import cache.

        Returns:
            The node ID of the imported target, or None if not found.
        """
        if not import_cache:
            return None

        imported_file_paths = import_cache.get(name, set()) | import_cache.get("*", set())
        if not imported_file_paths:
            return None

        for nid in candidate_ids:
            node = graph.get_node(nid)
            if node is not None and node.file_path in imported_file_paths:
                return nid

        return None

    def _common_prefix_len(self, a: str, b: str) -> int:
        """
        Return the length of the common directory prefix between two paths.

        Used as a proximity heuristic - symbols in the same directory subtree
        are more likely to be related than symbols in distant directories.

        Args:
            a: First path.
            b: Second path.

        Returns:
            The number of path components shared at the beginning.
        """
        parts_a = a.split("/")
        parts_b = b.split("/")
        common = 0
        for pa, pb in zip(parts_a, parts_b, strict=False):
            if pa == pb:
                common += 1
            else:
                break
        return common

    def _pick_closest(
        self,
        candidate_ids: list[str],
        graph: KnowledgeGraph,
        caller_file_path: str = "",
    ) -> str | None:
        """
        Pick the candidate sharing the longest directory prefix with the caller.

        Falls back to shortest file path when no caller path is provided.
        Returns ``None`` if no candidates can be resolved to actual nodes.

        Args:
            candidate_ids: List of candidate node IDs to choose from.
            graph: The knowledge graph.
            caller_file_path: Path of the file containing the call.

        Returns:
            The node ID of the closest match, or None if no candidates resolve.
        """
        best_id: str | None = None
        best_score: tuple[int, int] = (-1, 0)

        for nid in candidate_ids:
            node = graph.get_node(nid)
            if node is None:
                continue
            if caller_file_path:
                prefix = self._common_prefix_len(caller_file_path, node.file_path)
                score = (prefix, -len(node.file_path))
            else:
                score = (0, -len(node.file_path))
            if score > best_score:
                best_score = score
                best_id = nid

        return best_id

    def _make_edge(
        self,
        source_id: str,
        target_id: str,
        confidence: float,
        seen: set[str],
    ) -> ResolvedEdge | None:
        """
        Create a deduplicated ResolvedEdge, returning None if already seen.

        Args:
            source_id: The source node ID.
            target_id: The target node ID.
            confidence: The confidence score for this relationship.
            seen: Set of already-seen relationship IDs for deduplication.

        Returns:
            A new ResolvedEdge, or None if already seen.
        """
        rel_id = f"calls:{source_id}->{target_id}"
        if rel_id in seen:
            return None
        seen.add(rel_id)
        return ResolvedEdge(
            rel_id=rel_id,
            rel_type=RelType.CALLS,
            source=source_id,
            target=target_id,
            properties={"confidence": confidence},
        )

    def _resolve_receiver_method(
        self,
        params: ResolveReceiverMethodParams,
    ) -> ResolvedEdge | None:
        """
        Resolve ``Receiver.method()`` to the METHOD node and return a ResolvedEdge.

        Looks for a METHOD node whose ``name`` matches *method_name* and whose
        ``class_name`` matches *receiver*. Searches same-file first, then globally.

        Args:
            params: ResolveReceiverMethodParams dataclass containing all parameters.

        Returns:
            A ResolvedEdge for the receiver method, or None if not found.
        """
        receiver = params.receiver
        method_name = params.method_name
        source_id = params.source_id
        file_path = params.file_path
        call_index = params.call_index
        graph = params.graph

        same_file_match: str | None = None
        global_match: str | None = None

        for nid in call_index.get(method_name, []):
            node = graph.get_node(nid)
            if node is not None and node.label == NodeLabel.METHOD and node.class_name == receiver:
                if node.file_path == file_path:
                    same_file_match = nid
                    break
                elif global_match is None:
                    global_match = nid
            if same_file_match is not None:
                break

        target = same_file_match or global_match
        if target is not None:
            return ResolvedEdge(
                rel_id=f"calls:{source_id}->{target}",
                rel_type=RelType.CALLS,
                source=source_id,
                target=target,
                properties={"confidence": 0.8},
            )
        return None
