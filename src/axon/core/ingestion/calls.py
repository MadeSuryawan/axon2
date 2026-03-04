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

from logging import getLogger

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)
from axon.core.ingestion.parser_phase import FileParseData
from axon.core.ingestion.symbol_lookup import (
    build_file_symbol_index,
    build_name_index,
    find_containing_symbol,
)
from axon.core.parsers.base import CallInfo, SymbolInfo

logger = getLogger(__name__)


class Calls:
    _CALLABLE_LABELS: tuple[NodeLabel, ...] = (
        NodeLabel.FUNCTION,
        NodeLabel.METHOD,
        NodeLabel.CLASS,
    )

    _KIND_TO_LABEL: dict[str, NodeLabel] = {
        "function": NodeLabel.FUNCTION,
        "method": NodeLabel.METHOD,
        "class": NodeLabel.CLASS,
    }

    # Names that should never produce CALLS edges.  These are language builtins,
    # stdlib utilities, framework hooks, and common JS/TS globals whose definitions
    # do not exist in the user's codebase.  Filtering them before resolution
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
    ) -> None:
        self._parse_data = parse_data
        self._source_id: str = ""
        self._file_path: str = ""
        self._graph = graph
        self._seen: set[str] = set()
        self._call_index = build_name_index(graph, self._CALLABLE_LABELS)
        self._file_sym_index = build_file_symbol_index(graph, self._CALLABLE_LABELS)
        self._no_symbols: list[tuple[str, int, str]] = []

    @property
    def no_symbols(self) -> list[tuple[str, int, str]]:
        return self._no_symbols

    def process_calls(self) -> None:
        """
        Resolve call expressions and create CALLS relationships in the graph.

        For each call expression in the parse data:

        1. Determine which symbol in the file *contains* the call (by line
        number range).
        2. Resolve the call to a target symbol node.
        3. Create a CALLS relationship from the containing symbol to the
        target, with a ``confidence`` property.

        Skips calls where:
        - The containing symbol cannot be determined.
        - The target cannot be resolved.
        - A relationship with the same ID already exists (deduplication).
        """

        for fpd in self._parse_data:
            self._file_path = fpd.file_path
            # Process all call expressions in the file
            self._process_expressions_calls(fpd.parse_result.calls, fpd)
            # Process decorators on all symbols
            self._process_symbol_decorators(fpd.parse_result.symbols)

        logger.debug(f"calls containing no symbols -> {len(self._no_symbols)}")

    def _process_expressions_calls(self, calls: list[CallInfo], fpd: FileParseData) -> None:
        """
        Process all call expressions in the file.

        Args:
            calls: List of parsed call expressions.
            fpd: File parse data for the current file.
        """

        for call in calls:
            if self._is_blocklisted(call.name, call.receiver):
                continue

            source_id = find_containing_symbol(call.line, fpd.file_path, self._file_sym_index)
            if not source_id:
                self._no_symbols.append((call.name, call.line, fpd.file_path))
                continue

            self._source_id = source_id
            self._file_path = fpd.file_path

            # Main call resolution
            self._resolve_and_link_call(call)
            # Callback arguments: bare identifiers passed as arguments
            self._process_callback_arguments(call)
            # Receiver-based method calls
            self._process_receiver_calls(call)

    def _is_blocklisted(self, name: str, receiver: str) -> bool:
        """
        Check if a call name should be skipped based on the blocklist.

        Calls with receivers of "self" or "this" are never blocklisted to allow
        method resolution within the same class.

        Args:
            name: The call name to check.
            receiver: The receiver object, if any (e.g., "self", "this", or a class name).

        Returns:
            True if the call should be skipped, False otherwise.
        """
        if receiver in ("self", "this"):
            return False
        return name in self._CALL_BLOCKLIST

    def _resolve_and_link_call(
        self,
        call: CallInfo,
        confidence_multiplier: float = 1.0,
    ) -> None:
        """
        Resolve a call and create a CALLS edge if a target is found.

        Args:
            call: The parsed call information.
            confidence_multiplier: Factor to apply to the resolved confidence.
        """
        target_id, confidence = self._resolve_call(call)
        if target_id:
            self._add_calls_edge(
                self._source_id,
                target_id,
                confidence * confidence_multiplier,
            )

    def _resolve_call(self, call: CallInfo) -> tuple[str | None, float]:
        """
        Resolve a call expression to a target node ID and confidence score.

        Resolution strategy (tried in order):

        1. **Same-file exact match** (confidence 1.0) -- the called symbol is
        defined in the same file as the caller.
        2. **Import-resolved match** (confidence 1.0) -- the called name was
        imported into this file; find the symbol in the imported file.
        3. **Global fuzzy match** (confidence 0.5) -- any symbol with this name
        anywhere in the codebase.  If multiple matches exist, the one with
        the shortest file path is chosen (heuristic for proximity).

        For method calls (``call.receiver`` is non-empty):
        - If the receiver is ``"self"`` or ``"this"``, look for a method with
        that name in the same class (same file, matching class_name).
        - Otherwise, try to resolve the method name globally.

        Args:
            call: The parsed call information containing receiver.

        Returns:
            A tuple of ``(node_id, confidence)`` or ``(None, 0.0)`` if the
            call cannot be resolved.
        """

        name = call.name
        receiver = call.receiver

        if receiver in ("self", "this") and (result := self._resolve_self_method(name)):
            return result, 1.0

        # Without type info the receiver doesn't help — fall through to name-based resolution.
        if not (candidate_ids := self._call_index.get(name, [])):
            return None, 0.0

        # 1. Same-file exact match.
        for nid in candidate_ids:
            if (node := self._graph.get_node(nid)) and node.file_path == self._file_path:
                return nid, 1.0

        # 2. Import-resolved match.
        if imported_target := self._resolve_via_imports(name, candidate_ids):
            return imported_target, 1.0

        # 3. Global fuzzy match -- prefer shortest file path.
        return self._pick_closest(candidate_ids), 0.5

    def _resolve_self_method(self, method_name: str) -> str | None:
        """
        Find a method with *method_name* in the same file (same class).

        When the receiver is ``self`` or ``this`` the target must be a Method
        node defined in the same file.
        """
        for nid in self._call_index.get(method_name, []):
            node = self._graph.get_node(nid)
            if node and node.label == NodeLabel.METHOD and node.file_path == self._file_path:
                return nid
        return None

    def _resolve_via_imports(self, name: str, candidate_ids: list[str]) -> str | None:
        """
        Check if *name* was imported into *file_path* and resolve to the target.

        Looks at IMPORTS relationships originating from this file's File node.
        For each imported file, checks whether any candidate symbol is defined
        there.  Also checks the ``symbols`` property to see if the specific
        name was explicitly imported.
        """
        graph = self._graph
        source_file_id = generate_id(NodeLabel.FILE, self._file_path)

        if not (import_rels := graph.get_outgoing(source_file_id, RelType.IMPORTS)):
            return None

        # Collect file paths of imported files, optionally filtering by
        # the imported symbol names.
        imported_file_ids: set[str] = set()
        for rel in import_rels:
            symbols_str: str = rel.properties.get("symbols", "")
            imported_names = {s.strip() for s in symbols_str.split(",") if s.strip()}

            # If the specific name was imported, or if it's a wildcard/full
            # module import (no specific names), include this target file.
            if not imported_names or name in imported_names:
                target_node = graph.get_node(rel.target)
                if target_node is not None:
                    imported_file_ids.add(target_node.file_path)

        for nid in candidate_ids:
            if (node := graph.get_node(nid)) and node.file_path in imported_file_ids:
                return nid
        return None

    def _process_callback_arguments(self, call: CallInfo) -> None:
        """
        Process callback arguments passed as bare identifiers.

        Creates CALLS edges for arguments like ``map(transform, items)``
        or ``Depends(get_db)``.

        Args :
            call: The parsed call information.
        """
        for arg_name in call.arguments:
            if self._is_blocklisted(arg_name, receiver=""):
                continue

            arg_call = CallInfo(name=arg_name, line=call.line)
            self._resolve_and_link_call(
                arg_call,
                confidence_multiplier=0.8,
            )

    def _process_receiver_calls(self, call: CallInfo) -> None:
        """
        Process receiver-based method calls.

        Creates CALLS edges for both the receiver class and the method
        resolution on that receiver.

        Args:
            call: The parsed call information containing receiver.
        """
        receiver = call.receiver
        if not receiver or receiver in ("self", "this"):
            return
        # Link to the receiver class
        receiver_call = CallInfo(name=receiver, line=call.line)
        self._resolve_and_link_call(receiver_call)

        # Resolve the method on the receiver
        self._resolve_receiver_method(self._source_id, receiver, call.name)

    def _resolve_receiver_method(self, source_id: str, receiver: str, method_name: str) -> None:
        """
        Resolve ``Receiver.method()`` to the METHOD node and create a CALLS edge.

        Looks for a METHOD node whose ``name`` matches *method_name* and whose
        ``class_name`` matches *receiver*.  Searches same-file first, then
        globally.
        """
        same_file_match: str = ""
        global_match: str = ""
        for nid in self._call_index.get(method_name, []):
            if (
                (node := self._graph.get_node(nid))
                and node.label == NodeLabel.METHOD
                and node.class_name == receiver
            ):
                if node.file_path == self._file_path:
                    same_file_match = nid
                    break
                elif not global_match:
                    global_match = nid

        if target := same_file_match or global_match:
            self._add_calls_edge(source_id, target, 0.8)

    def _process_symbol_decorators(self, symbols: list[SymbolInfo]) -> None:
        """
        Process all decorators on a symbol and create CALLS edges.

        Decorators are implicit calls — @cost_decorator on a function is
        equivalent to calling cost_decorator(func).

        Args:
            symbols: list of the symbol information containing decorators.
        """
        for symbol in symbols:
            if not symbol.decorators:
                continue

            if not (source_id := self._get_decorator_source_id(symbol)):
                continue

            for dec_name in symbol.decorators:
                target_id, confidence = self._resolve_decorator_target(dec_name, symbol.start_line)
                if not target_id:
                    continue
                self._add_calls_edge(source_id, target_id, confidence)

    def _get_decorator_source_id(self, symbol: SymbolInfo) -> str | None:
        """
        Generate the source node ID for a decorated symbol.

        Args:
            symbol: The symbol information containing decorators.

        Returns:
            The generated node ID, or None if the symbol kind is unsupported.
        """
        if not (label := self._KIND_TO_LABEL.get(symbol.kind)):
            return

        symbol_name = (
            f"{symbol.class_name}.{symbol.name}"
            if symbol.kind == "method" and symbol.class_name
            else symbol.name
        )
        return generate_id(label, self._file_path, symbol_name)

    def _resolve_decorator_target(self, dec_name: str, line: int) -> tuple[str | None, float]:
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
        target_id, confidence = self._resolve_call(call_obj)

        # Try full dotted name as fallback
        if not target_id and "." in dec_name:
            call_obj = CallInfo(name=dec_name, line=line)
            target_id, confidence = self._resolve_call(call_obj)

        return target_id, confidence

    def _add_calls_edge(
        self,
        source_id: str,
        target_id: str,
        confidence: float,
    ) -> None:
        """Create a deduplicated CALLS relationship."""
        if not source_id or not target_id:
            return  # Guard against empty IDs
        rel_id = f"calls:{source_id}->{target_id}"
        if rel_id not in self._seen:
            self._seen.add(rel_id)
            self._graph.add_relationship(
                GraphRelationship(
                    id=rel_id,
                    type=RelType.CALLS,
                    source=source_id,
                    target=target_id,
                    properties={"confidence": confidence},
                ),
            )

    def _pick_closest(self, candidate_ids: list[str]) -> str | None:
        """
        Pick the candidate with the shortest file path (proximity heuristic).

        Returns ``None`` if no candidates can be resolved to actual nodes.
        """
        best_id = None
        best_path_len = float("inf")

        for nid in candidate_ids:
            node = self._graph.get_node(nid)
            if node is not None and len(node.file_path) < best_path_len:
                best_path_len = len(node.file_path)
                best_id = nid

        return best_id
