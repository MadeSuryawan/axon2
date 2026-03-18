"""Tests for the call tracing phase (Phase 5)."""

from dataclasses import dataclass

import pytest

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)
from axon.core.ingestion.calls import Calls
from axon.core.ingestion.parser_phase import FileParseData
from axon.core.ingestion.symbol_lookup import build_name_index
from axon.core.parsers.base import CallInfo, ParseResult

_CALLABLE_LABELS = (NodeLabel.FUNCTION, NodeLabel.METHOD, NodeLabel.CLASS)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _add_file_node(graph: KnowledgeGraph, path: str) -> str:
    """Add a File node and return its ID."""
    node_id = generate_id(NodeLabel.FILE, path)
    graph.add_node(
        GraphNode(
            id=node_id,
            label=NodeLabel.FILE,
            name=path.rsplit("/", 1)[-1],
            file_path=path,
        ),
    )
    return node_id


@dataclass(frozen=True)
class SymbolNodeInfo:
    """Parameters for adding a symbol node to the graph."""

    label: NodeLabel
    file_path: str
    name: str
    start_line: int
    end_line: int
    class_name: str = ""


def _add_symbol_node(
    graph: KnowledgeGraph,
    info: SymbolNodeInfo,
) -> str:
    """Add a symbol node with a DEFINES relationship from the file node."""
    symbol_name = (
        f"{info.class_name}.{info.name}"
        if info.label == NodeLabel.METHOD and info.class_name
        else info.name
    )
    node_id = generate_id(info.label, info.file_path, symbol_name)
    graph.add_node(
        GraphNode(
            id=node_id,
            label=info.label,
            name=info.name,
            file_path=info.file_path,
            start_line=info.start_line,
            end_line=info.end_line,
            class_name=info.class_name,
        ),
    )
    file_id = generate_id(NodeLabel.FILE, info.file_path)
    graph.add_relationship(
        GraphRelationship(
            id=f"defines:{file_id}->{node_id}",
            type=RelType.DEFINES,
            source=file_id,
            target=node_id,
        ),
    )
    return node_id


@pytest.fixture()
def graph() -> KnowledgeGraph:
    """
    Build a graph matching the test fixture specification.

    File: src/auth.py
        Function: validate (lines 1-10)
        Function: hash_password (lines 12-20)

    File: src/app.py
        Function: login (lines 1-15)

    File: src/utils.py
        Function: helper (lines 1-5)
    """
    g = KnowledgeGraph()

    # Files
    _add_file_node(g, "src/auth.py")
    _add_file_node(g, "src/app.py")
    _add_file_node(g, "src/utils.py")

    # Symbols in src/auth.py
    _add_symbol_node(g, SymbolNodeInfo(NodeLabel.FUNCTION, "src/auth.py", "validate", 1, 10))
    _add_symbol_node(g, SymbolNodeInfo(NodeLabel.FUNCTION, "src/auth.py", "hash_password", 12, 20))

    # Symbols in src/app.py
    _add_symbol_node(g, SymbolNodeInfo(NodeLabel.FUNCTION, "src/app.py", "login", 1, 15))

    # Symbols in src/utils.py
    _add_symbol_node(g, SymbolNodeInfo(NodeLabel.FUNCTION, "src/utils.py", "helper", 1, 5))

    return g


@pytest.fixture()
def parse_data() -> list[FileParseData]:
    """
    Parse data with calls matching the fixture specification.

    src/auth.py: hash_password() at line 5 (inside validate)
    src/app.py: validate() at line 8 (inside login)
    """
    return [
        FileParseData(
            file_path="src/auth.py",
            language="python",
            parse_result=ParseResult(
                calls=[CallInfo(name="hash_password", line=5)],
            ),
        ),
        FileParseData(
            file_path="src/app.py",
            language="python",
            parse_result=ParseResult(
                calls=[CallInfo(name="validate", line=8)],
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# build_name_index (callable labels)
# ---------------------------------------------------------------------------


class TestBuildCallIndex:
    """build_name_index creates correct mapping from graph symbol nodes."""

    def test_build_call_index(self, graph: KnowledgeGraph) -> None:
        index = build_name_index(graph, _CALLABLE_LABELS)

        # All four functions should appear.
        assert "validate" in index
        assert "hash_password" in index
        assert "login" in index
        assert "helper" in index

        # Each name maps to exactly one node ID.
        assert len(index["validate"]) == 1
        assert len(index["hash_password"]) == 1

        # IDs match expected generate_id output.
        expected_validate = generate_id(
            NodeLabel.FUNCTION,
            "src/auth.py",
            "validate",
        )
        assert index["validate"] == [expected_validate]

    def test_build_call_index_includes_classes(self) -> None:
        """Class nodes are included (for constructor calls)."""
        g = KnowledgeGraph()
        _add_file_node(g, "src/models.py")
        _add_symbol_node(g, SymbolNodeInfo(NodeLabel.CLASS, "src/models.py", "User", 1, 20))

        index = build_name_index(g, _CALLABLE_LABELS)
        assert "User" in index
        assert len(index["User"]) == 1

    def test_build_call_index_multiple_same_name(self) -> None:
        """Multiple symbols with the same name produce a list with all IDs."""
        g = KnowledgeGraph()
        _add_file_node(g, "src/a.py")
        _add_file_node(g, "src/b.py")
        _add_symbol_node(g, SymbolNodeInfo(NodeLabel.FUNCTION, "src/a.py", "init", 1, 5))
        _add_symbol_node(g, SymbolNodeInfo(NodeLabel.FUNCTION, "src/b.py", "init", 1, 5))

        index = build_name_index(g, _CALLABLE_LABELS)
        assert "init" in index
        assert len(index["init"]) == 2


# ---------------------------------------------------------------------------
# resolve_call — same-file
# ---------------------------------------------------------------------------


class TestResolveCallSameFile:
    """hash_password call in auth.py resolves locally (confidence 1.0)."""

    def test_resolve_call_same_file(self, graph: KnowledgeGraph) -> None:
        call = CallInfo(name="hash_password", line=5)
        # Instantiate Calls and set state for testing resolution
        calls_obj = Calls([], graph)
        calls_obj._file_path = "src/auth.py"

        target_id, confidence = calls_obj._resolve_call(call)

        expected_id = generate_id(
            NodeLabel.FUNCTION,
            "src/auth.py",
            "hash_password",
        )
        assert target_id == expected_id
        assert confidence == 1.0


# ---------------------------------------------------------------------------
# resolve_call — global fuzzy
# ---------------------------------------------------------------------------


class TestResolveCallGlobal:
    """validate call in app.py resolves globally (confidence 0.5)."""

    def test_resolve_call_global(self, graph: KnowledgeGraph) -> None:
        call = CallInfo(name="validate", line=8)
        # Instantiate Calls and set state for testing resolution
        calls_obj = Calls([], graph)
        calls_obj._file_path = "src/app.py"

        target_id, confidence = calls_obj._resolve_call(call)

        expected_id = generate_id(
            NodeLabel.FUNCTION,
            "src/auth.py",
            "validate",
        )
        assert target_id == expected_id
        assert confidence == 0.5


# ---------------------------------------------------------------------------
# resolve_call — unresolved
# ---------------------------------------------------------------------------


class TestResolveCallUnresolved:
    """Call to unknown function returns None."""

    def test_resolve_call_unresolved(self, graph: KnowledgeGraph) -> None:
        call = CallInfo(name="nonexistent_function", line=3)
        # Instantiate Calls and set state for testing resolution
        calls_obj = Calls([], graph)
        calls_obj._file_path = "src/auth.py"

        target_id, confidence = calls_obj._resolve_call(call)

        assert target_id is None
        assert confidence == 0.0


# ---------------------------------------------------------------------------
# process_calls — creates relationships
# ---------------------------------------------------------------------------


class TestProcessCallsCreatesRelationships:
    """process_calls creates CALLS edges in the graph."""

    def test_process_calls_creates_relationships(
        self,
        graph: KnowledgeGraph,
        parse_data: list[FileParseData],
    ) -> None:
        Calls(parse_data, graph).process_calls()

        calls_rels = graph.get_relationships_by_type(RelType.CALLS)
        assert len(calls_rels) == 2

        # Collect source->target pairs.
        pairs = {(r.source, r.target) for r in calls_rels}

        validate_id = generate_id(
            NodeLabel.FUNCTION,
            "src/auth.py",
            "validate",
        )
        hash_pw_id = generate_id(
            NodeLabel.FUNCTION,
            "src/auth.py",
            "hash_password",
        )
        login_id = generate_id(NodeLabel.FUNCTION, "src/app.py", "login")

        # validate -> hash_password (same-file call at line 5 inside validate)
        assert (validate_id, hash_pw_id) in pairs
        # login -> validate (cross-file call at line 8 inside login)
        assert (login_id, validate_id) in pairs


# ---------------------------------------------------------------------------
# process_calls — confidence scores
# ---------------------------------------------------------------------------


class TestProcessCallsConfidence:
    """Confidence scores are set correctly on CALLS relationships."""

    def test_process_calls_confidence(
        self,
        graph: KnowledgeGraph,
        parse_data: list[FileParseData],
    ) -> None:
        Calls(parse_data, graph).process_calls()

        calls_rels = graph.get_relationships_by_type(RelType.CALLS)

        validate_id = generate_id(
            NodeLabel.FUNCTION,
            "src/auth.py",
            "validate",
        )
        hash_pw_id = generate_id(
            NodeLabel.FUNCTION,
            "src/auth.py",
            "hash_password",
        )
        login_id = generate_id(NodeLabel.FUNCTION, "src/app.py", "login")

        confidences = {(r.source, r.target): r.properties["confidence"] for r in calls_rels}

        # Same-file call: confidence 1.0
        assert confidences[(validate_id, hash_pw_id)] == 1.0
        # Cross-file global match: confidence 0.5
        assert confidences[(login_id, validate_id)] == 0.5


# ---------------------------------------------------------------------------
# process_calls — no duplicates
# ---------------------------------------------------------------------------


class TestProcessCallsNoDuplicates:
    """Same call twice does not create duplicate edges."""

    def test_process_calls_no_duplicates(
        self,
        graph: KnowledgeGraph,
    ) -> None:
        # Two identical calls to hash_password inside validate.
        duplicate_parse_data = [
            FileParseData(
                file_path="src/auth.py",
                language="python",
                parse_result=ParseResult(
                    calls=[
                        CallInfo(name="hash_password", line=5),
                        CallInfo(name="hash_password", line=7),
                    ],
                ),
            ),
        ]

        Calls(duplicate_parse_data, graph).process_calls()

        calls_rels = graph.get_relationships_by_type(RelType.CALLS)
        # Both calls resolve to validate -> hash_password, but only one
        # relationship should exist.
        assert len(calls_rels) == 1


# ---------------------------------------------------------------------------
# resolve_call — self.method()
# ---------------------------------------------------------------------------


class TestResolveMethodCallSelf:
    """self.method() resolves within the same class."""

    def test_resolve_method_call_self(self) -> None:
        g = KnowledgeGraph()

        _add_file_node(g, "src/service.py")
        _add_symbol_node(
            g,
            SymbolNodeInfo(
                label=NodeLabel.CLASS,
                file_path="src/service.py",
                name="AuthService",
                start_line=1,
                end_line=30,
            ),
        )
        _add_symbol_node(
            g,
            SymbolNodeInfo(
                label=NodeLabel.METHOD,
                file_path="src/service.py",
                name="login",
                start_line=3,
                end_line=15,
                class_name="AuthService",
            ),
        )
        _add_symbol_node(
            g,
            SymbolNodeInfo(
                label=NodeLabel.METHOD,
                file_path="src/service.py",
                name="check_token",
                start_line=17,
                end_line=28,
                class_name="AuthService",
            ),
        )

        call = CallInfo(name="check_token", line=10, receiver="self")

        calls_obj = Calls([], g)
        calls_obj._file_path = "src/service.py"
        target_id, confidence = calls_obj._resolve_call(call)

        expected_id = generate_id(
            NodeLabel.METHOD,
            "src/service.py",
            "AuthService.check_token",
        )
        assert target_id == expected_id
        assert confidence == 1.0

    def test_resolve_method_call_this(self) -> None:
        """this.method() also resolves within the same class."""
        g = KnowledgeGraph()

        _add_file_node(g, "src/service.ts")
        _add_symbol_node(
            g,
            SymbolNodeInfo(
                label=NodeLabel.CLASS,
                file_path="src/service.ts",
                name="AuthService",
                start_line=1,
                end_line=30,
            ),
        )
        _add_symbol_node(
            g,
            SymbolNodeInfo(
                label=NodeLabel.METHOD,
                file_path="src/service.ts",
                name="checkToken",
                start_line=17,
                end_line=28,
                class_name="AuthService",
            ),
        )

        call = CallInfo(name="checkToken", line=10, receiver="this")

        calls_obj = Calls([], g)
        calls_obj._file_path = "src/service.ts"
        target_id, confidence = calls_obj._resolve_call(call)

        expected_id = generate_id(
            NodeLabel.METHOD,
            "src/service.ts",
            "AuthService.checkToken",
        )
        assert target_id == expected_id
        assert confidence == 1.0


# ---------------------------------------------------------------------------
# resolve_call — import-resolved
# ---------------------------------------------------------------------------


class TestResolveCallImportResolved:
    """Calls to imported symbols resolve with confidence 1.0."""

    def test_resolve_call_import_resolved(self) -> None:
        g = KnowledgeGraph()

        # Two files: app.py imports validate from auth.py.
        _add_file_node(g, "src/auth.py")
        _add_file_node(g, "src/app.py")

        _add_symbol_node(
            g,
            SymbolNodeInfo(
                label=NodeLabel.FUNCTION,
                file_path="src/auth.py",
                name="validate",
                start_line=1,
                end_line=10,
            ),
        )
        _add_symbol_node(
            g,
            SymbolNodeInfo(
                label=NodeLabel.FUNCTION,
                file_path="src/app.py",
                name="login",
                start_line=1,
                end_line=15,
            ),
        )

        # IMPORTS relationship: app.py -> auth.py with symbol "validate"
        app_file_id = generate_id(NodeLabel.FILE, "src/app.py")
        auth_file_id = generate_id(NodeLabel.FILE, "src/auth.py")
        g.add_relationship(
            GraphRelationship(
                id=f"imports:{app_file_id}->{auth_file_id}",
                type=RelType.IMPORTS,
                source=app_file_id,
                target=auth_file_id,
                properties={"symbols": "validate"},
            ),
        )

        call = CallInfo(name="validate", line=8)

        calls_obj = Calls([], g)
        calls_obj._file_path = "src/app.py"
        target_id, confidence = calls_obj._resolve_call(call)

        expected_id = generate_id(
            NodeLabel.FUNCTION,
            "src/auth.py",
            "validate",
        )
        assert target_id == expected_id
        assert confidence == 1.0


# ---------------------------------------------------------------------------
# Noise filtering — _CALL_BLOCKLIST
# ---------------------------------------------------------------------------


class TestCallBlocklist:
    """Calls to blocklisted names produce no CALLS edges."""

    def test_blocklist_is_frozenset(self) -> None:
        """_CALL_BLOCKLIST is a frozenset (immutable, O(1) membership)."""
        assert isinstance(Calls._CALL_BLOCKLIST, frozenset)

    def test_python_builtins_in_blocklist(self) -> None:
        """Common Python builtins are blocked."""
        for name in ("print", "len", "range", "isinstance", "super"):
            assert name in Calls._CALL_BLOCKLIST

    def test_js_globals_in_blocklist(self) -> None:
        """JS/TS built-ins are blocked."""
        for name in ("console", "setTimeout", "fetch", "JSON", "Promise"):
            assert name in Calls._CALL_BLOCKLIST

    def test_react_hooks_in_blocklist(self) -> None:
        """React hooks are blocked."""
        for name in ("useState", "useEffect", "useCallback", "useMemo"):
            assert name in Calls._CALL_BLOCKLIST

    def test_blocklisted_call_creates_no_edge(self) -> None:
        """A call to 'print' inside a function produces no CALLS edge."""
        g = KnowledgeGraph()
        _add_file_node(g, "src/main.py")
        _add_symbol_node(g, SymbolNodeInfo(NodeLabel.FUNCTION, "src/main.py", "do_work", 1, 10))

        parse_data = [
            FileParseData(
                file_path="src/main.py",
                language="python",
                parse_result=ParseResult(
                    calls=[CallInfo(name="print", line=5)],
                ),
            ),
        ]

        Calls(parse_data, g).process_calls()
        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        assert len(calls_rels) == 0

    def test_blocklisted_argument_creates_no_edge(self) -> None:
        """A blocklisted name passed as argument produces no CALLS edge."""
        g = KnowledgeGraph()
        _add_file_node(g, "src/main.py")
        _add_symbol_node(g, SymbolNodeInfo(NodeLabel.FUNCTION, "src/main.py", "do_work", 1, 10))

        parse_data = [
            FileParseData(
                file_path="src/main.py",
                language="python",
                parse_result=ParseResult(
                    calls=[
                        CallInfo(name="apply_func", line=5, arguments=["str"]),
                    ],
                ),
            ),
        ]

        Calls(parse_data, g).process_calls()
        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        # apply_func is not in the graph so no edge for it; 'str' is blocklisted.
        assert len(calls_rels) == 0

    def test_non_blocklisted_call_still_resolves(self) -> None:
        """User-defined function names pass through the blocklist filter."""
        g = KnowledgeGraph()
        _add_file_node(g, "src/main.py")
        _add_symbol_node(g, SymbolNodeInfo(NodeLabel.FUNCTION, "src/main.py", "caller", 1, 10))
        _add_symbol_node(g, SymbolNodeInfo(NodeLabel.FUNCTION, "src/main.py", "my_helper", 12, 20))

        parse_data = [
            FileParseData(
                file_path="src/main.py",
                language="python",
                parse_result=ParseResult(
                    calls=[CallInfo(name="my_helper", line=5)],
                ),
            ),
        ]

        Calls(parse_data, g).process_calls()
        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        assert len(calls_rels) == 1
