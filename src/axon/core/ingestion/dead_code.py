"""
Phase 10: Dead code detection for Axon.

Scans the knowledge graph to find unreachable symbols (functions, methods,
classes) that have zero incoming CALLS relationships and are not entry points,
exported, constructors, test functions, or dunder methods.  Flags them by
setting ``is_dead = True`` on the corresponding graph node.
"""

from collections import defaultdict
from logging import getLogger
from pathlib import PurePosixPath
from typing import cast

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import GraphNode, NodeLabel, RelType

logger = getLogger(__name__)


class DeadCode:
    """
    Handles dead code detection for the Axon knowledge graph.

    Scans the graph to identify and flag unreachable symbols based on call relationships
    and specific exemptions (e.g., entry points, framework decorators).
    """

    _SYMBOL_LABELS: tuple[NodeLabel, ...] = (
        NodeLabel.FUNCTION,
        NodeLabel.METHOD,
        NodeLabel.CLASS,
    )

    _CONSTRUCTOR_NAMES: frozenset[str] = frozenset({"__init__", "__new__"})

    _NON_FRAMEWORK_DECORATORS: frozenset[str] = frozenset(
        {
            "functools.wraps",
            "functools.lru_cache",
            "functools.cached_property",
            "functools.cache",
        },
    )

    _FRAMEWORK_DECORATOR_NAMES: frozenset[str] = frozenset(
        {
            "task",
            "shared_task",
            "periodic_task",
            "job",
            "receiver",
            "on_event",
            "handler",
            "validator",
            "field_validator",
            "root_validator",
            "model_validator",
            "contextmanager",
            "asynccontextmanager",
            "fixture",
            "route",
            "endpoint",
            "command",
            "hybrid_property",
        },
    )

    _TYPING_STUB_DECORATORS: frozenset[str] = frozenset(
        {
            "overload",
            "typing.overload",
            "abstractmethod",
            "abc.abstractmethod",
        },
    )

    _ENUM_BASES: frozenset[str] = frozenset(
        {
            "Enum",
            "IntEnum",
            "StrEnum",
            "Flag",
            "IntFlag",
        },
    )

    def __init__(self, graph: KnowledgeGraph) -> None:
        """
        Initialize the DeadCode detector.

        Args:
            graph: The knowledge graph to analyze and mutate.
        """
        self._graph = graph
        self._methods: list[GraphNode] = []
        self._classes: list[GraphNode] = []
        self._un_flagged: dict[str, list[str]] = defaultdict(list)

    def process_dead_code(self) -> int:
        """
        Detect dead (unreachable) symbols and flag them in the graph.

        For each dead symbol the function sets ``node.is_dead = True``.

        Returns:
            The total number of symbols initially flagged as dead minus the false
            positives cleared in subsequent passes.
        """
        # Pre-cache nodes mapped by label to avoid repeated graph traversals
        # which heavily improves performance during false-positive resolution.
        self._methods = list(self._graph.get_nodes_by_label(NodeLabel.METHOD))
        self._classes = list(self._graph.get_nodes_by_label(NodeLabel.CLASS))

        dead_count = self._flag_initial_dead_nodes()

        # Second pass: un-flag overrides of called base-class methods.
        cleared_overrides = self._clear_override_false_positives()
        dead_count -= cleared_overrides

        # Third pass: un-flag methods on classes that structurally conform to a Protocol.
        protocol_cleared = self._clear_protocol_conformance_false_positives()
        dead_count -= protocol_cleared

        # Fourth pass: un-flag Protocol class stubs (interface contracts, never called directly).
        stub_cleared = self._clear_protocol_stub_false_positives()
        dead_count -= stub_cleared
        self._log_un_flagged()

        return dead_count

    def _log_un_flagged(self) -> None:
        for label, nodes in self._un_flagged.items():
            logger.debug("Un-flagged %s -> %d", label, len(nodes))

    def _flag_initial_dead_nodes(self) -> int:
        """Scan all eligible symbol nodes and flag those that appear structurally dead."""
        dead_count = 0
        for label in self._SYMBOL_LABELS:
            for node in self._graph.get_nodes_by_label(label):
                if not self._is_node_dead(node, label):
                    continue

                node.is_dead = True
                dead_count += 1
        return dead_count

    def _is_node_dead(self, node: GraphNode, label: NodeLabel) -> bool:
        """
        Determine if a single node should be considered dead based on structural graph rules.

        A symbol is considered dead when **all** of the following are true:
        1. It has zero incoming ``CALLS`` relationships.
        2. It is not exempted by naming conventions, test files, or public init files.
        3. It is not referenced dynamically via ``USES_TYPE``.
        4. It lacks specific decorators (framework, property, typing stubs).
        5. It is not an enum class.
        """
        is_exempt = self._is_exempt(
            name=node.name,
            is_entry_point=node.is_entry_point,
            is_exported=node.is_exported,
            file_path=node.file_path,
        )
        has_calls = self._graph.has_incoming(node.id, RelType.CALLS)
        has_type_ref = self._is_type_referenced(node.id, label)
        has_decorators = (
            self._has_framework_decorator(node)
            or self._has_property_decorator(node)
            or self._has_typing_stub_decorator(node)
        )

        return not (
            is_exempt
            or has_calls
            or has_type_ref
            or has_decorators
            or self._is_enum_class(node, label)
        )

    def _is_exempt(
        self,
        name: str,
        *,
        is_entry_point: bool,
        is_exported: bool,
        file_path: str = "",
    ) -> bool:
        """Return ``True`` if the symbol is exempt from dead-code flagging."""
        return (
            is_entry_point
            or is_exported
            or name in self._CONSTRUCTOR_NAMES
            or name.startswith("test_")
            or self._is_test_class(name)
            or self._is_test_file(file_path)
            or self._is_dunder(name)
            or self._is_python_public_api(name, file_path)
        )

    def _is_test_class(self, name: str) -> bool:
        """Return ``True`` if *name* follows pytest class convention (``Test*``)."""
        return len(name) > 4 and name.startswith("Test") and name[4].isupper()

    def _is_test_file(self, file_path: str) -> bool:
        """Return ``True`` if the file is in a test directory or is a test file."""
        parts = PurePosixPath(file_path).parts
        return (
            "tests" in parts
            or "test" in parts
            or any(p.startswith("test_") for p in parts)
            or file_path.endswith("conftest.py")
        )

    def _is_dunder(self, name: str) -> bool:
        """Return ``True`` if *name* is a dunder (double-underscore) method."""
        return name.startswith("__") and name.endswith("__") and len(name) > 4

    def _is_python_public_api(self, name: str, file_path: str) -> bool:
        """Return ``True`` if *name* is a public symbol in an ``__init__.py`` file."""
        return file_path.endswith("__init__.py") and not name.startswith("_")

    def _is_type_referenced(self, node_id: str, label: NodeLabel) -> bool:
        """Return ``True`` if *node_id* is a class with incoming USES_TYPE edges."""
        if label != NodeLabel.CLASS:
            return False
        return self._graph.has_incoming(node_id, RelType.USES_TYPE)

    def _has_framework_decorator(self, node: GraphNode) -> bool:
        """Return ``True`` if *node* has a framework decorator (dotted or undotted)."""
        decorators: list[str] = cast(list[str], node.properties.get("decorators", []))
        return any(
            dec in self._FRAMEWORK_DECORATOR_NAMES
            or ("." in dec and dec not in self._NON_FRAMEWORK_DECORATORS)
            for dec in decorators
        )

    def _has_property_decorator(self, node: GraphNode) -> bool:
        """Return ``True`` if *node* is a ``@property`` (accessed as attribute, not called)."""
        decorators: list[str] = cast(list[str], node.properties.get("decorators", []))
        return "property" in decorators

    def _has_typing_stub_decorator(self, node: GraphNode) -> bool:
        """Return ``True`` if *node* is an ``@overload`` or ``@abstractmethod`` stub."""
        decorators: list[str] = cast(list[str], node.properties.get("decorators", []))
        return any(d in self._TYPING_STUB_DECORATORS for d in decorators)

    def _is_enum_class(self, node: GraphNode, label: NodeLabel) -> bool:
        """Return ``True`` if *node* is an enum class (members accessed via dot, not called)."""
        if label != NodeLabel.CLASS:
            return False
        bases: list[str] = cast(list[str], node.properties.get("bases", []))
        return bool(self._ENUM_BASES & set(bases))

    def _clear_override_false_positives(self) -> int:
        """
        Un-flag methods that override a non-dead base class method.

        When ``A extends B`` and ``B.method`` is called, ``A.method`` (the
        override) has zero incoming CALLS and gets flagged dead.  This pass
        detects that situation and clears ``is_dead`` on the override.
        """
        # Build mapping: class_name -> set of method names that are NOT dead.
        alive_methods_by_class: dict[str, set[str]] = {}
        for method in self._methods:
            if not method.is_dead and method.class_name:
                alive_methods_by_class.setdefault(method.class_name, set()).add(method.name)

        # Build map of child -> parent class mapping from EXTENDS relationships.
        child_to_parents: dict[str, list[str]] = {}
        for rel in self._graph.get_relationships_by_type(RelType.EXTENDS):
            child_node = self._graph.get_node(rel.source)
            parent_node = self._graph.get_node(rel.target)
            if child_node and parent_node:
                child_to_parents.setdefault(child_node.name, []).append(parent_node.name)

        cleared = 0
        for method in self._methods:
            if not method.is_dead or not method.class_name:
                continue

            parent_classes = child_to_parents.get(method.class_name, [])
            for parent_name in parent_classes:
                alive_in_parent = alive_methods_by_class.get(parent_name, set())
                if method.name in alive_in_parent:
                    method.is_dead = False
                    cleared += 1
                    # logger.debug("Un-flagged override: %s.%s", method.class_name, method.name)
                    self._un_flagged["override"].append(f"{method.class_name}.{method.name}")
                    break

        return cleared

    def _clear_protocol_conformance_false_positives(self) -> int:
        """Un-flag methods on classes that structurally conform to a Protocol."""
        protocol_methods = self._get_protocol_methods()
        if not protocol_methods:
            return 0

        class_methods = self._get_class_methods()
        clearable = self._get_clearable_methods(protocol_methods, class_methods)
        if not clearable:
            return 0

        cleared = 0
        for method in self._methods:
            if method.is_dead and method.class_name:
                names_to_clear = clearable.get(method.class_name)
                if names_to_clear and method.name in names_to_clear:
                    method.is_dead = False
                    cleared += 1
                    # logger.debug(
                    #     "Un-flagged protocol conformance: %s.%s",
                    #     method.class_name,
                    #     method.name,
                    # )
                    self._un_flagged["protocol conformance"].append(
                        f"{method.class_name}.{method.name}",
                    )

        return cleared

    def _get_protocol_methods(self) -> dict[str, set[str]]:
        protocol_methods: dict[str, set[str]] = {}
        for cls_node in self._classes:
            if cls_node.properties.get("is_protocol"):
                methods = {
                    m.name
                    for m in self._methods
                    if m.class_name == cls_node.name and not self._is_dunder(m.name)
                }
                if methods:
                    protocol_methods[cls_node.name] = methods
        return protocol_methods

    def _get_class_methods(self) -> dict[str, set[str]]:
        class_methods: dict[str, set[str]] = {}
        for method in self._methods:
            if method.class_name:
                class_methods.setdefault(method.class_name, set()).add(method.name)
        return class_methods

    def _get_clearable_methods(
        self,
        protocol_methods: dict[str, set[str]],
        class_methods: dict[str, set[str]],
    ) -> dict[str, set[str]]:
        clearable: dict[str, set[str]] = {}
        for proto_name, required in protocol_methods.items():
            for cls_name, methods in class_methods.items():
                if cls_name != proto_name and required <= methods:
                    clearable.setdefault(cls_name, set()).update(required)
        return clearable

    def _clear_protocol_stub_false_positives(self) -> int:
        """
        Un-flag methods on Protocol classes.

        Protocol stubs define the interface contract — they are never called
        directly (calls resolve to concrete implementations).  Flagging them
        as dead is always a false positive.
        """
        protocol_class_names: set[str] = {
            cls_node.name for cls_node in self._classes if cls_node.properties.get("is_protocol")
        }

        if not protocol_class_names:
            return 0

        cleared = 0
        for method in self._methods:
            if not method.is_dead or not method.class_name:
                continue
            if method.class_name in protocol_class_names:
                method.is_dead = False
                cleared += 1
                # logger.debug("Un-flagged protocol stub: %s.%s", method.class_name, method.name)
                self._un_flagged["protocol stub"].append(f"{method.class_name}.{method.name}")

        return cleared
