"""
Phase 6: Heritage extraction for Axon.

Takes FileParseData from the parser phase and creates EXTENDS / IMPLEMENTS
relationships between Class and Interface nodes in the knowledge graph.

Heritage tuples have the shape ``(class_name, kind, parent_name)`` where
*kind* is either ``"extends"`` or ``"implements"``.

The main entry point is :meth:`Heritage.process_heritage`, which resolves
class/interface names to graph nodes and creates the appropriate inheritance
relationships.
"""

from collections import defaultdict
from logging import getLogger

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphRelationship,
    NodeLabel,
    RelType,
)
from axon.core.ingestion.parser_phase import FileParseData
from axon.core.ingestion.symbol_lookup import build_name_index

logger = getLogger(__name__)


class Heritage:
    """
    Handles heritage (inheritance) extraction for the Axon knowledge graph.

    Scans parse data for EXTENDS and IMPLEMENTS relationships between classes
    and interfaces, resolves symbol names to graph nodes, and creates the
    corresponding relationships in the knowledge graph.

    This analyzer supports:
    - Python class inheritance (extends)
    - Python interface implementation (implements)
    - TypeScript extends/implements
    - Protocol/ABC marker annotation for dead-code detection
    """

    # Node labels that can participate in heritage relationships
    _HERITAGE_LABELS: tuple[NodeLabel, ...] = (
        NodeLabel.CLASS,
        NodeLabel.INTERFACE,
    )

    # Mapping from heritage kind string to relationship type
    _KIND_TO_REL: dict[str, RelType] = {
        "extends": RelType.EXTENDS,
        "implements": RelType.IMPLEMENTS,
    }

    # Marker names that indicate a class is acting as a protocol/ABC
    # These allow dead-code detection to leverage structural subtyping
    _PROTOCOL_MARKERS: frozenset[str] = frozenset({"Protocol", "ABC", "ABCMeta"})

    def __init__(
        self,
        graph: KnowledgeGraph,
        parse_data: list[FileParseData],
    ) -> None:
        """
        Initialize the Heritage analyzer.

        Args:
            graph: The knowledge graph to analyze and enrich with heritage
                   relationships.
            parse_data: File parse results produced by the parser phase,
                        containing heritage tuples.
        """
        self._graph = graph
        self._parse_data = parse_data
        self._symbol_index: dict[str, list[str]] = {}
        self._skipping: dict[str, list[str]] = defaultdict(list)

    def process_heritage(self) -> None:
        """
        Create EXTENDS and IMPLEMENTS relationships from heritage tuples.

        This is the main entry point. It:
        1. Builds a symbol index for efficient name resolution.
        2. Processes each heritage tuple from the parse data.
        3. Resolves child and parent class/interface names to node IDs.
        4. Creates the appropriate relationship in the graph.
        5. Handles protocol/ABC marker annotation for unresolved parents.

        If either the child or parent node cannot be resolved (e.g., an
        external class not present in the graph), the tuple is silently
        skipped unless it's a known protocol marker.
        """
        # Step 1: Build symbol index for fast name resolution
        # This maps class/interface names to their node IDs
        self._symbol_index = self._build_symbol_index()

        # Step 2: Process each file's parse data
        for file_parse_data in self._parse_data:
            self._process_file_heritage(file_parse_data)

        self._log_skipping()

    def _log_skipping(self) -> None:
        for reason, data in self._skipping.items():
            logger.debug("Skipping heritage %s -> %d", reason, len(data))

    def _build_symbol_index(self) -> dict[str, list[str]]:
        """
        Build an index mapping class/interface names to node IDs.

        The index is used for fast symbol resolution when processing
        heritage tuples. Only Class and Interface nodes are included.

        Returns:
            A dictionary mapping symbol names to lists of node IDs.
        """
        return build_name_index(self._graph, self._HERITAGE_LABELS)

    def _process_file_heritage(self, file_parse_data: FileParseData) -> None:
        """
        Process all heritage tuples for a single file.

        Args:
            file_parse_data: Parse data for a single file.
        """
        # Iterate through each heritage tuple in this file
        for class_name, kind, parent_name in file_parse_data.parse_result.heritage:
            self._process_heritage_tuple(
                class_name=class_name,
                kind=kind,
                parent_name=parent_name,
                file_path=file_parse_data.file_path,
            )

    def _process_heritage_tuple(
        self,
        class_name: str,
        kind: str,
        parent_name: str,
        file_path: str,
    ) -> None:
        """
        Process a single heritage tuple and create a relationship if possible.

        Args:
            class_name: Name of the child class/interface.
            kind: Either "extends" or "implements".
            parent_name: Name of the parent class/interface.
            file_path: Path to the file containing this heritage declaration.
        """
        # Validate the heritage kind
        rel_type = self._KIND_TO_REL.get(kind)
        if rel_type is None:
            # logger.warning(
            #     "Unknown heritage kind %r for %s in %s, skipping",
            #     kind,
            #     class_name,
            #     file_path,
            # )
            data = f"Unknown heritage kind {kind} for {class_name} in {file_path}, skipping"
            self._skipping["unknown kind"].append(data)
            return

        # Resolve both child and parent to node IDs
        child_id = self._resolve_node(class_name, file_path)
        parent_id = self._resolve_node(parent_name, file_path)

        # Handle unresolved child - cannot create relationship without it
        if child_id is None:
            # logger.debug(
            #     "Skipping heritage %s %s %s in %s: unresolved child",
            #     class_name,
            #     kind,
            #     parent_name,
            #     file_path,
            # )
            data = f"Skipping heritage {class_name} {kind} {parent_name} in {file_path}: unresolved child"
            self._skipping["unresolved child"].append(data)
            return

        # Handle unresolved parent
        if parent_id is None:
            self._handle_unresolved_parent(
                child_id=child_id,
                class_name=class_name,
                kind=kind,
                parent_name=parent_name,
                file_path=file_path,
            )
            return

        # Both resolved - create the heritage relationship
        self._create_heritage_relationship(
            child_id=child_id,
            parent_id=parent_id,
            rel_type=rel_type,
            kind=kind,
        )

    def _resolve_node(
        self,
        name: str,
        file_path: str,
    ) -> str | None:
        """
        Resolve a symbol *name* to a node ID, preferring same-file matches.

        This method implements a smart resolution strategy:
        1. First, check if the name exists in the symbol index.
        2. Prefer any candidate defined in the same file (most likely match).
        3. Fall back to the first candidate for cross-file references.

        Args:
            name: The symbol name to resolve.
            file_path: The file path where this symbol is referenced.

        Returns:
            The node ID if resolved, otherwise ``None``.
        """
        # Look up candidates in the symbol index
        candidate_ids = self._symbol_index.get(name)
        if not candidate_ids:
            return None

        # Prefer same-file matches (most likely to be the correct reference)
        for node_id in candidate_ids:
            node = self._graph.get_node(node_id)
            if node is not None and node.file_path == file_path:
                return node_id

        # Fall back to first candidate (cross-file reference)
        return candidate_ids[0]

    def _handle_unresolved_parent(
        self,
        child_id: str,
        class_name: str,
        kind: str,
        parent_name: str,
        file_path: str,
    ) -> None:
        """
        Handle the case where a parent class cannot be resolved.

        If the parent is a known protocol/ABC marker (Protocol, ABC, ABCMeta),
        annotate the child node so dead-code detection can leverage structural
        subtyping later. Otherwise, silently skip the relationship.

        Args:
            child_id: Node ID of the child class.
            class_name: Name of the child class.
            kind: Either "extends" or "implements".
            parent_name: Name of the unresolved parent.
            file_path: Path to the file containing this heritage declaration.
        """
        # Check if parent is a known protocol/ABC marker
        if parent_name in self._PROTOCOL_MARKERS:
            child_node = self._graph.get_node(child_id)
            if child_node is not None:
                # Annotate the child as a protocol for dead-code detection
                child_node.properties["is_protocol"] = True
                logger.debug(
                    "Annotated %s as protocol in %s (parent: %s)",
                    class_name,
                    file_path,
                    parent_name,
                )
        else:
            # External parent class - skip silently
            # logger.debug(
            #     "Skipping heritage %s %s %s in %s: unresolved parent",
            #     class_name,
            #     kind,
            #     parent_name,
            #     file_path,
            # )
            data = f"Skipping heritage {class_name} {kind} {parent_name} in {file_path}: unresolved parent"
            self._skipping["unresolved parent"].append(data)

    def _create_heritage_relationship(
        self,
        child_id: str,
        parent_id: str,
        rel_type: RelType,
        kind: str,
    ) -> None:
        """
        Create an EXTENDS or IMPLEMENTS relationship in the graph.

        Args:
            child_id: Node ID of the child class/interface.
            parent_id: Node ID of the parent class/interface.
            rel_type: The relationship type (EXTENDS or IMPLEMENTS).
            kind: The kind string for relationship ID generation.
        """
        # Generate unique relationship ID: "kind:child_id->parent_id"
        rel_id = f"{kind}:{child_id}->{parent_id}"

        # Create and add the relationship to the graph
        relationship = GraphRelationship(
            id=rel_id,
            type=rel_type,
            source=child_id,
            target=parent_id,
        )
        self._graph.add_relationship(relationship)
