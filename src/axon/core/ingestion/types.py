"""
Phase 7: Type analysis for Axon.

Takes FileParseData from the parser phase and resolves type annotation
references to their corresponding Class, Interface, or TypeAlias nodes,
creating USES_TYPE relationships from Function/Method nodes to the resolved
type nodes.
"""

from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from os import cpu_count

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphRelationship,
    NodeLabel,
    RelType,
)
from axon.core.ingestion.parser_phase import FileParseData
from axon.core.ingestion.resolved import ResolvedEdge
from axon.core.ingestion.symbol_lookup import (
    build_file_symbol_index,
    build_name_index,
    find_containing_symbol,
)

logger = getLogger(__name__)

_TYPE_LABELS: tuple[NodeLabel, ...] = (
    NodeLabel.CLASS,
    NodeLabel.INTERFACE,
    NodeLabel.TYPE_ALIAS,
)

_CONTAINER_LABELS: tuple[NodeLabel, ...] = (
    NodeLabel.FUNCTION,
    NodeLabel.METHOD,
)


class Types:
    """Process type references and create USES_TYPE relationships in the graph."""

    def __init__(
        self,
        parse_data: list[FileParseData],
        graph: KnowledgeGraph,
        name_index: dict[str, list[str]],
    ) -> None:
        self._graph = graph
        self._parse_data = parse_data
        self._name_index = name_index if name_index else build_name_index(graph, _TYPE_LABELS)
        self._no_symbols: list[tuple[str, int, str]] = []
        self._file_sym_index = build_file_symbol_index(self._graph, _CONTAINER_LABELS)

    def process_types(
        self,
        *,
        parallel: bool = False,
        collect: bool = False,
    ) -> list[ResolvedEdge] | None:
        """
        Resolve type references and create USES_TYPE relationships in the graph.

        Args:
            parse_data: File parse results from the parser phase.
            graph: The knowledge graph to populate with USES_TYPE relationships.
            name_index: Optional pre-built name index; built automatically if None.
            parallel: When ``True``, resolve files in parallel using threads.
            collect: When ``True``, return flat list of edges instead of writing.
        """

        if parallel:
            workers = min(cpu_count() or 4, 8, len(self._parse_data))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                all_edges = list(pool.map(self._resolve_file_types, self._parse_data))
        else:
            all_edges = [self._resolve_file_types(fpd) for fpd in self._parse_data]

        flat = [edge for file_edges in all_edges for edge in file_edges]

        if collect:
            return flat

        self._add_type_ref_relationship(flat)
        logger.debug("type refs containing no symbols -> %d", len(self._no_symbols))

    def _resolve_file_types(self, fpd: FileParseData) -> list[ResolvedEdge]:
        """
        Resolve type references for a single file — pure read, no graph mutation.

        Returns one :class:`ResolvedEdge` per unique ``(source, target, role)``
        triple.  Per-file dedup via a local ``seen`` set.
        """
        seen: set[str] = set()
        edges: list[ResolvedEdge] = []

        for type_ref in fpd.parse_result.type_refs:
            source_id = find_containing_symbol(type_ref.line, fpd.file_path, self._file_sym_index)
            if source_id is None:
                self._no_symbols.append((type_ref.name, type_ref.line, fpd.file_path))
                continue

            if not (target_id := self._resolve_type(type_ref.name, fpd.file_path)):
                continue

            role = type_ref.kind
            rel_id = f"uses_type:{source_id}->{target_id}:{role}"
            if rel_id in seen:
                continue
            seen.add(rel_id)

            edges.append(
                ResolvedEdge(
                    rel_id=rel_id,
                    rel_type=RelType.USES_TYPE,
                    source=source_id,
                    target=target_id,
                    properties={"role": role},
                ),
            )
        return edges

    def _resolve_type(self, type_name: str, file_path: str) -> str | None:
        """
        Resolve a type name to a target node ID.

        Resolution strategy (tried in order):

        1. **Same-file match** -- the type is defined in the same file as the
           reference.
        2. **Global match** -- any type with this name anywhere in the codebase.
           If multiple matches exist the first one is returned.
        """
        candidate_ids = self._name_index.get(type_name, [])
        if not candidate_ids:
            return None

        for nid in candidate_ids:
            node = self._graph.get_node(nid)
            if node is not None and node.file_path == file_path:
                return nid

        return candidate_ids[0]

    def _add_type_ref_relationship(self, flat: list[ResolvedEdge]) -> None:
        """Create a deduplicated USES_TYPE relationship."""
        written: set[str] = set()
        for edge in flat:
            if edge.rel_id in written:
                continue
            written.add(edge.rel_id)
            self._graph.add_relationship(
                GraphRelationship(
                    id=edge.rel_id,
                    type=edge.rel_type,
                    source=edge.source,
                    target=edge.target,
                    properties=edge.properties,
                ),
            )
