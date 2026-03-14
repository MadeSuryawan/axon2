"""
Pipeline orchestrator for Axon.

Runs all ingestion phases in sequence, populates an in-memory knowledge graph,
bulk-loads it into a storage backend, and returns a summary of the results.

Phases executed:
    0. Incremental diff (reserved -- not yet implemented)
    1. File walking
    2. Structure processing (File/Folder nodes + CONTAINS edges)
    3. Code parsing (symbol nodes + DEFINES edges)
    4. Import resolution (IMPORTS edges)
    5. Call tracing (CALLS edges)
    6. Heritage extraction (EXTENDS / IMPLEMENTS edges)
    7. Type analysis (USES_TYPE edges)
    8. Community detection (COMMUNITY nodes + MEMBER_OF edges)
    9. Process detection (PROCESS nodes + STEP_IN_PROCESS edges)
    10. Dead code detection (flags unreachable symbols)
    11. Change coupling (COUPLED_WITH edges from git history)
"""

from collections.abc import Callable
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from time import monotonic
from typing import Any

from rich import print as rprint

from axon.config.ignore import load_gitignore
from axon.core.embeddings.embedder import embed_graph
from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import GraphRelationship, NodeLabel
from axon.core.ingestion.calls import Calls
from axon.core.ingestion.community import Community
from axon.core.ingestion.coupling import Coupling
from axon.core.ingestion.dead_code import DeadCode
from axon.core.ingestion.heritage import Heritage
from axon.core.ingestion.imports import Imports
from axon.core.ingestion.parser_phase import FileParseData, Parsing
from axon.core.ingestion.processes import Processes
from axon.core.ingestion.structure import Structure
from axon.core.ingestion.symbol_lookup import build_name_index
from axon.core.ingestion.types import process_types
from axon.core.ingestion.walker import FileEntry, walk_repo
from axon.core.storage.base import StorageBackend

logger = getLogger(__name__)


@dataclass
class PipelineResult:
    """Summary of a pipeline run."""

    files: int = 0
    symbols: int = 0
    relationships: int = 0
    clusters: int = 0
    processes: int = 0
    dead_code: int = 0
    coupled_pairs: int = 0
    embeddings: int = 0
    duration_seconds: float = 0.0
    incremental: bool = False
    changed_files: int = 0


_SYMBOL_LABELS: frozenset[NodeLabel] = frozenset(NodeLabel) - {
    NodeLabel.FILE,
    NodeLabel.FOLDER,
    NodeLabel.COMMUNITY,
    NodeLabel.PROCESS,
}


class Pipelines:
    """
    Orchestrates the execution of the ingestion pipeline.

    Args:
        repo_path: Root directory of the repository to analyze.
        storage: An already-initialized :class:`StorageBackend` to persist the graph.
        full: When ``True``, skip incremental-diff logic (Phase 0) and force a full
            re-index.  Currently Phase 0 is a no-op regardless of this flag.
        embeddings: When ``True`` (default), generate and store vector embeddings after
            bulk-loading.
    """

    def __init__(
        self,
        repo_path: Path,
        storage: StorageBackend | None = None,
        *,
        full: bool = False,
        embeddings: bool = True,
    ) -> None:
        self._repo_path = repo_path
        self._storage = storage
        self._full = full
        self._embeddings = embeddings
        self._gitignore: list[str] = []
        self._files: list[FileEntry] = []
        self._graph = KnowledgeGraph()
        self._result = PipelineResult()
        self._parsed_data: list[FileParseData] = []

    @property
    def graph(self) -> KnowledgeGraph:
        return self._graph

    @property
    def result(self) -> PipelineResult:
        return self._result

    def _phases(self) -> dict[str, Callable[[], None | Any]]:

        return {
            "Processing structure": lambda: Structure(self._graph).process_structure(self._files),
            "Parsing code": lambda: setattr(
                self,
                "_parsed_data",
                Parsing(self._graph).process_parsing(self._files),
            ),
            "Resolving imports": lambda: Imports(self._graph, self._parsed_data).process_imports(
                parallel=True,
            ),
            "Tracing calls": lambda: Calls(self._parsed_data, self._graph).process_calls(),
            "Extracting heritage": lambda: Heritage(
                self._graph,
                self._parsed_data,
            ).process_heritage(),
            "Analyzing types": lambda: process_types(self._parsed_data, self._graph),
            "Detecting communities": lambda: setattr(
                self._result,
                "clusters",
                Community(self._graph).process_communities(),
            ),
            "Detecting execution flows": lambda: setattr(
                self._result,
                "processes",
                Processes(self._graph).process_processes(),
            ),
            "Finding dead code": lambda: setattr(
                self._result,
                "dead_code",
                DeadCode(self._graph).process_dead_code(),
            ),
            "Analyzing git history": lambda: setattr(
                self._result,
                "coupled_pairs",
                Coupling(self._graph, self._repo_path).process_coupling(),
            ),
        }

    def run_pipelines(self) -> None:
        """
        Run phases 1-11 of the ingestion pipeline.

        When *storage* is provided the graph is bulk-loaded into it after
        all phases complete.  When ``None``, only the in-memory graph is
        returned (useful for branch comparison snapshots).
        """
        rprint("[b green]Running pipelines :\n")
        start = monotonic()
        self._walk_files()

        for phase, pipeline in self._phases().items():
            rprint(f"[b blue]{phase}..."), pipeline()

        self._update_rest_result()
        self._result.duration_seconds = monotonic() - start

    def _walk_files(self) -> None:
        rprint("[b blue]Walking files...")
        self._gitignore = load_gitignore(self._repo_path)
        self._files = walk_repo(self._repo_path, self._gitignore)
        self._result.files = len(self._files)

    def _update_rest_result(self) -> None:
        self._result.symbols = sum(1 for n in self._graph.iter_nodes() if n.label in _SYMBOL_LABELS)
        self._result.relationships = self._graph.relationship_count

        if not self._storage:
            return

        rprint("\n[b blue]4 Stages loading to storage...")
        self._storage.bulk_load(self._graph)

        if not self._embeddings:
            rprint("\n[b yellow]Embedding disabled — search will use FTS only")
            return

        try:
            rprint("\n[b blue]Generating embeddings...")
            node_embeddings = embed_graph(self._graph)
            self._storage.store_embeddings(node_embeddings)
            self._result.embeddings = len(node_embeddings)
        except (RuntimeError, ValueError, OSError, SystemError):
            logger.warning(
                "Embedding phase failed — search will use FTS only",
                exc_info=True,
            )

    def build_graph(self) -> KnowledgeGraph:
        """
        Run phases 1-11 and return the in-memory graph (no storage load).

        This is used by branch comparison to build a graph snapshot without
        needing a storage backend.
        """
        self.run_pipelines()
        return self.graph


def reindex_files(
    file_entries: list[FileEntry],
    storage: StorageBackend,
    *,
    rebuild_fts: bool = True,
) -> KnowledgeGraph:
    """
    Re-index specific files through phases 2-7 (file-local phases).

    Removes old nodes for these files from storage, re-parses them,
    and inserts updated nodes/relationships. Returns the partial graph
    for further processing (global phases, embeddings).

    Parameters
    ----------
    file_entries:
        The files to re-index (already read from disk).
    repo_path:
        Root directory of the repository.
    storage:
        An already-initialised storage backend.
    rebuild_fts:
        Wether to rebuild the FTS or not

    Returns
    -------
    KnowledgeGraph
        The partial in-memory graph containing only the reindexed files.
    """
    # DETACH DELETE drops inbound edges from unchanged files — save them
    # before deletion and re-insert after rebuild.
    changed_files = {entry.path for entry in file_entries}
    saved_edges = _collect_saved_edges(changed_files, storage, file_entries)

    graph = storage.load_graph()
    Structure(graph).process_structure(file_entries)

    parse_data = Parsing(graph).process_parsing(file_entries)
    shared_name_index, heritage_name_index = _get_heritage_name_idx(graph)

    Imports(graph, parse_data).process_imports()
    Calls(parse_data, graph).process_calls()
    Heritage(graph, parse_data).process_heritage()
    process_types(parse_data, graph)
    _update_storage(graph, storage, changed_files, saved_edges, rebuild_fts=rebuild_fts)

    return graph


def _collect_saved_edges(
    changed_files: set[str],
    storage: StorageBackend,
    file_entries: list[FileEntry],
) -> list[GraphRelationship]:

    saved_edges = []
    for fp in changed_files:
        saved_edges.extend(
            storage.get_inbound_cross_file_edges(fp, exclude_source_files=changed_files),
        )

    for entry in file_entries:
        storage.remove_nodes_by_file(entry.path)

    return saved_edges


def _get_heritage_name_idx(graph: KnowledgeGraph) -> tuple[dict[str, list[str]], ...]:
    shared_labels = (
        NodeLabel.FUNCTION,
        NodeLabel.METHOD,
        NodeLabel.CLASS,
        NodeLabel.INTERFACE,
        NodeLabel.TYPE_ALIAS,
    )
    shared_name_index = build_name_index(graph, shared_labels)
    heritage_labels = {NodeLabel.CLASS, NodeLabel.INTERFACE}
    heritage_name_index: dict[str, list[str]] = {}
    for name, ids in shared_name_index.items():
        filtered = [nid for nid in ids if (n := graph.get_node(nid)) and n.label in heritage_labels]
        heritage_name_index.update({name: filtered}) if filtered else None

    return shared_name_index, heritage_name_index


def _update_storage(
    graph: KnowledgeGraph,
    storage: StorageBackend,
    changed_files: set[str],
    saved_edges: list[GraphRelationship],
    *,
    rebuild_fts: bool,
) -> None:

    incremental_nodes = [
        node
        for node in graph.iter_nodes()
        if (
            node.file_path in changed_files
            or (
                node.label == NodeLabel.FOLDER
                and any(
                    file_path == node.file_path or file_path.startswith(f"{node.file_path}/")
                    for file_path in changed_files
                )
            )
        )
    ]
    incremental_node_ids = {node.id for node in incremental_nodes}
    incremental_relationships = [
        rel
        for rel in graph.iter_relationships()
        if rel.source in incremental_node_ids or rel.target in incremental_node_ids
    ]

    storage.add_nodes(incremental_nodes)
    storage.add_relationships(incremental_relationships)

    if saved_edges:
        storage.add_relationships(saved_edges)

    if rebuild_fts:
        storage.rebuild_fts_indexes()
