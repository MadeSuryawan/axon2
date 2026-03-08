"""
Watch mode for Axon — re-indexes on file changes.

Uses ``watchfiles`` (Rust-backed) for efficient file system monitoring with
native debouncing.  Changes are processed in tiers:

- **File-local** (immediate): Phases 2-7 on changed files only.
- **Global** (after quiet period): Hydrate graph from storage, run
  communities/processes/dead-code on the full graph.
- **Embeddings** (with global): Re-embed dirty nodes + CALLS neighbors.
- **Coupling** (commit-triggered): Re-run git coupling when HEAD changes.
"""

from asyncio import Event, Lock, to_thread
from collections.abc import Callable
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from subprocess import CalledProcessError
from subprocess import run as subprocess_run
from time import monotonic
from typing import Any, cast

from watchfiles import awatch

from axon.config.ignore import load_gitignore, should_ignore
from axon.config.languages import is_supported
from axon.core.embeddings.embedder import embed_nodes
from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import GraphNode, GraphRelationship, NodeLabel, RelType
from axon.core.ingestion.community import Community
from axon.core.ingestion.coupling import Coupling
from axon.core.ingestion.dead_code import DeadCode
from axon.core.ingestion.pipeline import reindex_files
from axon.core.ingestion.processes import Processes
from axon.core.ingestion.walker import FileEntry, read_file
from axon.core.storage.base import StorageBackend

logger = getLogger(__name__)


@dataclass(frozen=True)
class WatcherDeps:
    repo_path: Path
    storage: StorageBackend
    stop_event: Event | None = None
    lock: Lock | None = None


class Watcher:
    """
    Watcher for a single repository — re-indexes on file changes.

    Args:
        deps: Dependencies for the watcher (repo path, storage, etc.).
    """

    # Debounce: global phases fire after this many seconds of no file changes.
    _QUIET_PERIOD = 5.0

    # Maximum time dirty files can accumulate before forcing a global phase,
    # even if changes keep arriving (prevents starvation under continuous writes).
    _MAX_DIRTY_AGE = 60.0

    # How often watchfiles yields (controls quiet-period check granularity).
    _POLL_INTERVAL_MS = 500

    def __init__(self, deps: WatcherDeps) -> None:
        self._repo_path = deps.repo_path
        self._storage = deps.storage
        self._stop_event = deps.stop_event
        self._lock = deps.lock
        self._global_lock = Lock()
        self._gitignore = load_gitignore(self._repo_path)
        self._dirty_files: set[str] = set()
        self._last_change_time: float = 0.0
        self._first_dirty_time: float = 0.0
        self._git_command = ["git", "rev-parse", "HEAD"]
        self._last_known_commit = self._get_head_sha()
        self._changed_paths: list[Path] = []
        self._run_coupling = False
        self._ignored: set[str] = set()

    def _get_head_sha(self) -> str | None:
        """Return the current git HEAD sha, or None if not in a git repo."""
        try:
            result = subprocess_run(
                self._git_command,
                cwd=self._repo_path,
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (RuntimeError, CalledProcessError, FileNotFoundError):
            logger.debug("Failed to get git HEAD sha", exc_info=True)
        return None

    async def _run_sync(self, fn: Callable[..., Any]) -> tuple[int, set[str]] | None:
        """Run a synchronous function in a thread, optionally with a lock."""

        if self._lock:
            async with self._lock:
                return await to_thread(fn)
        return await to_thread(fn)

    async def watch(self) -> None:
        """Watch loop — monitor files and re-index on changes."""

        async for changes in awatch(
            self._repo_path,
            rust_timeout=self._POLL_INTERVAL_MS,
            yield_on_timeout=True,
            stop_event=self._stop_event,
        ):
            seen: set[str] = set()
            for _, path_str in changes:
                if path_str not in seen:
                    seen.add(path_str)
                    self._changed_paths.append(Path(path_str))

            # --- Tier 1: Immediate file-local reindex ---
            if self._changed_paths:
                # Copy and clear to avoid accumulation and re-processing
                batch = self._changed_paths.copy()
                self._changed_paths.clear()
                await self._reindex_changed_paths(batch)

            # --- Tier 2: Debounced global phases ---
            await self._debounce_global_phases()

        logger.info("Watch stopped.")

    async def _reindex_changed_paths(self, paths: list[Path]) -> None:
        """Re-index changed paths through file-local phases."""
        _index_result = await self._run_sync(lambda: self._reindex_files(paths))
        _index_result = cast(tuple[int, set[str]], _index_result)
        count, reindexed = _index_result
        if reindexed:
            self._dirty_files |= reindexed
            self._last_change_time = monotonic()
            if self._first_dirty_time == 0.0:
                self._first_dirty_time = self._last_change_time
            logger.info(f"Reindexed {count} file(s), {len(reindexed)} paths dirty")

    async def _debounce_global_phases(self) -> None:
        """Run global phases after a quiet period."""
        now = monotonic()
        quiet_elapsed = (
            self._last_change_time > 0 and (now - self._last_change_time) >= self._QUIET_PERIOD
        )
        starvation = (
            self._first_dirty_time > 0 and (now - self._first_dirty_time) >= self._MAX_DIRTY_AGE
        )
        if self._dirty_files and not self._global_lock.locked() and (quiet_elapsed or starvation):
            snapshot = self._dirty_files.copy()
            self._dirty_files.clear()
            self._first_dirty_time = 0.0

            current_commit = await to_thread(self._get_head_sha)
            self._run_coupling = current_commit != self._last_known_commit

            if self._run_coupling:
                self._last_known_commit = current_commit

            try:
                async with self._global_lock:
                    logger.info("Running incremental global phases...")
                    await self._run_sync(
                        self._run_incremental_global_phases,
                    )
            except Exception:
                logger.exception("Global phases failed; re-queueing dirty files")
                self._dirty_files |= snapshot
                self._last_change_time = monotonic()

    def _reindex_files(self, paths: list[Path]) -> tuple[int, set[str]]:
        """
        Re-index changed files through file-local phases.

        Returns (count_reindexed, set_of_relative_file_paths_reindexed).
        """

        entries: list[FileEntry] = []
        reindexed_paths: set[str] = set()

        for abs_path in paths:
            if not abs_path.is_file():
                try:
                    relative = str(abs_path.relative_to(self._repo_path))
                    self._storage.remove_nodes_by_file(relative)
                    reindexed_paths.add(relative)
                except (ValueError, OSError):
                    pass
                continue

            try:
                relative = str(abs_path.relative_to(self._repo_path))
            except ValueError:
                continue

            if relative in self._ignored or should_ignore(relative, self._gitignore):
                self._ignored.add(relative)
                continue

            if not is_supported(abs_path):
                continue

            if entry := read_file(self._repo_path, abs_path):
                entries.append(entry)
                reindexed_paths.add(relative)

        if entries:
            reindex_files(entries, self._repo_path, self._storage)

        return len(entries), reindexed_paths

    def _run_incremental_global_phases(
        self,
    ) -> None:
        """
        Run global phases incrementally using graph hydrated from storage.

        This function orchestrates the complete global analysis pipeline:
        1. Hydrates the graph from storage
        2. Analyzes graph structure (communities, processes, dead code)
        3. Persists synthetic entities discovered during analysis
        4. Updates dead code flags
        5. Optionally processes code coupling
        6. Updates embeddings for affected nodes
        7. Rebuilds full-text search indexes

        Args:
            storage: The storage backend containing the graph data.
            repo_path: Path to the repository root.
            dirty_files: Set of file paths that have been modified since last run.
            run_coupling: Whether to run the coupling analysis phase.

        Raises:
            Exceptions from storage operations are propagated to the caller.
        """
        graph = self._hydrate_graph()
        self._analyze_graph_structure(graph)
        nodes, relationships = self._collect_synthetic_entities(graph)
        self._persist_synthetic_entities(nodes, relationships)

        self._update_dead_code_flags(graph)

        if self._run_coupling:
            self._process_code_coupling(graph)

        self._update_embeddings_for_dirty_nodes(graph)

        self._storage.rebuild_fts_indexes()

        logger.info("Incremental global phases complete.")

    def _hydrate_graph(self) -> KnowledgeGraph:
        """
        Hydrate a KnowledgeGraph from the storage backend.

        This operation clears synthetic nodes first to ensure a clean state,
        then loads the complete graph including all nodes and relationships.
        """
        self._storage.delete_synthetic_nodes()
        logger.info("Hydrating graph from storage...")
        return self._storage.load_graph()

    def _analyze_graph_structure(self, graph: KnowledgeGraph) -> None:
        """
        Run graph analysis phases to detect communities, processes, and dead code.

        This function orchestrates three analysis passes over the graph:
        1. Community detection to find clusters of related code
        2. Process identification to detect workflow patterns
        3. Dead code analysis to find unused symbols

        Returns:
            A dictionary containing counts for each analysis type:
            - 'communities': Number of communities detected
            - 'processes': Number of processes identified
            - 'dead_code': Number of dead code symbols found
        """
        num_communities = Community(graph).process_communities()
        logger.info(f"Communities: {num_communities}")

        num_processes = Processes(graph).process_processes()
        logger.info(f"Processes: {num_processes}")

        num_dead = DeadCode(graph).process_dead_code()
        logger.info(f"Dead code: {num_dead}")

    def _collect_synthetic_entities(
        self,
        graph: KnowledgeGraph,
    ) -> tuple[list[GraphNode], list[GraphRelationship]]:
        """
        Collect synthetic entities (communities and processes) from the graph.

        Args:
            graph: The knowledge graph containing synthetic entities.

        Returns:
            A tuple of (nodes, relationships) where:
            - nodes: List of COMMUNITY and PROCESS nodes
            - relationships: List of MEMBER_OF and STEP_IN_PROCESS relationships
        """
        nodes: list[GraphNode] = []
        relationships: list[GraphRelationship] = []

        community_nodes = graph.get_nodes_by_label(NodeLabel.COMMUNITY)
        process_nodes = graph.get_nodes_by_label(NodeLabel.PROCESS)
        nodes.extend(community_nodes)
        nodes.extend(process_nodes)

        member_rels = graph.get_relationships_by_type(RelType.MEMBER_OF)
        step_rels = graph.get_relationships_by_type(RelType.STEP_IN_PROCESS)
        relationships.extend(member_rels)
        relationships.extend(step_rels)

        return nodes, relationships

    def _persist_synthetic_entities(
        self,
        nodes: list[GraphNode],
        relationships: list[GraphRelationship],
    ) -> None:
        """
        Persist synthetic entities to storage.

        Args:
            storage: The storage backend to save entities to.
            nodes: List of synthetic nodes to persist.
            relationships: List of synthetic relationships to persist.
        """
        if nodes:
            self._storage.add_nodes(nodes)
        if relationships:
            self._storage.add_relationships(relationships)

    def _update_dead_code_flags(self, graph: KnowledgeGraph) -> None:
        """
        Update dead code flags in storage based on current graph analysis.

        Args:
            storage: The storage backend to update.
            graph: The knowledge graph containing dead code annotations.
        """
        dead_ids = {n.id for n in graph.iter_nodes() if n.is_dead}
        alive_ids = {n.id for n in graph.iter_nodes() if not n.is_dead}
        self._storage.update_dead_flags(dead_ids, alive_ids)

    def _process_code_coupling(self, graph: KnowledgeGraph) -> int:
        """
        Process code coupling analysis and persist results.

        This function clears existing coupling relationships, runs the coupling
        analysis algorithm, and persists any new coupling relationships found.

        Args:
            storage: The storage backend to update.
            graph: The knowledge graph to analyze.
            repo_path: Path to the repository root.

        Returns:
            The number of coupling pairs detected.
        """
        self._storage.remove_relationships_by_type(RelType.COUPLED_WITH)
        num_coupled = Coupling(graph, self._repo_path).process_coupling()

        coupled_rels = list(graph.get_relationships_by_type(RelType.COUPLED_WITH))
        if coupled_rels:
            self._storage.add_relationships(coupled_rels)

        logger.info("Coupling: %d pairs", num_coupled)
        return num_coupled

    def _update_embeddings_for_dirty_nodes(
        self,
        graph: KnowledgeGraph,
    ) -> None:
        """
        Update embeddings for nodes affected by dirty files.

        This function identifies nodes in dirty files and their CALLS neighbors,
        then re-computes and updates their embeddings in storage.

        Args:
            storage: The storage backend to update embeddings in.
            graph: The knowledge graph containing nodes to embed.
            dirty_files: Set of file paths that have been modified.

        Note:
            Runtime errors during embedding are logged as warnings but not
            propagated, allowing the watch process to continue.
        """
        dirty_node_ids = self._compute_dirty_node_ids(graph)
        if not dirty_node_ids:
            return

        logger.info("Re-embedding %d nodes...", len(dirty_node_ids))
        try:
            embeddings = embed_nodes(graph, dirty_node_ids)
            if embeddings:
                self._storage.upsert_embeddings(embeddings)
        except RuntimeError:
            logger.warning("Incremental embedding failed", exc_info=True)

    def _compute_dirty_node_ids(self, graph: KnowledgeGraph) -> set[str]:
        """Find all node IDs in dirty files + their immediate CALLS neighbors."""
        if not self._dirty_files:
            return set()

        dirty_node_ids = {n.id for n in graph.iter_nodes() if n.file_path in self._dirty_files}

        neighbor_ids: set[str] = set()
        for node_id in dirty_node_ids:
            for rel in graph.get_outgoing(node_id, RelType.CALLS):
                neighbor_ids.add(rel.target)
            for rel in graph.get_incoming(node_id, RelType.CALLS):
                neighbor_ids.add(rel.source)

        return dirty_node_ids | neighbor_ids
