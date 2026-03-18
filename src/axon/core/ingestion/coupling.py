"""
Phase 11: Change Coupling Analysis for Axon.

Analyzes git history to discover files that frequently change together.
Co-change frequency is a strong indicator of logical coupling -- files that
must be modified in tandem likely share implicit dependencies that may not
be visible in static analysis alone.

The main entry point is :meth:`Coupling.process_coupling`, which parses the git log,
builds a co-change matrix, computes coupling strengths, and writes
``COUPLED_WITH`` relationships into the knowledge graph.
"""

from collections import defaultdict
from itertools import combinations
from logging import getLogger
from pathlib import Path
from subprocess import CalledProcessError, TimeoutExpired
from subprocess import run as subprocess_run

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelType,
)
from axon.core.ingestion.resolved import ResolvedEdge

logger = getLogger(__name__)


class Coupling:
    """
    Handles change coupling analysis for the Axon knowledge graph.

    Scans git history to identify files that frequently change together,
    computes coupling strength, and creates ``COUPLED_WITH`` relationships
    between highly coupled file nodes in the graph.
    """

    # Default threshold for minimum co-changes to include a pair
    _DEFAULT_MIN_COCHANGES: int = 1

    # Maximum files per commit to consider (skip merge commits, bulk reformats)
    _DEFAULT_MAX_FILES_PER_COMMIT: int = 50

    def __init__(
        self,
        graph: KnowledgeGraph,
        repo_path: Path,
        file_nodes: list[GraphNode] | None = None,
        min_cochanges: int = 3,
        min_strength: float = 0.3,
    ) -> None:
        """
        Initialize the Coupling analyzer.

        Args:
            graph: The knowledge graph containing File nodes.
            repo_path: Root of the git repository to analyze.
            file_nodes: List of file nodes to analyze.
                When ``None``, all ``File`` nodes in the graph are used.
            min_cochanges: Minimum co-change count to include a pair.
            min_strength: Minimum coupling strength to create a relationship.
        """
        self._graph = graph
        self._repo_path = repo_path
        self._file_nodes: list[GraphNode] = file_nodes or self._graph.get_nodes_by_label(
            NodeLabel.FILE,
        )
        self._graph_files: set[str] = {n.file_path for n in self._file_nodes}
        self._path_to_id: dict[str, str] = {n.file_path: n.id for n in self._file_nodes}
        self._min_cochanges: int = min_cochanges
        self._min_strength: float = min_strength

    def resolve_coupling(self, commits: list[list[str]] | None = None) -> list[ResolvedEdge]:
        """
        Analyze git history and create ``COUPLED_WITH`` relationships.

        Parses the git log (or uses pre-supplied *commits* for testing),
        computes pairwise coupling strengths, and returns a list of
        ``ResolvedEdge`` objects for pairs that exceed *min_strength*.

        Args:
            min_strength: Minimum coupling strength to create a relationship.
            commits: Pre-parsed commit data. When provided, git log parsing
                is skipped — useful for deterministic testing.
            min_cochanges: Minimum number of co-changes to consider a pair.

        Returns:
            A list of ``ResolvedEdge`` objects representing ``COUPLED_WITH``
            relationships.
        """
        if not commits:
            commits = self._parse_git_log()

        cochange, total_changes = self._build_cochange_matrix(commits)

        edges: list[ResolvedEdge] = []
        for (file_a, file_b), co_changes in cochange.items():
            strength = self._calculate_coupling(file_a, file_b, co_changes, total_changes)
            if strength < self._min_strength:
                continue

            id_a = self._path_to_id.get(file_a)
            id_b = self._path_to_id.get(file_b)
            if id_a is None or id_b is None:
                continue

            rel_id = f"coupled:{id_a}->{id_b}"
            edges.append(
                ResolvedEdge(
                    rel_id=rel_id,
                    rel_type=RelType.COUPLED_WITH,
                    source=id_a,
                    target=id_b,
                    properties={"strength": strength, "co_changes": co_changes},
                ),
            )

        logger.info("Created %d COUPLED_WITH relationships", len(edges))
        return edges

    def _build_cochange_matrix(
        self,
        commits: list[list[str]],
        max_files_per_commit: int = 50,
    ) -> tuple[dict[tuple[str, str], int], dict[str, int]]:
        """
        Build a co-change frequency matrix and per-file change counts.

        For every pair of files that appear in the same commit, their co-change
        count is incremented.  Only pairs whose count meets or exceeds
        *min_cochanges* are retained.

        Commits touching more than *max_files_per_commit* files are skipped
        (merge commits, bulk reformats) to avoid O(n^2) pair explosion and
        coupling noise.

        Keys are *sorted* tuples ``(file_a, file_b)`` so that ``(A, B)`` and
        ``(B, A)`` map to the same entry.

        Args:
            commits: List of commits, each a list of changed file paths.
            min_cochanges: Minimum co-change count to keep a pair.
            max_files_per_commit: Skip commits with more files than this.

        Returns:
            A tuple of (cochange_matrix, total_changes) where cochange_matrix
            maps ``(file_a, file_b)`` sorted tuples to their count, and
            total_changes maps each file to its total commit count.
        """
        counts: dict[tuple[str, str], int] = defaultdict(int)
        total_changes: dict[str, int] = defaultdict(int)

        for files in commits:
            unique_files = sorted(set(files))
            for f in unique_files:
                total_changes[f] += 1
            if len(unique_files) > max_files_per_commit:
                continue
            for a, b in combinations(unique_files, 2):
                counts[(a, b)] += 1

        filtered = {pair: count for pair, count in counts.items() if count >= self._min_cochanges}
        return filtered, dict(total_changes)

    @staticmethod
    def _calculate_coupling(
        file_a: str,
        file_b: str,
        co_changes: int,
        total_changes: dict[str, int],
    ) -> float:
        max_changes = max(total_changes.get(file_a, 0), total_changes.get(file_b, 0))
        if max_changes == 0:
            return 0.0
        return co_changes / max_changes

    def process_coupling(self, commits: list[list[str]] | None = None) -> int:
        """
        Analyze git history and create ``COUPLED_WITH`` relationships.

        Parses the git log (or uses pre-supplied *commits* for testing),
        computes pairwise coupling strengths, and adds edges between ``File``
        nodes that exceed *min_strength*.

        Args:
            graph: The knowledge graph containing ``File`` nodes.
            repo_path: Root of the git repository.
            min_strength: Minimum coupling strength to create a relationship.
            commits: Pre-parsed commit data.  When provided, ``parse_git_log``
                is skipped — useful for deterministic testing.
            min_cochanges: Minimum co-change count to include a pair.  Defaults
                to 3 to keep the matrix small; pass 1 for tests with small
                commit sets.

        Returns:
            The number of ``COUPLED_WITH`` relationships created.
        """
        edges = self.resolve_coupling(commits)

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

        logger.info("Created %d COUPLED_WITH relationships", len(edges))
        return len(edges)

    def _parse_git_log(
        self,
        since_months: int = 6,
    ) -> list[list[str]]:
        """
        Run ``git log`` and return commits as lists of changed file paths.

        Each inner list contains the file paths that were modified in a single
        commit. Only files present in the graph are kept, so the output is
        already filtered to source files known to the graph.

        Args:
            since_months: How far back in history to look.

        Returns:
            A list of commits, each represented as a list of changed file paths.
            Returns an empty list when the git command fails (e.g. not a repo).
        """
        cmd = [
            "git",
            "log",
            "--name-only",
            "--pretty=format:COMMIT:%H",
            f"--since={since_months} months ago",
        ]

        try:
            result = subprocess_run(
                cmd,
                cwd=self._repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
        except (CalledProcessError, FileNotFoundError, TimeoutExpired):
            logger.debug("git log failed for %s — not a git repo?", self._repo_path)
            return []

        commits: list[list[str]] = []
        current_files: list[str] = []

        for line in result.stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith("COMMIT:"):
                # Start of a new commit — flush the previous one.
                if current_files:
                    commits.append(current_files)
                current_files = []
            elif stripped in self._graph_files:
                current_files.append(stripped)

        # Flush the last commit.
        if current_files:
            commits.append(current_files)

        return commits
