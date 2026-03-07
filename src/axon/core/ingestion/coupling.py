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
from subprocess import CalledProcessError
from subprocess import run as subprocess_run

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelType,
)

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

    def __init__(self, graph: KnowledgeGraph, repo_path: Path) -> None:
        """
        Initialize the Coupling analyzer.

        Args:
            graph: The knowledge graph containing File nodes.
            repo_path: Root of the git repository to analyze.
        """
        self._graph = graph
        self._repo_path = repo_path
        self._file_nodes: list[GraphNode] = []
        self._graph_files: set[str] = set()
        self._path_to_id: dict[str, str] = {}

    def process_coupling(
        self,
        min_strength: float = 0.3,
        *,
        commits: list[list[str]] | None = None,
    ) -> int:
        """
        Analyze git history and create ``COUPLED_WITH`` relationships.

        Parses the git log (or uses pre-supplied *commits* for testing),
        computes pairwise coupling strengths, and adds edges between ``File``
        nodes that exceed *min_strength*.

        Args:
            min_strength: Minimum coupling strength to create a relationship.
            commits: Pre-parsed commit data. When provided, git log parsing
                is skipped — useful for deterministic testing.

        Returns:
            The number of ``COUPLED_WITH`` relationships created.
        """
        # Pre-cache file nodes to avoid repeated graph traversals (performance)
        self._cache_file_nodes()

        if commits is None:
            commits = self._parse_git_log()

        # Build co-change matrix (threshold of 1 — filter by strength later)
        cochange = self._build_cochange_matrix(commits)

        # Count total changes per file across all commits
        total_changes = self._count_total_changes(commits)

        # Create coupling relationships for pairs above threshold
        return self._create_coupling_relationships(cochange, total_changes, min_strength)

    def _cache_file_nodes(self) -> None:
        """
        Cache file nodes and their mappings for efficient lookups.

        This pre-caching significantly improves performance when processing
        large graphs with many file nodes.
        """
        self._file_nodes = self._graph.get_nodes_by_label(NodeLabel.FILE)
        self._graph_files = {n.file_path for n in self._file_nodes}
        self._path_to_id = {n.file_path: n.id for n in self._file_nodes}

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
        except (CalledProcessError, FileNotFoundError):
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

    def _build_cochange_matrix(
        self,
        commits: list[list[str]],
        min_cochanges: int = _DEFAULT_MIN_COCHANGES,
        max_files_per_commit: int = _DEFAULT_MAX_FILES_PER_COMMIT,
    ) -> dict[tuple[str, str], int]:
        """
        Build a co-change frequency matrix from commit data.

        For every pair of files that appear in the same commit, their co-change
        count is incremented. Only pairs whose count meets or exceeds
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
            A dict mapping ``(file_a, file_b)`` sorted tuples to their count.
        """
        counts: dict[tuple[str, str], int] = defaultdict(int)

        for files in commits:
            unique_files = sorted(set(files))
            if len(unique_files) > max_files_per_commit:
                continue
            for a, b in combinations(unique_files, 2):
                counts[(a, b)] += 1

        return {pair: count for pair, count in counts.items() if count >= min_cochanges}

    def _count_total_changes(self, commits: list[list[str]]) -> dict[str, int]:
        """
        Count total changes per file across all commits.

        Args:
            commits: List of commits, each a list of changed file paths.

        Returns:
            Mapping of file path to its total commit count.
        """
        total_changes: dict[str, int] = defaultdict(int)
        for files in commits:
            for f in set(files):
                total_changes[f] += 1
        return total_changes

    def _calculate_coupling(
        self,
        file_a: str,
        file_b: str,
        co_changes: int,
        total_changes: dict[str, int],
    ) -> float:
        """
        Compute the coupling strength between two files.

        The formula is::

            coupling = co_changes / max(total_changes[file_a], total_changes[file_b])

        This yields a value in ``[0.0, 1.0]`` — higher means more tightly coupled.

        Args:
            file_a: First file path.
            file_b: Second file path.
            co_changes: Number of commits where both files changed together.
            total_changes: Mapping of file path to its total commit count.

        Returns:
            A float between 0.0 and 1.0 representing coupling strength.
        """
        max_changes = max(total_changes.get(file_a, 0), total_changes.get(file_b, 0))
        if max_changes == 0:
            return 0.0
        return co_changes / max_changes

    def _create_coupling_relationships(
        self,
        cochange: dict[tuple[str, str], int],
        total_changes: dict[str, int],
        min_strength: float,
    ) -> int:
        """
        Create COUPLED_WITH relationships for file pairs above the strength threshold.

        Args:
            cochange: Co-change frequency matrix.
            total_changes: Total changes per file.
            min_strength: Minimum coupling strength required.

        Returns:
            The number of relationships created.
        """
        count = 0
        for (file_a, file_b), co_changes in cochange.items():
            strength = self._calculate_coupling(file_a, file_b, co_changes, total_changes)
            if strength < min_strength:
                continue

            id_a = self._path_to_id.get(file_a)
            id_b = self._path_to_id.get(file_b)
            if id_a is None or id_b is None:
                continue

            rel_id = f"coupled:{id_a}->{id_b}"
            self._graph.add_relationship(
                GraphRelationship(
                    id=rel_id,
                    type=RelType.COUPLED_WITH,
                    source=id_a,
                    target=id_b,
                    properties={"strength": strength, "co_changes": co_changes},
                ),
            )
            count += 1

        logger.info("Created %d COUPLED_WITH relationships", count)
        return count
