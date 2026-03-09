"""
Phase 8: Community detection for Axon.

Uses igraph + leidenalg to partition the call graph into functional clusters
(communities). Each community groups tightly-connected symbols that likely
belong to the same logical feature or module.
"""

from collections import Counter
from logging import getLogger
from pathlib import PurePosixPath
from typing import cast

from igraph import Graph
from leidenalg import ModularityVertexPartition, find_partition

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)

logger = getLogger(__name__)


class Community:
    """
    Handles community detection for the Axon knowledge graph.

    Uses igraph + leidenalg to partition the call graph into functional clusters
    (communities). Each community groups tightly-connected symbols that likely
    belong to the same logical feature or module.
    """

    _CALLABLE_LABELS: tuple[NodeLabel, ...] = (
        NodeLabel.FUNCTION,
        NodeLabel.METHOD,
        NodeLabel.CLASS,
    )

    def __init__(
        self,
        graph: KnowledgeGraph,
        min_community_size: int = 2,
    ) -> None:
        """
        Initialize the Community detector.

        Args:
            graph: The knowledge graph to analyze and augment.
            min_community_size: Minimum number of members for a community to be
                created. Communities smaller than this are skipped.
        """
        self._graph = graph
        self._min_community_size = min_community_size

    def process_communities(self) -> int:
        """
        Detect communities in the call graph and add them to the knowledge graph.

        Uses the Leiden algorithm with modularity-based vertex partitioning.

        For each detected community that meets the minimum size threshold:
        - A :attr:`NodeLabel.COMMUNITY` node is created with a generated label
          and metadata (cohesion score, symbol count).
        - :attr:`RelType.MEMBER_OF` relationships are created from each member
          symbol to the community node.

        Returns:
            The number of community nodes created.
        """
        ig_graph, index_to_node_id = self._export_to_igraph(self._graph)

        if not self._is_graph_size_sufficient(ig_graph):
            return 0

        partition = find_partition(ig_graph, ModularityVertexPartition)
        return self._process_partition(partition, index_to_node_id)

    def _is_graph_size_sufficient(self, ig_graph: Graph) -> bool:
        """Check if the igraph object has enough vertices for clustering."""
        if ig_graph.vcount() < 3:
            logger.debug(
                "Call graph too small for community detection (%d nodes), skipping.",
                ig_graph.vcount(),
            )
            return False
        return True

    def _process_partition(
        self,
        partition: ModularityVertexPartition,
        index_to_node_id: dict[int, str],
    ) -> int:
        """
        Process the graph partition, extract and store valid communities.

        Args:
            partition: The vertex partition from leidenalg.
            index_to_node_id: Mapping connecting igraph vertex indices back to Axon node IDs.

        Returns:
            The number of community nodes created.
        """
        community_count = 0
        modularity_score = cast(float, partition.modularity)

        for i, members in enumerate(partition):
            member_indices = cast(list[int], members)
            if len(member_indices) < self._min_community_size:
                continue

            member_ids = [index_to_node_id[idx] for idx in member_indices]
            self._store_community(i, member_ids, modularity_score)
            community_count += 1

            # logger.info(
            #     "Community %d: %r with %d members (modularity=%.3f)",
            #     i,
            #     self._generate_label(self._graph, member_ids),
            #     len(member_indices),
            #     modularity_score,
            # )

        logger.info(
            "Community detection complete: %d communities created.",
            community_count,
        )
        return community_count

    def _store_community(
        self,
        community_index: int,
        member_ids: list[str],
        modularity_score: float,
    ) -> None:
        """
        Store a single community in the knowledge graph.

        Args:
            community_index: The index of the community (used for ID generation).
            member_ids: List of Axon node IDs that compose the community.
            modularity_score: The cohesion score from the graph partition.
        """
        community_id = generate_id(NodeLabel.COMMUNITY, f"community_{community_index}")
        label = self._generate_label(self._graph, member_ids)

        self._create_community_node(
            community_id,
            label,
            modularity_score,
            len(member_ids),
        )
        self._create_member_relationships(community_id, member_ids)

    def _create_community_node(
        self,
        community_id: str,
        label: str,
        modularity_score: float,
        symbol_count: int,
    ) -> None:
        """Create and add the community node to the graph."""
        community_node = GraphNode(
            id=community_id,
            label=NodeLabel.COMMUNITY,
            name=label,
            properties={
                "cohesion": modularity_score,
                "symbol_count": symbol_count,
            },
        )
        self._graph.add_node(community_node)

    def _create_member_relationships(
        self,
        community_id: str,
        member_ids: list[str],
    ) -> None:
        """Create MEMBER_OF relations from member nodes to the community node."""
        for member_id in member_ids:
            rel_id = f"member_of:{member_id}->{community_id}"
            self._graph.add_relationship(
                GraphRelationship(
                    id=rel_id,
                    type=RelType.MEMBER_OF,
                    source=member_id,
                    target=community_id,
                ),
            )

    def _export_to_igraph(
        self,
        graph: KnowledgeGraph,
    ) -> tuple[Graph, dict[int, str]]:
        """
        Extract the call graph from *graph* and build an igraph representation.

        Only Function, Method, and Class nodes are included. Only CALLS
        relationships between those nodes are used as edges.

        Args:
            graph: The Axon knowledge graph.

        Returns:
            A tuple of ``(igraph_graph, vertex_index_to_node_id)`` where the
            mapping connects igraph vertex indices back to Axon node IDs.
        """
        node_id_to_index, index_to_node_id = self._map_callable_nodes(graph)
        edge_list = self._extract_call_edges(graph, node_id_to_index)

        ig_graph = Graph(directed=True)
        ig_graph.add_vertices(len(node_id_to_index))
        ig_graph.add_edges(edge_list)

        return ig_graph, index_to_node_id

    def _map_callable_nodes(
        self,
        graph: KnowledgeGraph,
    ) -> tuple[dict[str, int], dict[int, str]]:
        """
        Map graph nodes filtered by callable labels to integer indices.

        Returns:
            A tuple containing (node_id_to_index, index_to_node_id) dicts.
        """
        node_id_to_index: dict[str, int] = {}
        index_to_node_id: dict[int, str] = {}

        for label in self._CALLABLE_LABELS:
            for node in graph.get_nodes_by_label(label):
                idx = len(node_id_to_index)
                node_id_to_index[node.id] = idx
                index_to_node_id[idx] = node.id

        return node_id_to_index, index_to_node_id

    def _extract_call_edges(
        self,
        graph: KnowledgeGraph,
        node_id_to_index: dict[str, int],
    ) -> list[tuple[int, int]]:
        """
        Extract edges mapped to their corresponding integer indices.

        Only includes CALLS relations between nodes that were previously mapped.

        Args:
            graph: The knowledge graph holding the relationships.
            node_id_to_index: Index mapping for existing node IDs.

        Returns:
            A list of tuples representing edges (source index, target index).
        """
        edge_list: list[tuple[int, int]] = []
        for rel in graph.get_relationships_by_type(RelType.CALLS):
            src_idx = node_id_to_index.get(rel.source)
            tgt_idx = node_id_to_index.get(rel.target)
            if src_idx is not None and tgt_idx is not None:
                edge_list.append((src_idx, tgt_idx))

        return edge_list

    def _generate_label(self, graph: KnowledgeGraph, member_ids: list[str]) -> str:
        """
        Generate a heuristic label for a community based on member file paths.

        Strategy:
        - Extract the parent directory from each member's ``file_path``.
        - If all members share the same directory, use that directory name.
        - Otherwise, combine the two most frequent directories with ``+``.
        - Capitalize and clean up the result.

        Falls back to ``"Cluster"`` if no file paths are available.

        Args:
            graph: The knowledge graph (used to look up member nodes).
            member_ids: List of node IDs belonging to this community.

        Returns:
            A human-readable label string.
        """
        directories = self._extract_directories(graph, member_ids)

        if not directories:
            return "Cluster"

        return self._compute_label_from_directories(directories)

    def _extract_directories(
        self,
        graph: KnowledgeGraph,
        member_ids: list[str],
    ) -> list[str]:
        """
        Get all directory names from the member nodes.

        Returns:
            A list of parent directory names.
        """
        directories: list[str] = []
        for nid in member_ids:
            node = graph.get_node(nid)
            if node is not None and node.file_path:
                parent = PurePosixPath(node.file_path).parent.name
                if parent:
                    directories.append(parent)
        return directories

    def _compute_label_from_directories(self, directories: list[str]) -> str:
        """
        Compute the final label text based on directory occurrences.

        Args:
            directories: List of directory names discovered.

        Returns:
            The formatted community label string.
        """
        counts = Counter(directories)
        most_common = counts.most_common(2)

        if len(most_common) == 1 or most_common[0][0] == most_common[-1][0]:
            # All members in the same directory.
            return most_common[0][0].capitalize()

        # Mixed directories: combine top two.
        label = f"{most_common[0][0]}+{most_common[1][0]}"
        return label.capitalize()
