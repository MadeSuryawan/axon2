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

    _HERITAGE_EDGE_TYPES: tuple[RelType, ...] = (
        RelType.EXTENDS,
        RelType.IMPLEMENTS,
        RelType.USES_TYPE,
    )
    _CALLS_WEIGHT = 1.0
    _HERITAGE_WEIGHT = 0.5

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

    def export_to_igraph(self) -> tuple[Graph, dict[int, str]]:
        """
        Extract the call + heritage graph and build an igraph representation.

        This method combines callable nodes (Function, Method, Class) from the
        knowledge graph with weighted edges representing both CALLS relationships
        (weight=1.0) and heritage relationships (EXTENDS, IMPLEMENTS, USES_TYPE)
        with weight=0.5. The heritage edges have lower weight to influence
        community structure without dominating over direct call relationships.

        The igraph is built with directed edges and includes edge weights for
        use in community detection algorithms that consider edge strength.

        Args:
            self: The Community instance containing the knowledge graph.

        Returns:
            A tuple containing:
                - Graph: The igraph representation of the combined call
                  and heritage graph.
                - dict[int, str]: Mapping from igraph vertex indices to Axon
                  node IDs for later reference.

        Raises:
            KeyError: If a relationship references a node not in the callable
                set (should not occur in well-formed graphs).
            ValueError: If the graph contains invalid relationship data.
        """
        # Step 1: Map all callable nodes (Function, Method, Class) to integer indices.
        # This creates a bidirectional mapping between Axon node IDs and igraph indices.
        node_id_to_index, index_to_node_id = self._map_callable_nodes_to_indices()

        # Step 2: Extract all edges with their weights from the knowledge graph.
        # CALLS edges get weight 1.0 (full strength), heritage edges get 0.5.
        edge_list, edge_weights = self._extract_weighted_edges(node_id_to_index)

        # Step 3: Construct the igraph with vertices, edges, and weights.
        # The graph is directed to preserve caller/callee relationships.
        ig_graph = self._build_igraph(node_id_to_index, edge_list, edge_weights)

        return ig_graph, index_to_node_id

    def _map_callable_nodes_to_indices(
        self,
    ) -> tuple[dict[str, int], dict[int, str]]:
        """
        Map all callable nodes to integer indices for igraph vertex mapping.

        Iterates through all callable node labels (Function, Method, Class) and
        assigns each unique node a sequential integer index. Maintains bidirectional
        mappings for forward construction and reverse lookup.

        Returns:
            A tuple containing:
                - dict[str, int]: Mapping from Axon node ID to igraph vertex index.
                - dict[int, str]: Mapping from igraph vertex index to Axon node ID.

        Note:
            The index assignment follows the order of NodeLabel iteration,
            then insertion order within each label type.
        """
        node_id_to_index: dict[str, int] = {}
        index_to_node_id: dict[int, str] = {}

        for label in self._CALLABLE_LABELS:
            for node in self._graph.get_nodes_by_label(label):
                idx = len(node_id_to_index)
                node_id_to_index[node.id] = idx
                index_to_node_id[idx] = node.id

        return node_id_to_index, index_to_node_id

    def _extract_weighted_edges(
        self,
        node_id_to_index: dict[str, int],
    ) -> tuple[list[tuple[int, int]], list[float]]:
        """
        Extract all relationships from the graph and convert to indexed edges with weights.

        Processes CALLS relationships first (higher weight), then iterates through
        heritage relationship types (EXTENDS, IMPLEMENTS, USES_TYPE) with lower weight.
        Only includes edges where both source and target nodes exist in the callable set.

        Args:
            node_id_to_index: Mapping from Axon node IDs to igraph indices.

        Returns:
            A tuple containing:
                - list[tuple[int, int]]: List of (source_index, target_index) edge tuples.
                - list[float]: Corresponding edge weights matching edge_list order.

        Note:
            The conditional check `src_idx is not None and tgt_idx is not None`
            filters out edges to/from nodes that are not callable (e.g., variables,
            parameters, or imported modules).
        """
        edge_list: list[tuple[int, int]] = []
        edge_weights: list[float] = []

        # Process CALLS relationships with full weight (1.0).
        # These represent direct function/method invocations.
        for rel in self._graph.get_relationships_by_type(RelType.CALLS):
            src_idx = node_id_to_index.get(rel.source)
            tgt_idx = node_id_to_index.get(rel.target)
            # Only include edge if both endpoints are callable nodes.
            if src_idx is not None and tgt_idx is not None:
                edge_list.append((src_idx, tgt_idx))
                edge_weights.append(self._CALLS_WEIGHT)

        # Process heritage relationships with reduced weight (0.5).
        # Lower weight ensures call relationships dominate community formation.
        for rel_type in self._HERITAGE_EDGE_TYPES:
            for rel in self._graph.get_relationships_by_type(rel_type):
                src_idx = node_id_to_index.get(rel.source)
                tgt_idx = node_id_to_index.get(rel.target)
                # Only include edge if both endpoints are callable nodes.
                if src_idx is not None and tgt_idx is not None:
                    edge_list.append((src_idx, tgt_idx))
                    edge_weights.append(self._HERITAGE_WEIGHT)

        return edge_list, edge_weights

    def _build_igraph(
        self,
        node_id_to_index: dict[str, int],
        edge_list: list[tuple[int, int]],
        edge_weights: list[float],
    ) -> Graph:
        """
        Construct an igraph from vertex and edge mappings.

        Creates a directed igraph with the specified number of vertices and edges.
        Edge weights are optionally attached to the graph's edge sequence if provided.

        Args:
            node_id_to_index: Mapping from node IDs to indices (used only for count).
            edge_list: List of (source_index, target_index) tuples.
            edge_weights: List of edge weights corresponding to edge_list.

        Returns:
            Graph: The constructed igraph with vertices, edges, and optional weights.

        Note:
            The weight attribute is only set if edge_weights is non-empty to avoid
            igraph warnings about empty attribute assignment.
        """
        num_vertices = len(node_id_to_index)

        ig_graph = Graph(directed=True)
        ig_graph.add_vertices(num_vertices)
        ig_graph.add_edges(edge_list)

        # Only assign weights if we have edges; avoids empty attribute issues.
        if edge_weights:
            ig_graph.es["weight"] = edge_weights

        return ig_graph

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
            if (
                node is not None
                and node.file_path
                and (parent := PurePosixPath(node.file_path).parent.name)
            ):
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
