"""Tests for the storage backend abstraction layer."""

from pathlib import Path
from typing import Any

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import GraphNode, GraphRelationship, RelType
from axon.core.storage.base import NodeEmbedding, SearchResult, StorageBackend

# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


class TestSearchResult:
    """Verify SearchResult dataclass defaults and construction."""

    def test_creation_with_defaults(self) -> None:
        result = SearchResult(node_id="n1", score=0.95)
        assert result.node_id == "n1"
        assert result.score == 0.95
        assert result.node_name == ""
        assert result.file_path == ""
        assert result.label == ""
        assert result.snippet == ""

    def test_creation_with_all_fields(self) -> None:
        result = SearchResult(
            node_id="function:app.py:main",
            score=0.87,
            node_name="main",
            file_path="app.py",
            label="function",
            snippet="def main() -> None: ...",
        )
        assert result.node_id == "function:app.py:main"
        assert result.score == 0.87
        assert result.node_name == "main"
        assert result.file_path == "app.py"
        assert result.label == "function"
        assert result.snippet == "def main() -> None: ..."


# ---------------------------------------------------------------------------
# NodeEmbedding
# ---------------------------------------------------------------------------


class TestNodeEmbedding:
    """Verify NodeEmbedding dataclass defaults and construction."""

    def test_creation_with_defaults(self) -> None:
        emb = NodeEmbedding(node_id="n1")
        assert emb.node_id == "n1"
        assert emb.embedding == []

    def test_creation_with_data(self) -> None:
        vec = [0.1, 0.2, 0.3]
        emb = NodeEmbedding(node_id="n2", embedding=vec)
        assert emb.node_id == "n2"
        assert emb.embedding == [0.1, 0.2, 0.3]

    def test_embedding_default_is_independent(self) -> None:
        """Mutable default must not be shared across instances."""
        a = NodeEmbedding(node_id="a")
        b = NodeEmbedding(node_id="b")
        a.embedding.append(1.0)
        assert b.embedding == []


# ---------------------------------------------------------------------------
# StorageBackend protocol
# ---------------------------------------------------------------------------


class _DummyBackend:
    def initialize(self, path: Path) -> None:
        pass

    def close(self) -> None:
        pass

    def add_nodes(self, nodes: list[GraphNode]) -> None:
        pass

    def add_relationships(self, rels: list[GraphRelationship]) -> None:
        pass

    def remove_nodes_by_file(self, file_path: str) -> int:
        return 0

    def get_inbound_cross_file_edges(
        self,
        file_path: str,
        exclude_source_files: set[str] | None = None,
    ) -> list[GraphRelationship]:
        return []

    def get_node(self, node_id: str) -> GraphNode | None:
        return None

    def get_callers(self, node_id: str) -> list[GraphNode]:
        return []

    def get_callees(self, node_id: str) -> list[GraphNode]:
        return []

    def get_type_refs(self, node_id: str) -> list[GraphNode]:
        return []

    def get_callers_with_confidence(self, node_id: str) -> list[tuple[GraphNode, float]]:
        return []

    def get_callees_with_confidence(self, node_id: str) -> list[tuple[GraphNode, float]]:
        return []

    def traverse(
        self,
        start_id: str,
        depth: int,
        direction: str = "callers",
    ) -> list[GraphNode]:
        return []

    def traverse_with_depth(
        self,
        start_id: str,
        depth: int,
        direction: str = "callers",
    ) -> list[tuple[GraphNode, int]]:
        return []

    def get_process_memberships(self, node_ids: list[str]) -> dict[str, str]:
        return {}

    def execute_raw(self, query: str) -> list[list[Any]]:
        return []

    def exact_name_search(self, name: str, limit: int = 5) -> list[SearchResult]:
        return []

    def fts_search(self, query: str, limit: int) -> list[SearchResult]:
        return []

    def fuzzy_search(
        self,
        query: str,
        limit: int,
        max_distance: int = 2,
    ) -> list[SearchResult]:
        return []

    def store_embeddings(self, embeddings: list[NodeEmbedding]) -> None:
        pass

    def vector_search(self, vector: list[float], limit: int) -> list[SearchResult]:
        return []

    def get_indexed_files(self) -> dict[str, str]:
        return {}

    def bulk_load(self, graph: KnowledgeGraph) -> None:
        pass

    def load_graph(self) -> KnowledgeGraph:
        return KnowledgeGraph()

    def delete_synthetic_nodes(self) -> None:
        pass

    def upsert_embeddings(self, embeddings: list[NodeEmbedding]) -> None:
        pass

    def update_dead_flags(self, dead_ids: set[str], alive_ids: set[str]) -> None:
        pass

    def remove_relationships_by_type(self, rel_type: RelType) -> None:
        pass

    def rebuild_fts_indexes(self) -> None:
        pass


class TestStorageBackend:
    """Verify the StorageBackend protocol is runtime-checkable."""

    def test_is_a_type(self) -> None:
        assert isinstance(StorageBackend, type)

    def test_runtime_checkable(self) -> None:
        """A class implementing all required methods should be recognised."""

        assert isinstance(_DummyBackend(), StorageBackend)

    def test_non_conforming_class_fails(self) -> None:
        """A class missing required methods should NOT match the protocol."""

        class _Incomplete:
            pass

        assert not isinstance(_Incomplete(), StorageBackend)
