"""
KuzuDB storage backend for Axon.

Implements the :class:`StorageBackend` protocol using KuzuDB, an embedded
graph database that speaks Cypher. Each :class:`NodeLabel` maps to a
separate node table, and a single ``CodeRelation`` relationship table group
covers all source-to-target combinations.
"""

from __future__ import annotations

from collections import deque
from contextlib import suppress
from csv import writer as csv_writer
from hashlib import sha256
from logging import getLogger
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from kuzu import Connection, Database, QueryResult

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import GraphNode, GraphRelationship, NodeLabel, RelType
from axon.core.storage.base import BackendNotInitializedError, NodeEmbedding, SearchResult

logger = getLogger(__name__)

_NODE_TABLE_NAMES: list[str] = [label.name.title().replace("_", "") for label in NodeLabel]

_LABEL_TO_TABLE: dict[str, str] = {
    label.value: label.name.title().replace("_", "") for label in NodeLabel
}

_LABEL_MAP: dict[str, NodeLabel] = {label.value: label for label in NodeLabel}

_REL_TYPE_MAP: dict[str, RelType] = {rt.value: rt for rt in RelType}

_SEARCHABLE_TABLES: list[str] = [
    t for t in _NODE_TABLE_NAMES if t not in ("Folder", "Community", "Process")
]

_NODE_PROPERTIES = (
    "id STRING, "
    "name STRING, "
    "file_path STRING, "
    "start_line INT64, "
    "end_line INT64, "
    "content STRING, "
    "signature STRING, "
    "language STRING, "
    "class_name STRING, "
    "is_dead BOOL, "
    "is_entry_point BOOL, "
    "is_exported BOOL, "
    "PRIMARY KEY (id)"
)

_REL_PROPERTIES = (
    "rel_type STRING, "
    "confidence DOUBLE, "
    "role STRING, "
    "step_number INT64, "
    "strength DOUBLE, "
    "co_changes INT64, "
    "symbols STRING"
)


def _escape(value: str) -> str:
    """Escape a string for safe inclusion in a Cypher literal."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _table_for_id(node_id: str) -> str | None:
    """Extract the table name from a node ID by mapping its label prefix."""
    prefix = node_id.split(":", 1)[0]
    return _LABEL_TO_TABLE.get(prefix)


EMBEDDING_PROPERTIES = "node_id STRING, vec DOUBLE[], PRIMARY KEY(node_id)"


class KuzuBackend:
    """
    StorageBackend implementation backed by KuzuDB.

        Usage::

            backend = KuzuBackend()
            backend.initialize(Path("/tmp/axon_db"))
            backend.bulk_load(graph)
            node = backend.get_node("function:src/app.py:main")
            backend.close()
    """

    def __init__(self) -> None:
        self._db: Database | None = None
        self._conn: Connection | None = None

    def _ensure_initialized(self) -> Connection:
        """
        Ensure the backend is initialized and return the connection.

        Raises:
            BackendNotInitializedError: If initialize() has not been called.

        Returns:
            The active Connection instance.
        """
        if self._conn is None:
            raise BackendNotInitializedError()
        return self._conn

    def initialize(self, path: Path, *, read_only: bool = False) -> None:
        """
                Open or create the KuzuDB database at *path* and set up the schema.

        Args:
            path: Filesystem path to the KuzuDB database directory.
            read_only: If ``True``, open the database in read-only mode.
                This allows multiple concurrent readers (e.g. MCP server
                instances) without lock conflicts.  Schema creation is
                skipped since the database must already exist.
        """
        self._db = Database(str(path), read_only=read_only)
        self._conn = Connection(self._db)
        if not read_only:
            self._create_schema()

    def close(self) -> None:
        """
                Release the connection and database handles.

        Explicitly deletes the connection and database objects to ensure
        KuzuDB releases file locks and flushes data.
        """
        if self._conn is not None:
            with suppress(Exception):
                del self._conn
            self._conn = None

        if self._db is not None:
            with suppress(Exception):
                del self._db
            self._db = None

    def add_nodes(self, nodes: list[GraphNode]) -> None:
        """Insert nodes into their respective label tables."""
        for node in nodes:
            self._insert_node(node)

    def add_relationships(self, rels: list[GraphRelationship]) -> None:
        """Insert relationships by matching source and target nodes."""
        for rel in rels:
            self._insert_relationship(rel)

    def remove_nodes_by_file(self, file_path: str) -> int:
        """
                Delete all nodes whose ``file_path`` matches across every table.

        Returns:
            Always 0 — exact count is not tracked for performance.
        """
        conn = self._ensure_initialized()
        for table in _NODE_TABLE_NAMES:
            try:
                conn.execute(
                    f"MATCH (n:{table}) WHERE n.file_path = $fp DETACH DELETE n",
                    parameters={"fp": file_path},
                )
            except RuntimeError:
                logger.debug(f"Failed to remove nodes from table {table}", exc_info=True)
        return 0

    def _parse_edge_row(self, row: list[Any]) -> GraphRelationship | None:
        """Parse a result row into a GraphRelationship."""
        src_id: str = row[0] or ""
        tgt_id: str = row[2] or ""
        rel_type_str: str = row[3] or ""
        rel_type = _REL_TYPE_MAP.get(rel_type_str)
        if rel_type is None:
            return None

        props: dict[str, Any] = {}
        if row[4] is not None:
            props["confidence"] = float(row[4])
        if row[5] is not None and row[5] != "":
            props["role"] = str(row[5])
        if row[6] is not None and row[6] != 0:
            props["step_number"] = int(row[6])
        if row[7] is not None and row[7] != 0.0:
            props["strength"] = float(row[7])
        if row[8] is not None and row[8] != 0:
            props["co_changes"] = int(row[8])
        if row[9] is not None and row[9] != "":
            props["symbols"] = str(row[9])

        rel_id = f"{rel_type_str}:{src_id}->{tgt_id}"
        return GraphRelationship(
            id=rel_id,
            type=rel_type,
            source=src_id,
            target=tgt_id,
            properties=props,
        )

    def get_inbound_cross_file_edges(
        self,
        file_path: str,
        exclude_source_files: set[str] | None = None,
    ) -> list[GraphRelationship]:
        """
                Return inbound edges where target is in *file_path* and source is not.

        Edges whose source file is in *exclude_source_files* are skipped.

        Args:
            file_path: Target file whose inbound edges to collect.
            exclude_source_files: Source file paths to skip.
        """
        conn = self._ensure_initialized()
        exclude = exclude_source_files or set()
        edges: list[GraphRelationship] = []
        try:
            result = conn.execute(
                "MATCH (caller)-[r:CodeRelation]->(n) "
                "WHERE n.file_path = $fp AND caller.file_path <> $fp "
                "RETURN caller.id, caller.file_path, n.id, "
                "r.rel_type, r.confidence, r.role, "
                "r.step_number, r.strength, r.co_changes, r.symbols",
                parameters={"fp": file_path},
            )
            if not isinstance(result, QueryResult):
                return []
            while result.has_next():
                row = result.get_next()
                if not isinstance(row, list):
                    continue
                src_file: str = row[1] or ""
                if src_file in exclude:
                    continue
                rel = self._parse_edge_row(row)
                if rel is not None:
                    edges.append(rel)
        except (RuntimeError, ConnectionError, SystemError):
            logger.debug(
                "Failed to query inbound cross-file edges for %s",
                file_path,
                exc_info=True,
            )
        return edges

    def get_node(self, node_id: str) -> GraphNode | None:
        """Return a single node by ID, or ``None`` if not found."""
        conn = self._ensure_initialized()
        table = _table_for_id(node_id)
        if table is None:
            return None

        query = f"MATCH (n:{table}) WHERE n.id = $nid RETURN n.*"
        try:
            result = conn.execute(query, parameters={"nid": node_id})
            if isinstance(result, QueryResult) and result.has_next():
                row = result.get_next()
                return self._row_to_node(row, node_id) if isinstance(row, list) else None
        except RuntimeError:
            logger.debug(f"get_node failed for {node_id}", exc_info=True)
        return None

    def get_callers(self, node_id: str) -> list[GraphNode]:
        """Return nodes that CALL the node identified by *node_id*."""
        self._ensure_initialized()
        table = _table_for_id(node_id)
        if table is None:
            return []

        query = (
            f"MATCH (caller)-[r:CodeRelation]->(callee:{table}) "
            f"WHERE callee.id = $nid AND r.rel_type = 'calls' "
            f"RETURN caller.*"
        )
        return self._query_nodes(query, parameters={"nid": node_id})

    def get_callees(self, node_id: str) -> list[GraphNode]:
        """Return nodes called by the node identified by *node_id*."""
        self._ensure_initialized()
        table = _table_for_id(node_id)
        if table is None:
            return []

        query = (
            f"MATCH (caller:{table})-[r:CodeRelation]->(callee) "
            f"WHERE caller.id = $nid AND r.rel_type = 'calls' "
            f"RETURN callee.*"
        )
        return self._query_nodes(query, parameters={"nid": node_id})

    def get_type_refs(self, node_id: str) -> list[GraphNode]:
        """Return nodes referenced via USES_TYPE from *node_id*."""
        self._ensure_initialized()
        table = _table_for_id(node_id)
        if table is None:
            return []

        query = (
            f"MATCH (src:{table})-[r:CodeRelation]->(tgt) "
            f"WHERE src.id = $nid AND r.rel_type = 'uses_type' "
            f"RETURN tgt.*"
        )
        return self._query_nodes(query, parameters={"nid": node_id})

    def get_callers_with_confidence(self, node_id: str) -> list[tuple[GraphNode, float]]:
        """Return ``(node, confidence)`` for all callers of *node_id*."""
        self._ensure_initialized()
        table = _table_for_id(node_id)
        if table is None:
            return []
        query = (
            f"MATCH (caller)-[r:CodeRelation]->(callee:{table}) "
            f"WHERE callee.id = $nid AND r.rel_type = 'calls' "
            f"RETURN caller.*, r.confidence"
        )
        return self._query_nodes_with_confidence(query, parameters={"nid": node_id})

    def get_callees_with_confidence(self, node_id: str) -> list[tuple[GraphNode, float]]:
        """Return ``(node, confidence)`` for all callees of *node_id*."""
        self._ensure_initialized()
        table = _table_for_id(node_id)
        if table is None:
            return []
        query = (
            f"MATCH (caller:{table})-[r:CodeRelation]->(callee) "
            f"WHERE caller.id = $nid AND r.rel_type = 'calls' "
            f"RETURN callee.*, r.confidence"
        )
        return self._query_nodes_with_confidence(query, parameters={"nid": node_id})

    _MAX_BFS_DEPTH = 10

    def traverse(self, start_id: str, depth: int, direction: str = "callers") -> list[GraphNode]:
        """BFS traversal through CALLS edges — flat result list (no depth info)."""
        return [node for node, _ in self.traverse_with_depth(start_id, depth, direction)]

    def traverse_with_depth(
        self,
        start_id: str,
        depth: int,
        direction: str = "callers",
    ) -> list[tuple[GraphNode, int]]:
        """
                BFS traversal returning ``(node, hop_depth)`` pairs.

        ``hop_depth`` is 1-based: direct callers/callees are depth 1.

        Args:
            start_id: The ID of the node to start traversal from.
            depth: Maximum number of hops to traverse.
            direction: ``"callers"`` follows incoming CALLS (blast radius),
                       ``"callees"`` follows outgoing CALLS (dependencies).
        """
        self._ensure_initialized()
        depth = min(depth, self._MAX_BFS_DEPTH)
        if _table_for_id(start_id) is None:
            return []

        visited: set[str] = set()
        result_list: list[tuple[GraphNode, int]] = []
        queue: deque[tuple[str, int]] = deque([(start_id, 0)])

        while queue:
            current_id, current_depth = queue.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)

            if current_id != start_id:
                node = self.get_node(current_id)
                if node is not None:
                    result_list.append((node, current_depth))

            if current_depth < depth:
                neighbors = (
                    self.get_callers(current_id)
                    if direction == "callers"
                    else self.get_callees(current_id)
                )
                for neighbor in neighbors:
                    if neighbor.id not in visited:
                        queue.append((neighbor.id, current_depth + 1))

        return result_list

    def get_process_memberships(self, node_ids: list[str]) -> dict[str, str]:
        """
                Return ``{node_id: process_name}`` for nodes in any Process.

        Uses parameterized IN clause to safely query all node IDs at once.
        """
        conn = self._ensure_initialized()
        if not node_ids:
            return {}

        mapping: dict[str, str] = {}
        try:
            result = conn.execute(
                "MATCH (n)-[r:CodeRelation]->(p:Process) "
                "WHERE n.id IN $ids AND r.rel_type = 'step_in_process' "
                "RETURN n.id, p.name",
                parameters={"ids": node_ids},
            )
            if not isinstance(result, QueryResult):
                return {}
            while result.has_next():
                row = result.get_next()
                if not isinstance(row, list):
                    continue
                nid = row[0] if row else ""
                pname = row[1] if len(row) > 1 else ""
                if nid and pname and nid not in mapping:
                    mapping[nid] = pname
        except RuntimeError:
            logger.debug("get_process_memberships failed", exc_info=True)
        return mapping

    def execute_raw(self, query: str) -> list[list[Any]]:
        """Execute a raw Cypher query and return all result rows."""
        conn = self._ensure_initialized()
        result = conn.execute(query)
        rows: list[list[Any]] = []
        if not isinstance(result, QueryResult):
            return []
        while result.has_next():
            _result = result.get_next()
            if not isinstance(_result, list):
                continue
            rows.append(_result)
        return rows

    def exact_name_search(self, name: str, limit: int = 5) -> list[SearchResult]:
        """
                Search for nodes with an exact name match across all searchable tables.

        Returns results sorted by label priority (functions/methods first),
        preferring source files over test files.
        """
        conn = self._ensure_initialized()
        candidates: list[SearchResult] = []

        for table in _SEARCHABLE_TABLES:
            cypher = (
                f"MATCH (n:{table}) WHERE n.name = $name "
                f"RETURN n.id, n.name, n.file_path, n.content, n.signature "
                f"LIMIT {limit}"
            )
            try:
                result = conn.execute(cypher, parameters={"name": name})
                if not isinstance(result, QueryResult):
                    continue
                while result.has_next():
                    row = result.get_next()
                    if not isinstance(row, list):
                        continue
                    node_id = row[0] or ""
                    node_name = row[1] or ""
                    file_path = row[2] or ""
                    content = row[3] or ""
                    signature = row[4] or ""
                    label_prefix = node_id.split(":", 1)[0] if node_id else ""
                    snippet = content[:200] if content else signature[:200]
                    score = 2.0 if "/tests/" not in file_path else 1.0
                    candidates.append(
                        SearchResult(
                            node_id=node_id,
                            score=score,
                            node_name=node_name,
                            file_path=file_path,
                            label=label_prefix,
                            snippet=snippet,
                        ),
                    )
            except RuntimeError:
                logger.debug(f"exact_name_search failed on table {table}", exc_info=True)

        candidates.sort(key=lambda r: (-r.score, r.node_id))
        return candidates[:limit]

    def fts_search(self, query: str, limit: int) -> list[SearchResult]:
        """
                BM25 full-text search using KuzuDB's native FTS extension.

        Searches across all node tables using pre-built FTS indexes on
        ``name``, ``content``, and ``signature`` fields.  Results are
        ranked by BM25 relevance score.

        Returns the top *limit* results sorted by score descending.
        """
        conn = self._ensure_initialized()
        escaped_q = _escape(query)
        candidates: list[SearchResult] = []

        for table in _SEARCHABLE_TABLES:
            idx_name = f"{table.lower()}_fts"
            cypher = (
                f"CALL QUERY_FTS_INDEX('{table}', '{idx_name}', '{escaped_q}') "
                f"RETURN node.id, node.name, node.file_path, node.content, "
                f"node.signature, score "
                f"ORDER BY score DESC LIMIT {limit}"
            )
            try:
                result = conn.execute(cypher)
                if not isinstance(result, QueryResult):
                    continue
                while result.has_next():
                    row = result.get_next()
                    if not isinstance(row, list):
                        continue
                    node_id = row[0] or ""
                    name = row[1] or ""
                    file_path = row[2] or ""
                    content = row[3] or ""
                    signature = row[4] or ""
                    bm25_score = float(row[5]) if row[5] is not None else 0.0

                    # Demote test file results — mirrors exact_name_search penalty.
                    if "/tests/" in file_path or "/test_" in file_path:
                        bm25_score *= 0.5

                    label_prefix = node_id.split(":", 1)[0] if node_id else ""

                    # Boost top-level definitions in source files.
                    if label_prefix in ("function", "class") and "/tests/" not in file_path:
                        bm25_score *= 1.2

                    snippet = content[:200] if content else signature[:200]

                    candidates.append(
                        SearchResult(
                            node_id=node_id,
                            score=bm25_score,
                            node_name=name,
                            file_path=file_path,
                            label=label_prefix,
                            snippet=snippet,
                        ),
                    )
            except RuntimeError:
                logger.debug(f"fts_search failed on table {table}", exc_info=True)

        candidates.sort(key=lambda r: (-r.score, r.node_id))
        return candidates[:limit]

    def fuzzy_search(self, query: str, limit: int, max_distance: int = 2) -> list[SearchResult]:
        """
                Fuzzy name search using Levenshtein edit distance.

        Scans all node tables for symbols whose name is within
        *max_distance* edits of *query*.  Converts edit distance to a
        score (0 edits = 1.0, *max_distance* edits = 0.3).
        """
        conn = self._ensure_initialized()
        escaped_q = _escape(query.lower())
        candidates: list[SearchResult] = []

        for table in _SEARCHABLE_TABLES:
            cypher = (
                f"MATCH (n:{table}) "
                f"WHERE levenshtein(lower(n.name), '{escaped_q}') <= {max_distance} "
                f"RETURN n.id, n.name, n.file_path, n.content, "
                f"levenshtein(lower(n.name), '{escaped_q}') AS dist "
                f"ORDER BY dist LIMIT {limit}"
            )
            try:
                result = conn.execute(cypher)
                if not isinstance(result, QueryResult):
                    continue
                while result.has_next():
                    row = result.get_next()
                    if not isinstance(row, list):
                        continue
                    node_id = row[0] or ""
                    name = row[1] or ""
                    file_path = row[2] or ""
                    content = row[3] or ""
                    dist = int(row[4]) if row[4] is not None else max_distance

                    score = max(0.3, 1.0 - (dist * 0.3))
                    label_prefix = node_id.split(":", 1)[0] if node_id else ""

                    candidates.append(
                        SearchResult(
                            node_id=node_id,
                            score=score,
                            node_name=name,
                            file_path=file_path,
                            label=label_prefix,
                            snippet=content[:200] if content else "",
                        ),
                    )
            except RuntimeError:
                logger.debug(f"fuzzy_search failed on table {table}", exc_info=True)

        candidates.sort(key=lambda r: (-r.score, r.node_id))
        return candidates[:limit]

    def store_embeddings(self, embeddings: list[NodeEmbedding]) -> None:
        """
                Persist embedding vectors into the Embedding node table.

        Attempts batch CSV COPY FROM first, falls back to individual MERGE.
        """
        conn = self._ensure_initialized()
        if not embeddings:
            return

        if self._bulk_store_embeddings_csv(embeddings):
            return

        for emb in embeddings:
            try:
                conn.execute(
                    "MERGE (e:Embedding {node_id: $nid}) SET e.vec = $vec",
                    parameters={"nid": emb.node_id, "vec": emb.embedding},
                )
            except RuntimeError:
                logger.debug(f"store_embeddings failed for node {emb.node_id}", exc_info=True)

    def _fetch_node_metadata(self, node_ids: list[str]) -> dict[str, GraphNode]:
        """Fetch node metadata in batches across tables."""
        node_cache: dict[str, GraphNode] = {}
        ids_by_table: dict[str, list[str]] = {}
        for nid in node_ids:
            if table := _table_for_id(nid):
                ids_by_table.setdefault(table, []).append(nid)

        for table, ids in ids_by_table.items():
            try:
                q = f"MATCH (n:{table}) WHERE n.id IN $ids RETURN n.*"
                nodes = self._query_nodes(q, parameters={"ids": ids})
                for node in nodes:
                    node_cache[node.id] = node
            except RuntimeError:
                logger.debug(f"Batch node fetch failed for table {table}", exc_info=True)
        return node_cache

    def vector_search(self, vector: list[float], limit: int) -> list[SearchResult]:
        """
                Find the closest nodes to *vector* using native ``array_cosine_similarity``.

        Computes cosine similarity directly in KuzuDB's Cypher engine —
        no Python-side computation or full-table load required.  Joins with
        node tables to fetch metadata in a single query.
        """
        conn = self._ensure_initialized()
        # Vector literals must be inlined — KuzuDB parameterized queries
        # cannot distinguish DOUBLE[] from LIST for array_cosine_similarity.
        vec_literal = "[" + ", ".join(str(v) for v in vector) + "]"

        try:
            result = conn.execute(
                f"MATCH (e:Embedding) "
                f"RETURN e.node_id, "
                f"array_cosine_similarity(e.vec, {vec_literal}) AS sim "
                f"ORDER BY sim DESC LIMIT {limit}",
            )
        except RuntimeError:
            logger.debug("vector_search failed", exc_info=True)
            return []

        emb_rows: list[tuple[str, float]] = []
        if isinstance(result, QueryResult):
            while result.has_next():
                row = result.get_next()
                if isinstance(row, list):
                    sim = float(row[1]) if row[1] is not None else 0.0
                    emb_rows.append((row[0] or "", sim))

        if not emb_rows:
            return []

        node_cache = self._fetch_node_metadata([r[0] for r in emb_rows])
        results: list[SearchResult] = []
        for node_id, sim in emb_rows:
            node = node_cache.get(node_id)
            label_prefix = node_id.split(":", 1)[0] if node_id else ""
            results.append(
                SearchResult(
                    node_id=node_id,
                    score=sim,
                    node_name=node.name if node else "",
                    file_path=node.file_path if node else "",
                    label=label_prefix,
                    snippet=(node.content[:200] if node and node.content else ""),
                ),
            )
        return results

    def get_indexed_files(self) -> dict[str, str]:
        """
                Return ``{file_path: sha256(content)}`` for all File nodes.

        Attempts to read pre-computed ``content_hash`` first. Falls back
        to computing the hash from content for databases that predate the
        schema addition.
        """
        conn = self._ensure_initialized()
        mapping: dict[str, str] = {}
        try:
            result = conn.execute("MATCH (n:File) RETURN n.file_path, n.content")
            if not isinstance(result, QueryResult):
                return {}
            while result.has_next():
                row = result.get_next()
                if not isinstance(row, list):
                    continue
                fp = row[0] or ""
                content = row[1] or ""
                mapping[fp] = sha256(content.encode()).hexdigest()
        except RuntimeError:
            logger.debug("get_indexed_files failed", exc_info=True)
        return mapping

    def _load_nodes_to_graph(self, graph: KnowledgeGraph) -> None:
        """Load all nodes from every table into the graph."""
        for table in _NODE_TABLE_NAMES:
            try:
                # Use _query_nodes which handles the row-to-node conversion
                nodes = self._query_nodes(f"MATCH (n:{table}) RETURN n.*")
                for node in nodes:
                    graph.add_node(node)
            except RuntimeError:
                logger.debug(f"load_graph: failed to read table {table}", exc_info=True)

    def _load_relationships_to_graph(self, graph: KnowledgeGraph) -> None:
        """Load all relationships from the database into the graph."""
        conn = self._ensure_initialized()
        try:
            result = conn.execute(
                "MATCH (a)-[r:CodeRelation]->(b) "
                "RETURN a.id, b.id, r.rel_type, r.confidence, r.role, "
                "r.step_number, r.strength, r.co_changes, r.symbols",
            )
            if not isinstance(result, QueryResult):
                return
            while result.has_next():
                row = result.get_next()
                if not isinstance(row, list):
                    continue
                # Mapping result format to _parse_edge_row expected format
                # _parse_edge_row expects: row[0]=src_id, row[1]=src_file(unused), row[2]=tgt_id, row[3...]=props
                src_id, tgt_id = row[0], row[1]
                mapped_row = [src_id, None, tgt_id] + row[2:]
                if rel := self._parse_edge_row(mapped_row):
                    graph.add_relationship(rel)
        except Exception:
            logger.error("load_graph: relationship query failed", exc_info=True)
            raise

    def load_graph(self) -> KnowledgeGraph:
        """Reconstruct a full :class:`KnowledgeGraph` from the database."""
        self._ensure_initialized()
        graph = KnowledgeGraph()
        self._load_nodes_to_graph(graph)
        self._load_relationships_to_graph(graph)
        return graph

    def delete_synthetic_nodes(self) -> None:
        """Remove all COMMUNITY and PROCESS nodes and their relationships."""
        conn = self._ensure_initialized()
        for table in ("Community", "Process"):
            try:
                conn.execute(f"MATCH (n:{table}) DETACH DELETE n")
            except RuntimeError:
                logger.debug(f"delete_synthetic_nodes: failed for {table}", exc_info=True)

    def upsert_embeddings(self, embeddings: list[NodeEmbedding]) -> None:
        """Insert or update embeddings without wiping existing ones."""
        conn = self._ensure_initialized()
        for emb in embeddings:
            try:
                conn.execute(
                    "MERGE (e:Embedding {node_id: $nid}) SET e.vec = $vec",
                    parameters={"nid": emb.node_id, "vec": emb.embedding},
                )
            except RuntimeError:
                logger.debug(f"upsert_embeddings failed for {emb.node_id}", exc_info=True)

    def update_dead_flags(self, dead_ids: set[str], alive_ids: set[str]) -> None:
        """Set is_dead=True on *dead_ids* and is_dead=False on *alive_ids*."""
        conn = self._ensure_initialized()

        def _batch_set(ids: set[str], *, value: bool) -> None:
            by_table: dict[str, list[str]] = {}
            for node_id in ids:
                table = _table_for_id(node_id)
                if table:
                    by_table.setdefault(table, []).append(node_id)
            for table, id_list in by_table.items():
                try:
                    conn.execute(
                        f"MATCH (n:{table}) WHERE n.id IN $ids SET n.is_dead = $val",
                        parameters={"ids": id_list, "val": value},
                    )
                except RuntimeError:
                    logger.debug(f"update_dead_flags failed for table {table}", exc_info=True)

        _batch_set(dead_ids, value=True)
        _batch_set(alive_ids, value=False)

    def remove_relationships_by_type(self, rel_type: RelType) -> None:
        """Delete all relationships of a specific type."""
        conn = self._ensure_initialized()
        try:
            conn.execute(
                "MATCH ()-[r:CodeRelation]->() WHERE r.rel_type = $rt DELETE r",
                parameters={"rt": rel_type.value},
            )
        except RuntimeError:
            logger.debug(
                f"remove_relationships_by_type failed for {rel_type.value}",
                exc_info=True,
            )

    def bulk_load(self, graph: KnowledgeGraph) -> None:
        """
                Replace the entire store with the contents of *graph*.

        Uses CSV-based COPY FROM for bulk loading nodes and relationships,
        falling back to individual inserts if COPY FROM fails.
        """
        conn = self._ensure_initialized()
        for table in _NODE_TABLE_NAMES:
            with suppress(Exception):
                conn.execute(f"MATCH (n:{table}) DETACH DELETE n")

        if not self._bulk_load_nodes_csv(graph):
            self.add_nodes(list(graph.iter_nodes()))

        if not self._bulk_load_rels_csv(graph):
            self.add_relationships(list(graph.iter_relationships()))

        self.rebuild_fts_indexes()

    def rebuild_fts_indexes(self) -> None:
        """
                Drop and recreate all FTS indexes.

        Must be called after any bulk data change so the BM25 indexes
        reflect the current node contents.
        """
        conn = self._ensure_initialized()
        for table in _NODE_TABLE_NAMES:
            idx_name = f"{table.lower()}_fts"
            with suppress(Exception):
                conn.execute(f"CALL DROP_FTS_INDEX('{table}', '{idx_name}')")

            try:
                conn.execute(
                    f"CALL CREATE_FTS_INDEX('{table}', '{idx_name}', "
                    f"['name', 'content', 'signature'])",
                )
            except RuntimeError:
                logger.debug(f"FTS index rebuild failed for {table}", exc_info=True)

    def _csv_copy(self, table: str, rows: list[list[Any]]) -> None:
        """
                Write *rows* to a temporary CSV and COPY FROM into *table*.

        Always cleans up the temp file, even on failure.
        """
        conn = self._ensure_initialized()
        csv_path: str | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                suffix=".csv",
                delete=False,
                newline="",
            ) as f:
                writer = csv_writer(f)
                writer.writerows(rows)
                csv_path = f.name
            conn.execute(f'COPY {table} FROM "{csv_path}" (HEADER=false)')
        finally:
            if csv_path:
                Path(csv_path).unlink(missing_ok=True)

    def _bulk_load_nodes_csv(self, graph: KnowledgeGraph) -> bool:
        """
                Load all nodes via temporary CSV files + COPY FROM.

        Returns True on success, False if COPY FROM is not available.
        """
        by_table: dict[str, list[GraphNode]] = {}
        for node in graph.iter_nodes():
            table = _LABEL_TO_TABLE.get(node.label.value)
            if table:
                by_table.setdefault(table, []).append(node)

        try:
            for table, nodes in by_table.items():
                self._csv_copy(
                    table,
                    [
                        [
                            node.id,
                            node.name,
                            node.file_path,
                            node.start_line,
                            node.end_line,
                            node.content,
                            node.signature,
                            node.language,
                            node.class_name,
                            node.is_dead,
                            node.is_entry_point,
                            node.is_exported,
                        ]
                        for node in nodes
                    ],
                )
            return True
        except RuntimeError:
            logger.debug("CSV bulk_load_nodes failed, falling back", exc_info=True)
            return False

    def _bulk_load_rels_csv(self, graph: KnowledgeGraph) -> bool:
        """
                Load all relationships via temporary CSV files + COPY FROM.

        Returns True on success, False if COPY FROM is not available.
        """
        by_pair: dict[tuple[str, str], list[GraphRelationship]] = {}
        for rel in graph.iter_relationships():
            src_table = _table_for_id(rel.source)
            dst_table = _table_for_id(rel.target)
            if src_table and dst_table:
                by_pair.setdefault((src_table, dst_table), []).append(rel)

        try:
            for (src_table, dst_table), rels in by_pair.items():
                self._csv_copy(
                    f"CodeRelation_{src_table}_{dst_table}",
                    [
                        [
                            rel.source,
                            rel.target,
                            rel.type.value,
                            float((rel.properties or {}).get("confidence", 1.0)),
                            str((rel.properties or {}).get("role", "")),
                            int((rel.properties or {}).get("step_number", 0)),
                            float((rel.properties or {}).get("strength", 0.0)),
                            int((rel.properties or {}).get("co_changes", 0)),
                            str((rel.properties or {}).get("symbols", "")),
                        ]
                        for rel in rels
                    ],
                )
            return True
        except RuntimeError:
            logger.debug("CSV bulk_load_rels failed, falling back", exc_info=True)
            return False

    def _bulk_store_embeddings_csv(self, embeddings: list[NodeEmbedding]) -> bool:
        """
                Store embeddings via temporary CSV + COPY FROM.

        Returns True on success, False if COPY FROM is not available.
        """
        conn = self._ensure_initialized()
        try:
            with suppress(Exception):
                conn.execute("MATCH (e:Embedding) DETACH DELETE e")

            self._csv_copy(
                "Embedding",
                [
                    [emb.node_id, "[" + ",".join(str(v) for v in emb.embedding) + "]"]
                    for emb in embeddings
                ],
            )
            return True
        except RuntimeError:
            logger.debug("CSV bulk_store_embeddings failed, falling back", exc_info=True)
            return False

    def _create_schema(self) -> None:
        """Create node/rel/embedding tables and the FTS extension."""
        conn = self._ensure_initialized()

        try:
            conn.execute("INSTALL fts")
            conn.execute("LOAD EXTENSION fts")
        except RuntimeError:
            logger.debug("FTS extension load skipped (may already be loaded)", exc_info=True)

        for table in _NODE_TABLE_NAMES:
            stmt = f"CREATE NODE TABLE IF NOT EXISTS {table}({_NODE_PROPERTIES})"
            conn.execute(stmt)

        conn.execute(f"CREATE NODE TABLE IF NOT EXISTS Embedding({EMBEDDING_PROPERTIES})")

        # Build the REL TABLE GROUP covering all table-to-table combinations.
        from_to_pairs: list[str] = []
        for src in _NODE_TABLE_NAMES:
            for dst in _NODE_TABLE_NAMES:
                from_to_pairs.append(f"FROM {src} TO {dst}")

        pairs_clause = ", ".join(from_to_pairs)
        rel_stmt = (
            f"CREATE REL TABLE GROUP IF NOT EXISTS CodeRelation({pairs_clause}, {_REL_PROPERTIES})"
        )
        try:
            conn.execute(rel_stmt)
        except RuntimeError:
            logger.debug("REL TABLE GROUP creation skipped", exc_info=True)

        self._create_fts_indexes()

    def _create_fts_indexes(self) -> None:
        """Create FTS indexes for every node table (idempotent)."""
        conn = self._ensure_initialized()
        for table in _NODE_TABLE_NAMES:
            idx_name = f"{table.lower()}_fts"
            with suppress(Exception):
                # Index may already exist — that's fine.
                conn.execute(
                    f"CALL CREATE_FTS_INDEX('{table}', '{idx_name}', "
                    f"['name', 'content', 'signature'])",
                )

    def _check_duplicate_node(self, table: str, node_id: str) -> bool:
        """Check if a node with the same id already exists."""
        conn = self._ensure_initialized()
        query = f"MATCH (n:{table}) WHERE n.id = $nid RETURN n.id"
        result = conn.execute(query, parameters={"nid": node_id})
        if not isinstance(result, QueryResult):
            return False
        return result.has_next()

    def _insert_node(self, node: GraphNode) -> None:
        """INSERT a single node into the appropriate label table using parameterized query."""
        conn = self._ensure_initialized()
        table = _LABEL_TO_TABLE.get(node.label.value)
        if table is None:
            logger.warning("Unknown label %s for node %s", node.label, node.id)
            return
        if self._check_duplicate_node(table, node.id):
            return
        query = (
            f"CREATE (:{table} {{"
            f"id: $id, name: $name, file_path: $file_path, "
            f"start_line: $start_line, end_line: $end_line, "
            f"content: $content, signature: $signature, "
            f"language: $language, class_name: $class_name, "
            f"is_dead: $is_dead, is_entry_point: $is_entry_point, "
            f"is_exported: $is_exported"
            f"}})"
        )
        params = {
            "id": node.id,
            "name": node.name,
            "file_path": node.file_path,
            "start_line": node.start_line,
            "end_line": node.end_line,
            "content": node.content,
            "signature": node.signature,
            "language": node.language,
            "class_name": node.class_name,
            "is_dead": node.is_dead,
            "is_entry_point": node.is_entry_point,
            "is_exported": node.is_exported,
        }
        try:
            conn.execute(query, parameters=params)
        except Exception:
            logger.exception(f"Insert node failed for {node.id}", exc_info=True)

    def _insert_relationship(self, rel: GraphRelationship) -> None:
        """MATCH source and target, then CREATE the relationship using parameterized query."""
        conn = self._ensure_initialized()
        src_table = _table_for_id(rel.source)
        tgt_table = _table_for_id(rel.target)
        if src_table is None or tgt_table is None:
            logger.warning(
                "Cannot resolve tables for relationship %s -> %s",
                rel.source,
                rel.target,
            )
            return

        props = rel.properties or {}

        query = (
            f"MATCH (a:{src_table}), (b:{tgt_table}) "
            f"WHERE a.id = $src AND b.id = $tgt "
            f"CREATE (a)-[:CodeRelation {{"
            f"rel_type: $rel_type, "
            f"confidence: $confidence, "
            f"role: $role, "
            f"step_number: $step_number, "
            f"strength: $strength, "
            f"co_changes: $co_changes, "
            f"symbols: $symbols"
            f"}}]->(b)"
        )
        params = {
            "src": rel.source,
            "tgt": rel.target,
            "rel_type": rel.type.value,
            "confidence": float(props.get("confidence", 1.0)),
            "role": str(props.get("role", "")),
            "step_number": int(props.get("step_number", 0)),
            "strength": float(props.get("strength", 0.0)),
            "co_changes": int(props.get("co_changes", 0)),
            "symbols": str(props.get("symbols", "")),
        }
        try:
            conn.execute(query, parameters=params)
        except RuntimeError:
            logger.debug(
                f"Insert relationship failed: {rel.source} -> {rel.target}",
                exc_info=True,
            )

    def _query_nodes(self, query: str, parameters: dict[str, Any] | None = None) -> list[GraphNode]:
        """Execute a query returning ``n.*`` columns and convert to GraphNode list."""
        conn = self._ensure_initialized()
        nodes: list[GraphNode] = []
        try:
            result = conn.execute(query, parameters=parameters or {})
            if not isinstance(result, QueryResult):
                return []
            while result.has_next():
                row = result.get_next()
                if not isinstance(row, list):
                    continue
                node = self._row_to_node(row)
                if node is not None:
                    nodes.append(node)
        except RuntimeError:
            logger.debug(f"_query_nodes failed: {query}", exc_info=True)
        return nodes

    def _query_nodes_with_confidence(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[tuple[GraphNode, float]]:
        """Execute a query returning ``n.*`` columns plus a trailing confidence column."""
        conn = self._ensure_initialized()
        pairs: list[tuple[GraphNode, float]] = []
        try:
            result = conn.execute(query, parameters=parameters or {})
            if not isinstance(result, QueryResult):
                return []
            while result.has_next():
                row = result.get_next()
                if not isinstance(row, list):
                    continue
                node = self._row_to_node(row[:-1])
                confidence = float(row[-1]) if row[-1] is not None else 1.0
                if node is not None:
                    pairs.append((node, confidence))
        except RuntimeError:
            logger.debug(f"query_nodes_with_confidence failed: {query}", exc_info=True)
        return pairs

    @staticmethod
    def _row_to_node(row: list[Any], node_id: str | None = None) -> GraphNode | None:
        """
                Convert a result row from ``RETURN n.*`` into a GraphNode.

        Column order matches the property definition:
        0=id, 1=name, 2=file_path, 3=start_line, 4=end_line,
        5=content, 6=signature, 7=language, 8=class_name,
        9=is_dead, 10=is_entry_point, 11=is_exported
        """
        try:
            nid = node_id or row[0]
            prefix = nid.split(":", 1)[0]
            label = _LABEL_MAP.get(prefix)
            if label is None:
                logger.warning("Unknown node label prefix %r in id %s", prefix, nid)
                return None

            return GraphNode(
                id=row[0],
                label=label,
                name=row[1] or "",
                file_path=row[2] or "",
                start_line=row[3] or 0,
                end_line=row[4] or 0,
                content=row[5] or "",
                signature=row[6] or "",
                language=row[7] or "",
                class_name=row[8] or "",
                is_dead=bool(row[9]),
                is_entry_point=bool(row[10]),
                is_exported=bool(row[11]),
            )
        except (IndexError, KeyError):
            logger.debug("Failed to convert row to GraphNode: %s", row, exc_info=True)
            return None
