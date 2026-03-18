"""
KuzuDB storage backend for Axon.

Implements the :class:`StorageBackend` protocol using KuzuDB, an embedded
graph database that speaks Cypher. Each :class:`NodeLabel` maps to a
separate node table, and a single ``CodeRelation`` relationship table group
covers all source-to-target combinations.
"""

from asyncio import gather
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from csv import writer as csv_writer
from hashlib import sha256
from json import dumps, loads
from logging import getLogger
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock
from time import sleep
from typing import Any

from kuzu import AsyncConnection, Connection, Database, QueryResult
from rich import print as rprint

from axon.config.progress_bar import p_bar, reset_pbar, tqdm
from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import GraphNode, GraphRelationship, NodeLabel, RelType
from axon.core.storage.base import NodeEmbedding, SearchResult

logger = getLogger(__name__)

_NODE_TABLE_NAMES = [label.name.title().replace("_", "") for label in NodeLabel]

_LABEL_TO_TABLE = {label.value: label.name.title().replace("_", "") for label in NodeLabel}

_LABEL_MAP = {label.value: label for label in NodeLabel}

_REL_TYPE_MAP = {rt.value: rt for rt in RelType}

_SEARCHABLE_TABLES = [t for t in _NODE_TABLE_NAMES if t not in ("Folder", "Community", "Process")]

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
    "cohesion DOUBLE, "
    "properties_json STRING, "
    "PRIMARY KEY (id)"
)

_DEDICATED_PROPS = frozenset({"cohesion"})

_REL_PROPERTIES = (
    "rel_type STRING, "
    "confidence DOUBLE, "
    "role STRING, "
    "step_number INT64, "
    "strength DOUBLE, "
    "co_changes INT64, "
    "symbols STRING"
)


def _serialize_extra_props(props: dict[str, Any] | None) -> str:
    if not props:
        return ""
    extra = {k: v for k, v in props.items() if k not in _DEDICATED_PROPS}
    return dumps(extra) if extra else ""


def escape_cypher(value: str) -> str:
    """Escape a string for safe inclusion in a Cypher literal."""
    value = value.replace("\x00", "")
    value = value.replace("/*", "")
    value = value.replace("*/", "")
    value = value.replace("//", "")
    value = value.replace(";", "")
    value = value.replace("\\", "\\\\")
    value = value.replace("'", "\\'")
    return value


def _safe_vec_literal(vector: list[float]) -> str:
    parts = []
    for v in vector:
        if not isfinite(f := float(v)):
            details = f"Non-finite float in embedding vector: {f}"
            raise ValueError(details)
        parts.append(repr(f))
    return "[" + ", ".join(parts) + "]"


def _table_for_id(node_id: str) -> str | None:
    """Extract the table name from a node ID by mapping its label prefix."""
    prefix = node_id.split(":", 1)[0]
    return _LABEL_TO_TABLE.get(prefix)


_EMBEDDING_PROPERTIES = "node_id STRING, vec DOUBLE[], PRIMARY KEY(node_id)"


class _Helpers:
    """
    Private base class encapsulating reusable implementation details and cross-cutting concerns for the KuzuDB storage backend.

    This class contains all helper methods organized by the public method
    they support, providing a clean separation between public API and
    internal implementation.
    """

    # === Instance state (shared with KuzuBackend via composition) ===
    _db: Database | None
    _conn: Connection | None
    # Using RLock to prevent deadlocks when methods call each other
    # (e.g., vector_search -> _fetch_node_metadata -> _query_nodes all need the lock)
    _lock: RLock
    _nodes_count: int
    _table_count: int

    def _ensure_initialized(self) -> Connection:
        """
        Ensure the backend is initialized and return the connection.

        Raises:
            BackendNotInitializedError: If initialize() has not been called.

        Returns:
            The active Connection instance.
        """
        if not self._conn:
            details = "KuzuBackend.initialize() must be called before use"
            raise RuntimeError(details)
        return self._conn

    # =============================================================================
    # SECTION: Schema & Initialization
    # Supports: initialize(), rebuild_fts_indexes()
    # =============================================================================

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
            with suppress(RuntimeError):
                conn.execute(f"ALTER TABLE {table} ADD properties_json STRING DEFAULT ''")

        conn.execute(f"CREATE NODE TABLE IF NOT EXISTS Embedding({_EMBEDDING_PROPERTIES})")

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

    def _build_index_for_table(self, table: str) -> str:
        """Build FTS index for a single table using a dedicated connection."""
        # Each thread gets its own connection from the same Database
        if not (db := self._db):
            details = "Database not initialized"
            raise RuntimeError(details)

        conn = Connection(db)
        try:
            idx_name = f"{table.lower()}_fts"
            with suppress(Exception), self._lock:
                conn.execute(f"CALL DROP_FTS_INDEX('{table}', '{idx_name}')")
            with self._lock:
                conn.execute(
                    f"CALL CREATE_FTS_INDEX('{table}', '{idx_name}', "
                    f"['name', 'content', 'signature'])",
                )
            return table
        finally:
            del conn  # Release connection

    # =============================================================================
    # SECTION: Node Insertion
    # Supports: add_nodes()
    # =============================================================================

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
            f"is_exported: $is_exported, cohesion: $cohesion, "
            f"properties_json: $properties_json"
            f"}})"
        )
        props = node.properties or {}
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
            "cohesion": props.get("cohesion"),
            "properties_json": _serialize_extra_props(props),
        }
        try:
            conn.execute(query, parameters=params)
        except RuntimeError:
            logger.debug("Insert node failed for %s", node.id, exc_info=True)

    # =============================================================================
    # SECTION: Relationship Insertion
    # Supports: add_relationships()
    # =============================================================================

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
                "Insert relationship failed: %s -> %s",
                rel.source,
                rel.target,
                exc_info=True,
            )

    # =============================================================================
    # SECTION: Node Query Results
    # Supports: get_node(), get_callers(), get_callees(), get_type_refs(),
    #           get_callers_with_confidence(), get_callees_with_confidence()
    # =============================================================================

    def _query_nodes(self, query: str, parameters: dict[str, Any] | None = None) -> list[GraphNode]:
        """Execute a query returning ``n.*`` columns and convert to GraphNode list."""
        conn = self._ensure_initialized()
        nodes: list[GraphNode] = []
        try:
            with self._lock:
                result = conn.execute(query, parameters=parameters or {})
            if not isinstance(result, QueryResult):
                return []
            while result.has_next():
                row = result.get_next()
                if not isinstance(row, list):
                    continue
                node = self._row_to_node(row)
                nodes.append(node) if node else None
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
            with self._lock:
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
        9=is_dead, 10=is_entry_point, 11=is_exported, 12=cohesion,
        13=properties_json
        """
        try:
            nid = node_id or row[0]
            prefix = nid.split(":", 1)[0]
            label = _LABEL_MAP.get(prefix)
            if not label:
                logger.warning("Unknown node label prefix %r in id %s", prefix, nid)
                return

            props: dict[str, Any] = {}
            if len(row) > 12 and row[12] is not None:
                props["cohesion"] = float(row[12])

            if len(row) > 13 and row[13]:
                try:
                    extra = loads(row[13])
                    if isinstance(extra, dict):
                        props.update(extra)
                except (ValueError, TypeError):
                    pass

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
                properties=props,
            )
        except (IndexError, KeyError):
            logger.debug("Failed to convert row to GraphNode: %s", row, exc_info=True)
            return None

    # =============================================================================
    # SECTION: Edge Parsing
    # Supports: get_inbound_cross_file_edges(), _load_relationships_to_graph()
    # =============================================================================

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

    # =============================================================================
    # SECTION: Search Results
    # Supports: fts_search(), fts_search_async()
    # =============================================================================

    def _parse_fts_result(
        self,
        result: QueryResult | list[QueryResult] | None,
    ) -> list[SearchResult]:
        """
        Parse FTS query result rows into SearchResult list.

        Common helper for both sync and async FTS search methods.

        Args:
            result: The QueryResult from a FTS query.

        Returns:
            List of SearchResult objects.
        """
        candidates: list[SearchResult] = []
        if not isinstance(result, QueryResult):
            return candidates

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

        return candidates

    # =============================================================================
    # SECTION: Bulk Loading
    # Supports: bulk_load()
    # =============================================================================

    def _csv_copy(self, table: str, rows: list[list[Any]]) -> None:
        """
        Write *rows* to a temporary CSV and COPY FROM into *table*.

        Uses PARALLEL=FALSE to avoid concurrency issues with KuzuDB's
        parallel CSV reader.  Always cleans up the temp file, even on failure.
        """
        conn = self._ensure_initialized()
        csv_path: str | None = None
        try:
            with NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
                writer = csv_writer(f)
                writer.writerows(rows)
                csv_path = f.name
            conn.execute(f"COPY {table} FROM '{csv_path}' (HEADER=false, PARALLEL=false)")
        finally:
            if csv_path:
                Path(csv_path).unlink(missing_ok=True)

    def _bulk_load_nodes_csv(self, graph: KnowledgeGraph, pbar: tqdm) -> bool:
        """
        Load all nodes via temporary CSV files + COPY FROM.

        Returns True on success, False if COPY FROM is not available.
        """
        # logger.info("Bulk loading nodes via temporary CSV files...")
        pbar = reset_pbar(pbar, self._nodes_count, "Collecting nodes table")
        by_table: dict[str, list[GraphNode]] = {}

        for node in graph.iter_nodes():
            table = _LABEL_TO_TABLE.get(node.label.value)
            by_table.setdefault(table, []).append(node) if table else None
            pbar.update()

        pbar = reset_pbar(pbar, len(by_table), "2. Bulk loading nodes")
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
                            (node.properties or {}).get("cohesion"),
                            _serialize_extra_props(node.properties),
                        ]
                        for node in nodes
                    ],
                )
                pbar.update()
        except RuntimeError:
            logger.debug("CSV nodes load bulk failed, falling back", exc_info=True)
            conn = self._ensure_initialized()
            for table in by_table:
                with suppress(RuntimeError):
                    conn.execute(f"MATCH (n:{table}) DETACH DELETE n")
            return False
        return True

    def _bulk_load_rels_csv(self, graph: KnowledgeGraph, pbar: tqdm) -> bool:
        """
        Load all relationships via temporary CSV files + COPY FROM.

        Returns True on success, False if COPY FROM is not available.
        """
        # logger.info("Bulk loading relationships via temporary CSV files...")
        pbar = reset_pbar(pbar, self._nodes_count, "Collecting relationships table")
        by_pair: dict[tuple[str, str], list[GraphRelationship]] = {}

        for rel in graph.iter_relationships():
            src_table = _table_for_id(rel.source)
            dst_table = _table_for_id(rel.target)
            if src_table and dst_table:
                by_pair.setdefault((src_table, dst_table), []).append(rel)
            pbar.update()

        pbar = reset_pbar(pbar, len(by_pair), "3. Bulk loading relationship")
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
                pbar.update()
        except RuntimeError:
            logger.debug("CSV relationship load bulk failed, falling back", exc_info=True)
            return False

        return True

    # =============================================================================
    # SECTION: Embedding Storage
    # Supports: store_embeddings(), upsert_embeddings()
    # =============================================================================

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

    # =============================================================================
    # SECTION: Vector Search
    # Supports: vector_search()
    # =============================================================================

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
                with self._lock:
                    nodes = self._query_nodes(q, parameters={"ids": ids})
                if isinstance(nodes, QueryResult):
                    return node_cache
                for node in nodes:
                    node_cache[node.id] = node
            except RuntimeError:
                logger.debug(f"Batch node fetch failed for table {table}", exc_info=True)
        return node_cache

    # =============================================================================
    # SECTION: Graph Loading
    # Supports: load_graph()
    # =============================================================================

    def _load_nodes_to_graph(self, graph: KnowledgeGraph) -> None:
        """Load all nodes from every table into the graph."""
        for table in _NODE_TABLE_NAMES:
            try:
                # Use _query_nodes which handles the row-to-node conversion
                for node in self._query_nodes(f"MATCH (n:{table}) RETURN n.*"):
                    graph.add_node(node)
            except RuntimeError:
                logger.debug(f"load_graph: failed to read table {table}", exc_info=True)

    def _load_relationships_to_graph(self, graph: KnowledgeGraph) -> None:
        """Load all relationships from the database into the graph."""
        conn = self._ensure_initialized()
        try:
            with self._lock:
                result = conn.execute(
                    "MATCH (a)-[r:CodeRelation]->(b) "
                    "RETURN a.id, b.id, r.rel_type, r.confidence, r.role, "
                    "r.step_number, r.strength, r.co_changes, r.symbols",
                )
            if not isinstance(result, QueryResult):
                return
            while result.has_next():
                if not isinstance((row := result.get_next()), list):
                    continue
                # Mapping result format to _parse_edge_row expected format
                # _parse_edge_row expects: row[0]=src_id, row[1]=src_file(unused), row[2]=tgt_id, row[3...]=props
                src_id, tgt_id = row[0], row[1]
                mapped_row = [src_id, None, tgt_id] + row[2:]
                if rel := self._parse_edge_row(mapped_row):
                    graph.add_relationship(rel)
        except Exception:
            logger.error(
                "load_graph: relationship query failed — graph incomplete",
                exc_info=True,
            )
            raise


class KuzuBackend(_Helpers):
    """
    StorageBackend implementation backed by KuzuDB.

        Usage::

            backend = KuzuBackend()
            backend.initialize(Path("/tmp/axon_db"))
            backend.bulk_load(graph)
            node = backend.get_node("function:src/app.py:main")
            backend.close()
    """

    _MAX_BFS_DEPTH = 10

    def __init__(self) -> None:
        self._db: Database | None = None
        self._conn: Connection | None = None
        # Using RLock to prevent deadlocks when methods call each other
        # (e.g., vector_search -> _fetch_node_metadata -> _query_nodes all need the lock)
        self._lock: RLock = RLock()
        self._nodes_count: int = 0
        self._table_count: int = len(_NODE_TABLE_NAMES)

    def add_nodes(self, nodes: list[GraphNode], pbar: tqdm | None = None) -> None:
        """Insert nodes into their respective label tables."""
        if pbar:
            pbar = reset_pbar(pbar, len(nodes), "2. Adding nodes directly")

        for node in nodes:
            self._insert_node(node)
            pbar.update() if pbar else None

    def add_relationships(self, rels: list[GraphRelationship], pbar: tqdm | None = None) -> None:
        """Insert relationships by matching source and target nodes."""
        if pbar:
            pbar = reset_pbar(pbar, len(rels), "3. Adding relationships directly")

        for rel in rels:
            self._insert_relationship(rel)
            pbar.update() if pbar else None

    def bulk_load(self, graph: KnowledgeGraph) -> None:
        """
        Replace the entire store with the contents of *graph*.

        Uses CSV-based COPY FROM for bulk loading nodes and relationships,
        falling back to individual inserts if COPY FROM fails.
        """
        conn = self._ensure_initialized()
        self._nodes_count = graph.node_count
        pbar = p_bar(desc="1. Deleting detached tables", total=self._table_count)
        for table in _NODE_TABLE_NAMES:
            with suppress(Exception):
                conn.execute(f"MATCH (n:{table}) DETACH DELETE n")
            pbar.update()

        if not self._bulk_load_nodes_csv(graph, pbar):
            self.add_nodes(list(graph.iter_nodes()), pbar=pbar)

        if not self._bulk_load_rels_csv(graph, pbar):
            self.add_relationships(list(graph.iter_relationships()), pbar=pbar)

        self.rebuild_fts_indexes(pbar)
        pbar.set_description_str("Completed")
        pbar.close()

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

    def delete_synthetic_nodes(self) -> None:
        """Remove all COMMUNITY and PROCESS nodes and their relationships."""
        conn = self._ensure_initialized()
        for table in ("Community", "Process"):
            try:
                conn.execute(f"MATCH (n:{table}) DETACH DELETE n")
            except RuntimeError:
                logger.debug("delete_synthetic_nodes: failed for %s", table, exc_info=True)

    def execute_raw(self, query: str) -> list[list[Any]]:
        """Execute a raw Cypher query and return all result rows."""
        conn = self._ensure_initialized()
        rows: list[list[Any]] = []
        with self._lock:
            result = conn.execute(query)
        if not isinstance(result, QueryResult):
            return rows
        while result.has_next():
            if not isinstance((_result := result.get_next()), list):
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
                with self._lock:
                    result = conn.execute(cypher, parameters={"name": name})
                if not isinstance(result, QueryResult):
                    continue
                while result.has_next():
                    if not isinstance((row := result.get_next()), list):
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
                logger.debug("exact_name_search failed on table %s", table, exc_info=True)

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
        escaped_q = escape_cypher(query)
        candidates: list[SearchResult] = []

        # NOTE: QUERY_FTS_INDEX is a KuzuDB stored procedure that does not support
        # parameterized $variables. String interpolation with escape_cypher() is the
        # only option here. escape_cypher strips comments, semicolons, and escapes quotes.
        for table in _SEARCHABLE_TABLES:
            idx_name = f"{table.lower()}_fts"
            cypher = (
                f"CALL QUERY_FTS_INDEX('{table}', '{idx_name}', '{escaped_q}') "
                f"RETURN node.id, node.name, node.file_path, node.content, "
                f"node.signature, score "
                f"ORDER BY score DESC LIMIT {limit}"
            )
            try:
                with self._lock:
                    result = conn.execute(cypher)

                candidates.extend(self._parse_fts_result(result))
            except RuntimeError:
                logger.debug("fts_search failed on table %s", table, exc_info=True)

        return sorted(candidates, key=lambda r: (-r.score, r.node_id))[:limit]

    async def fts_search_async(self, query: str, limit: int) -> list[SearchResult]:
        """
        Async BM25 full-text search using KuzuDB's native FTS extension.

        Searches across all node tables using pre-built FTS indexes on
        ``name``, ``content``, and ``signature`` fields. Results are
        ranked by BM25 relevance score.

        Uses AsyncConnection to query all tables in parallel for better
        performance with high concurrency workloads.

        Returns the top *limit* results sorted by score descending.
        """
        if not self._db:
            details = "KuzuBackend.initialize() must be called before use"
            logger.error(details)
            raise RuntimeError(details)

        # Use AsyncConnection with concurrent query support
        conn = AsyncConnection(self._db, max_concurrent_queries=len(_SEARCHABLE_TABLES))
        escaped_q = escape_cypher(query)

        async def search_single_table(table: str) -> list[SearchResult]:
            """Search a single table asynchronously."""
            idx_name = f"{table.lower()}_fts"
            cypher = (
                f"CALL QUERY_FTS_INDEX('{table}', '{idx_name}', '{escaped_q}') "
                f"RETURN node.id, node.name, node.file_path, node.content, "
                f"node.signature, score "
                f"ORDER BY score DESC LIMIT {limit}"
            )
            try:
                result = await conn.execute(cypher)
                return self._parse_fts_result(result)
            except RuntimeError:
                logger.debug("fts_search_async failed for table %s", table, exc_info=True)
                return []
            finally:
                conn.close()

        # Execute all table searches concurrently
        results = await gather(
            *[search_single_table(table) for table in _SEARCHABLE_TABLES],
        )

        # Merge and sort results
        all_candidates: list[SearchResult] = [item for sublist in results for item in sublist]
        return sorted(all_candidates, key=lambda r: (-r.score, r.node_id))[:limit]

    def fuzzy_search(self, query: str, limit: int, max_distance: int = 2) -> list[SearchResult]:
        """
        Fuzzy name search using Levenshtein edit distance.

        Scans all node tables for symbols whose name is within
        *max_distance* edits of *query*.  Converts edit distance to a
        score (0 edits = 1.0, *max_distance* edits = 0.3).
        """
        conn = self._ensure_initialized()
        candidates: list[SearchResult] = []

        for table in _SEARCHABLE_TABLES:
            cypher = (
                f"MATCH (n:{table}) "
                f"WHERE levenshtein(lower(n.name), $q) <= $dist "
                f"RETURN n.id, n.name, n.file_path, n.content, "
                f"levenshtein(lower(n.name), $q) AS dist "
                f"ORDER BY dist LIMIT $lim"
            )
            try:
                with self._lock:
                    result = conn.execute(
                        cypher,
                        parameters={"q": query.lower(), "dist": max_distance, "lim": limit},
                    )
                if not isinstance(result, QueryResult):
                    continue
                while result.has_next():
                    if not isinstance((row := result.get_next()), list):
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
                logger.debug("fuzzy_search failed on table %s", table, exc_info=True)

        candidates.sort(key=lambda r: (-r.score, r.node_id))
        return candidates[:limit]

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

    def get_file_index(self) -> dict[str, str]:
        """Return ``{file_path: node_id}`` for all File nodes."""
        conn = self._ensure_initialized()
        index: dict[str, str] = {}
        try:
            with self._lock:
                result = conn.execute("MATCH (n:File) RETURN n.file_path, n.id")
            if not isinstance(result, QueryResult):
                return index
            while result.has_next():
                if not isinstance((row := result.get_next()), list):
                    continue
                index[row[0]] = row[1]
        except RuntimeError:
            logger.debug("get_file_index failed", exc_info=True)
        return index

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
            with self._lock:
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
                if not isinstance((row := result.get_next()), list):
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
            with self._lock:
                if not isinstance(
                    (result := conn.execute("MATCH (n:File) RETURN n.file_path, n.content")),
                    QueryResult,
                ):
                    return {}
            while result.has_next():
                if not isinstance((row := result.get_next()), list):
                    continue
                fp = row[0] or ""
                content = row[1] or ""
                mapping[fp] = sha256(content.encode()).hexdigest()
        except RuntimeError:
            logger.debug("get_indexed_files failed", exc_info=True)
        return mapping

    def get_node(self, node_id: str) -> GraphNode | None:
        """Return a single node by ID, or ``None`` if not found."""
        conn = self._ensure_initialized()
        table = _table_for_id(node_id)
        if table is None:
            return None

        query = f"MATCH (n:{table}) WHERE n.id = $nid RETURN n.*"
        try:
            with self._lock:
                result = conn.execute(query, parameters={"nid": node_id})
            if (
                isinstance(result, QueryResult)
                and result.has_next()
                and isinstance((row := result.get_next()), list)
            ):
                return self._row_to_node(row, node_id)
        except RuntimeError:
            logger.warning("get_node failed for %s", node_id, exc_info=True)
        return None

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
            with self._lock:
                result = conn.execute(
                    "MATCH (n)-[r:CodeRelation]->(p:Process) "
                    "WHERE n.id IN $ids AND r.rel_type = 'step_in_process' "
                    "RETURN n.id, p.name",
                    parameters={"ids": node_ids},
                )
            if not isinstance(result, QueryResult):
                return mapping
            while result.has_next():
                if not isinstance((row := result.get_next()), list):
                    continue
                nid = row[0] if row else ""
                pname = row[1] if len(row) > 1 else ""
                if nid and pname and nid not in mapping:
                    mapping[nid] = pname
        except RuntimeError:
            logger.warning("get_process_memberships failed", exc_info=True)
        return mapping

    def get_symbol_name_index(self) -> dict[str, list[str]]:
        """Return ``{symbol_name: [node_id, ...]}`` for callable/type symbols."""
        conn = self._ensure_initialized()
        index: dict[str, list[str]] = {}
        tables = ["Function", "Method", "Class", "Interface", "TypeAlias"]
        for table in tables:
            try:
                with self._lock:
                    result = conn.execute(f"MATCH (n:{table}) RETURN n.name, n.id")
                if not isinstance(result, QueryResult):
                    return index
                while result.has_next():
                    if not isinstance((row := result.get_next()), list):
                        continue
                    index.setdefault(row[0], []).append(row[1])
            except RuntimeError:
                logger.debug("get_symbol_name_index failed for %s", table, exc_info=True)
        return index

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

    def initialize(
        self,
        path: Path,
        *,
        read_only: bool = False,
        max_retries: int = 0,
        retry_delay: float = 0.3,
    ) -> None:
        """
        Open or create the KuzuDB database at *path*.

        In read-only mode, schema creation is skipped (database must already exist).
        Retries on lock contention errors with exponential backoff.
        """
        for attempt in range(max_retries + 1):
            try:
                self._db = Database(str(path), read_only=read_only)
                self._conn = Connection(self._db)
                if not read_only:
                    self._create_schema()
                return
            except RuntimeError as e:
                if "lock" in str(e).lower() and attempt < max_retries:
                    logger.debug(
                        "Lock contention on attempt %d/%d, retrying in %.1fs",
                        attempt + 1,
                        max_retries,
                        retry_delay * (2**attempt),
                    )
                    self.close()
                    sleep(retry_delay * (2**attempt))
                    continue
                raise

    def load_graph(self) -> KnowledgeGraph:
        """Reconstruct a full :class:`KnowledgeGraph` from the database."""
        self._ensure_initialized()
        graph = KnowledgeGraph()
        self._load_nodes_to_graph(graph)
        self._load_relationships_to_graph(graph)
        return graph

    def rebuild_fts_indexes(self, pbar: tqdm | None = None, max_workers: int = 4) -> None:
        """
        Drop and recreate all FTS indexes.

        Must be called after any bulk data change so the BM25 indexes
        reflect the current node contents.

        Uses ThreadPoolExecutor to build FTS indexes in parallel for each table.
        Each thread gets its own connection from the same Database for thread safety.
        """
        self._ensure_initialized()
        if pbar:
            pbar = reset_pbar(pbar, self._table_count, "4. Rebuilding FTS indexes")
        else:
            pbar = tqdm(desc="Rebuilding FTS indexes", total=self._table_count)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._build_index_for_table, table): table
                for table in _NODE_TABLE_NAMES
            }
            for future in as_completed(futures):
                try:
                    future.result()
                    pbar.update() if pbar else None
                except RuntimeError:
                    logger.debug(f"FTS index rebuild failed for {futures[future]}", exc_info=True)

    def remove_nodes_by_file(self, file_path: str) -> int:
        """
        Delete all nodes whose ``file_path`` matches across every table.

        Returns:
            Always returns 0 — count is not tracked for performance reasons.
        """
        conn = self._ensure_initialized()
        total = 0
        for table in _NODE_TABLE_NAMES:
            try:
                count_result = conn.execute(
                    f"MATCH (n:{table}) WHERE n.file_path = $fp RETURN count(n)",
                    parameters={"fp": file_path},
                )
                if isinstance(count_result, QueryResult) and count_result.has_next():
                    if not isinstance((row := count_result.get_next()), list):
                        continue
                    total += int(row[0] or 0)
                conn.execute(
                    f"MATCH (n:{table}) WHERE n.file_path = $fp DETACH DELETE n",
                    parameters={"fp": file_path},
                )
            except RuntimeError:
                logger.debug("Failed to remove nodes from table %s", table, exc_info=True)
        return total

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
                "remove_relationships_by_type failed for %s",
                rel_type.value,
                exc_info=True,
            )

    def store_embeddings(self, embeddings: list[NodeEmbedding]) -> None:
        """
        Persist embedding vectors into the Embedding node table.

        Attempts batch CSV COPY FROM first, falls back to individual MERGE.
        """
        conn = self._ensure_initialized()
        if not embeddings:
            rprint("\n[b blue]No embeddings to store")
            return

        if self._bulk_store_embeddings_csv(embeddings):
            rprint("\n[b blue]Embeddings stored via CSV COPY")
            return

        for emb in embeddings:
            try:
                conn.execute(
                    "MERGE (e:Embedding {node_id: $nid}) SET e.vec = $vec",
                    parameters={"nid": emb.node_id, "vec": emb.embedding},
                )
            except RuntimeError:
                logger.debug("store_embeddings failed for node %s", emb.node_id, exc_info=True)
        rprint("\n[b blue]Embeddings stored via individual MERGE")

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
                    logger.debug("update_dead_flags failed for table %s", table, exc_info=True)

        _batch_set(dead_ids, value=True)
        _batch_set(alive_ids, value=False)

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
                logger.debug("upsert_embeddings failed for node %s", emb.node_id, exc_info=True)

    def vector_search(self, vector: list[float], limit: int) -> list[SearchResult]:
        """
        Find the closest nodes to *vector* using native ``array_cosine_similarity``.

        Computes cosine similarity directly in KuzuDB's Cypher engine —
        no Python-side computation or full-table load required.  Joins with
        node tables to fetch metadata in a single query.
        """
        conn = self._ensure_initialized()
        vec_literal = _safe_vec_literal(vector)

        try:
            with self._lock:
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
        if not isinstance(result, QueryResult):
            return []
        while result.has_next():
            if not isinstance((row := result.get_next()), list):
                continue
            emb_rows.append((row[0] or "", float(row[1]) if row[1] is not None else 0.0))

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
