"""
MCP server for Axon — exposes code intelligence tools over stdio transport.

Registers seven tools and three resources that give AI agents and MCP clients
access to the Axon knowledge graph.  The server lazily initialises a
:class:`KuzuBackend` from the ``.axon/kuzu`` directory in the current
working directory.

Usage::

    # MCP server only
    axon mcp

    # MCP server with live file watching (recommended)
    axon serve --watch
"""

from asyncio import Lock, run, to_thread
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from logging import getLogger
from pathlib import Path

from mcp.server import Server
from mcp.server.fastmcp.server import StreamableHTTPASGIApp
from mcp.server.stdio import stdio_server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import Resource, TextContent, Tool
from pydantic import AnyUrl

from axon.core.storage.kuzu_backend import KuzuBackend
from axon.mcp.resources import get_dead_code_list, get_overview, get_schema
from axon.mcp.tools import MAX_TRAVERSE_DEPTH, Tools

logger = getLogger(__name__)

server = Server("axon")

_storage: KuzuBackend | None = None
_lock: Lock | None = None
_db_path: Path | None = None

_tools = Tools()


def _resolve_db_path() -> Path:
    global _db_path
    if _db_path is None:
        _db_path = Path.cwd() / ".axon" / "kuzu"
    return _db_path


def set_storage(storage: KuzuBackend) -> None:
    """Inject a pre-initialised storage backend (e.g. from ``axon serve --watch``)."""
    global _storage
    _storage = storage


def set_lock(lock: Lock) -> None:
    """Inject a shared lock for coordinating storage access with the file watcher."""
    global _lock
    _lock = lock


@contextmanager
def _open_storage() -> Iterator[KuzuBackend]:
    """
    Open a short-lived read-only connection for a single tool/resource call.

    Used when no persistent storage was injected (read-only fallback mode).
    Each call gets a fresh connection that sees the latest on-disk data and
    releases the file lock immediately after the query completes.
    """
    db_path = _resolve_db_path()
    if not db_path.exists():
        details = f"No .axon/kuzu directory in {db_path.parent.parent}"
        raise FileNotFoundError(details)
    storage = KuzuBackend()
    storage.initialize(db_path, read_only=True, max_retries=3, retry_delay=0.3)
    try:
        yield storage
    finally:
        storage.close()


def _run(fn: Callable[[KuzuBackend], dict[str, str]]) -> dict[str, str]:
    with _open_storage() as st:
        return fn(st)


async def _with_storage(fn: Callable[[KuzuBackend], dict[str, str]]) -> dict[str, str]:
    """
    Run *fn* against the appropriate storage backend.

    Uses the injected persistent backend when available (with optional
    async lock), otherwise opens a short-lived read-only connection.
    """

    if not _storage:
        return await to_thread(_run, fn)

    if not _lock:
        return await to_thread(fn, _storage)

    async with _lock:
        return await to_thread(fn, _storage)


TOOLS: list[Tool] = [
    Tool(
        name="axon_list_repos",
        description="List all indexed repositories with their stats.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="axon_query",
        description=(
            "Search the knowledge graph using hybrid (keyword + vector) search. "
            "Returns ranked symbols matching the query."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default 20).",
                    "default": 20,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="axon_context",
        description=(
            "Get a 360-degree view of a symbol: callers, callees, type references, "
            "and community membership."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Name of the symbol to look up.",
                },
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="axon_impact",
        description=(
            "Blast radius analysis: find all symbols affected by changing a given symbol."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Name of the symbol to analyse.",
                },
                "depth": {
                    "type": "integer",
                    "description": f"Maximum traversal depth (default 3, max {MAX_TRAVERSE_DEPTH}).",
                    "default": 3,
                    "minimum": 1,
                    "maximum": MAX_TRAVERSE_DEPTH,
                },
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="axon_dead_code",
        description="List all symbols detected as dead (unreachable) code.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="axon_detect_changes",
        description=(
            "Parse a git diff and map changed files/lines to affected symbols "
            "in the knowledge graph."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "diff": {
                    "type": "string",
                    "description": "Raw git diff output.",
                },
            },
            "required": ["diff"],
        },
    ),
    Tool(
        name="axon_cypher",
        description="Execute a raw Cypher query against the knowledge graph.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Cypher query string.",
                },
            },
            "required": ["query"],
        },
    ),
]


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return the list of available Axon tools."""
    return TOOLS


def _dispatch_tool(name: str, arguments: dict, storage: KuzuBackend) -> dict[str, str]:
    """Sync tool dispatch — called directly or via ``to_thread``."""
    return {
        "axon_list_repos": _tools.handle_list_repos(),
        "axon_query": _tools.handle_query(
            storage,
            arguments.get("query", ""),
            limit=arguments.get("limit", 20),
        ),
        "axon_context": _tools.handle_context(storage, arguments.get("symbol", "")),
        "axon_impact": _tools.handle_impact(
            storage,
            arguments.get("symbol", ""),
            depth=arguments.get("depth", 3),
        ),
        "axon_dead_code": _tools.handle_dead_code(storage),
        "axon_detect_changes": _tools.handle_detect_changes(storage, arguments.get("diff", "")),
        "axon_cypher": _tools.handle_cypher(storage, arguments.get("query", "")),
        "axon_coupling": _tools.handle_coupling(
            storage,
            arguments.get("file_path", ""),
            min_strength=arguments.get("min_strength", 0.3),
        ),
        "axon_communities": _tools.handle_communities(
            storage,
            community=arguments.get("community"),
        ),
        "axon_explain": _tools.handle_explain(storage, arguments.get("symbol", "")),
        "axon_review_risk": _tools.handle_review_risk(storage, arguments.get("diff", "")),
        "axon_call_path": _tools.handle_call_path(
            storage,
            arguments.get("from_symbol", ""),
            arguments.get("to_symbol", ""),
            max_depth=arguments.get("max_depth", 10),
        ),
        "axon_file_context": _tools.handle_file_context(storage, arguments.get("file_path", "")),
        "axon_test_impact": _tools.handle_test_impact(
            storage,
            diff=arguments.get("diff", ""),
            symbols=arguments.get("symbols"),
        ),
        "axon_cycles": _tools.handle_cycles(storage, min_size=arguments.get("min_size", 2)),
        "unknown": f"Unknown tool: {name}",
    }


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch a tool call to the appropriate handler."""
    try:
        result = await _with_storage(lambda st: _dispatch_tool(name, arguments, st))
    except Exception as exc:
        logger.exception("Tool %s raised an unhandled exception", name)
        result = {"unknown": f"Internal error: {exc}"}

    return [TextContent(type="text", text=result.get(name, f"Unknown tool: {name}"))]


@server.list_resources()
async def list_resources() -> list[Resource]:
    """Return the list of available Axon resources."""
    return [
        Resource(
            uri="axon://overview",
            name="Codebase Overview",
            description="High-level statistics about the indexed codebase.",
            mimeType="text/plain",
        ),
        Resource(
            uri="axon://dead-code",
            name="Dead Code Report",
            description="List of all symbols flagged as unreachable.",
            mimeType="text/plain",
        ),
        Resource(
            uri="axon://schema",
            name="Graph Schema",
            description="Description of the Axon knowledge graph schema.",
            mimeType="text/plain",
        ),
    ]


def _dispatch_resource(uri_str: str, storage: KuzuBackend) -> dict[str, str]:
    """Synch resource dispatch."""
    return {
        "axon://overview": get_overview(storage),
        "axon://dead-code": get_dead_code_list(storage),
        "axon://schema": get_schema(),
        "unknown": f"Unknown resource: {uri_str}",
    }


@server.read_resource()
async def read_resource(uri: AnyUrl) -> str:
    """Read the contents of an Axon resource."""
    uri_str = str(uri)
    result = await _with_storage(lambda st: _dispatch_resource(uri_str, st))
    return result.get(uri_str, result["unknown"])


async def main() -> None:
    """Run the Axon MCP server over stdio transport."""
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def create_streamable_http_app() -> tuple[StreamableHTTPSessionManager, StreamableHTTPASGIApp]:
    """Create a streamable HTTP transport for the existing MCP server."""
    session_manager = StreamableHTTPSessionManager(app=server)
    return session_manager, StreamableHTTPASGIApp(session_manager)


if __name__ == "__main__":
    run(main())
