"""
FastAPI application factory for the Axon Web UI.

Creates a configured FastAPI app that wraps the StorageBackend,
serves API routes, and optionally mounts the frontend SPA.
"""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from logging import getLogger
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from httpx import AsyncClient, ReadError, Timeout
from httpx import Response as HttpxResponse
from mcp.server.fastmcp.server import StreamableHTTPASGIApp
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.routing import Route

from axon.core.storage.kuzu_backend import KuzuBackend
from axon.mcp.server import create_streamable_http_app
from axon.runtime import AxonRuntime
from axon.web.routes.analysis import router as analysis_router
from axon.web.routes.cypher import router as cypher_router
from axon.web.routes.diff import router as diff_router
from axon.web.routes.events import router as events_router
from axon.web.routes.files import router as files_router
from axon.web.routes.graph import router as graph_router
from axon.web.routes.host import router as host_router
from axon.web.routes.processes import router as processes_router
from axon.web.routes.search import router as search_router

logger = getLogger(__name__)


FRONTEND_DIR = Path(__file__).resolve().parent / "frontend" / "dist"


@dataclass
class AppDeps:
    """
    Dependency container for FastAPI application configuration.

    Consolidates all configuration parameters needed to create and configure
    the Axon web application, including database, runtime, and server settings.
    """

    db_path: Path  #: Path to the KuzuDB database directory
    repo_path: Path | None = None  #: Root path of the repository for file serving and reindex
    runtime: AxonRuntime | None = None  #: Optional pre-configured AxonRuntime instance
    host_url: str | None = None  #: Host URL to announce for client connections
    mcp_url: str | None = None  #: MCP server URL to announce
    mount_mcp: bool = False  #: Whether to mount the MCP server endpoints
    watch: bool = False  #: Enable SSE event streaming and file watching for reindex
    dev: bool = False  #: Developer mode (skips static frontend mount)
    mount_frontend: bool = True  #: Mount the frontend SPA at root path


def _create_runtime(deps: AppDeps) -> AxonRuntime:
    """
    Create and initialize the AxonRuntime based on AppDeps.

    Handles two initialization paths:
    1. If a runtime is provided in deps, update its configuration
    2. Otherwise, create a new runtime with storage initialized from db_path

    Args:
        deps: AppDeps containing the desired runtime configuration.

    Returns:
        A configured AxonRuntime instance ready for use.
    """
    # Path 1: Use provided runtime, update its configuration
    if deps.runtime:
        runtime = deps.runtime
        if deps.repo_path:
            runtime.repo_path = deps.repo_path
        runtime.watch = deps.watch
        runtime.host_url = deps.host_url or runtime.host_url
        runtime.mcp_url = deps.mcp_url or runtime.mcp_url
        # Initialize event_listeners if watching is enabled and not already set
        if runtime.event_listeners is None and deps.watch:
            runtime.event_listeners = []
        return runtime

    # Path 2: Create new runtime with storage backend
    storage = KuzuBackend()
    storage.initialize(deps.db_path, read_only=True)
    return AxonRuntime(
        storage=storage,
        repo_path=deps.repo_path,
        watch=deps.watch,
        host_url=deps.host_url,
        mcp_url=deps.mcp_url,
        owns_storage=True,
    )


def _create_mcp_setup(
    deps: AppDeps,
) -> tuple[StreamableHTTPSessionManager | None, StreamableHTTPASGIApp | None]:
    """
    Set up MCP session manager if mount_mcp is enabled.

    Creates the MCP server infrastructure when mounting is requested.

    Args:
        deps: AppDeps containing the mount_mcp flag.

    Returns:
        A tuple of (session_manager, streamable_http_app), or (None, None) if
        MCP is not enabled. The session manager handles MCP connections, and
        the streamable_http_app provides the ASGI interface for MCP requests.
    """
    if deps.mount_mcp:
        return create_streamable_http_app()
    return None, None


def _create_lifespan(
    runtime: AxonRuntime,
    session_manager: StreamableHTTPSessionManager | None,
) -> Callable[[FastAPI], Any]:
    """
    Create the lifespan context manager for the FastAPI app.

    Defines the application lifecycle with two phases:
    1. Startup: Initialize MCP session manager if configured
    2. Shutdown: Close storage backend if owned by the runtime

    Args:
        runtime: The AxonRuntime instance to manage.
        session_manager: Optional MCP session manager for MCP server lifecycle.

    Returns:
        An async context manager function for FastAPI's lifespan parameter.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Startup phase: Run MCP session manager if configured
        if session_manager is not None:
            async with session_manager.run():
                yield
        else:
            yield
        # Shutdown phase: Clean up storage if we own it
        if runtime.owns_storage:
            runtime.storage.close()
            logger.info("Storage backend closed")

    return lifespan


def _setup_app_state(app: FastAPI, runtime: AxonRuntime, deps: AppDeps) -> None:
    """
    Configure the app state with runtime and configuration values.

    Populates FastAPI's app.state with core application state:
    - runtime: The AxonRuntime instance for code intelligence operations
    - storage: The KuzuBackend storage layer
    - repo_path: Root path of the indexed repository
    - event_listeners: Listeners for file change events
    - watch: Whether file watching is enabled
    - host_url: Published host URL for UI access
    - mcp_url: Published MCP server URL
    - mode: 'host' for shared mode, 'standalone' for direct mode

    Args:
        app: The FastAPI application to configure.
        runtime: The AxonRuntime instance to extract state from.
        deps: AppDeps to determine mode setting.
    """
    app.state.runtime = runtime
    app.state.storage = runtime.storage
    app.state.repo_path = runtime.repo_path
    app.state.event_listeners = runtime.event_listeners
    app.state.watch = runtime.watch
    app.state.host_url = runtime.host_url
    app.state.mcp_url = runtime.mcp_url
    app.state.mode = "host" if deps.mount_mcp else "standalone"


def _add_cors_middleware(app: FastAPI) -> None:
    """Add CORS middleware to the FastAPI app."""
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://localhost(:\d+)?",
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
    )


def _register_routers(app: FastAPI) -> None:
    """Register all API routers to the FastAPI app."""
    app.include_router(graph_router, prefix="/api")
    app.include_router(host_router, prefix="/api")
    app.include_router(search_router, prefix="/api")
    app.include_router(analysis_router, prefix="/api")
    app.include_router(files_router, prefix="/api")
    app.include_router(cypher_router, prefix="/api")
    app.include_router(diff_router, prefix="/api")
    app.include_router(processes_router, prefix="/api")
    app.include_router(events_router, prefix="/api")


def _setup_mcp_route(app: FastAPI, streamable_http_app: StreamableHTTPASGIApp | None) -> None:
    """Add MCP route to the app if streamable_http_app is provided."""
    if streamable_http_app is not None:
        app.router.routes.append(Route("/mcp", endpoint=streamable_http_app))


def _mount_frontend(app: FastAPI, deps: AppDeps) -> None:
    """Mount the frontend static files if enabled."""
    if deps.mount_frontend and not deps.dev and FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
        logger.info("Serving frontend from %s", FRONTEND_DIR)
    elif deps.mount_frontend and deps.dev:
        logger.info("Dev mode: skipping static file mount (use Vite on :5173)")


def create_app(deps: AppDeps) -> FastAPI:
    """
    Build and return a fully configured FastAPI application.

    Args:
        deps: AppDeps containing all configuration for the app.

    Returns:
        A ready-to-run FastAPI instance.
    """
    runtime = _create_runtime(deps)
    session_manager, streamable_http_app = _create_mcp_setup(deps)
    lifespan = _create_lifespan(runtime, session_manager)

    app = FastAPI(
        title="Axon Web UI",
        description="Graph-powered code intelligence engine",
        version="1.0.1",
        lifespan=lifespan,
    )

    _setup_app_state(app, runtime, deps)
    _add_cors_middleware(app)
    _register_routers(app)
    _setup_mcp_route(app, streamable_http_app)
    _mount_frontend(app, deps)

    return app


async def _body_header(request: Request) -> tuple[bytes | None, dict[str, str]]:
    """
    Extract request body and filtered headers for proxying.

    Processes incoming request by:
    1. Reading the full request body (for POST/PUT requests)
    2. Filtering headers to exclude hop-by-hop and connection-level headers

    Args:
        request: The incoming FastAPI Request to extract data from.

    Returns:
        Tuple of (request_body, filtered_headers) ready for upstream proxy.
    """
    # Filter out headers that shouldn't be forwarded to upstream
    # - host: will be set by httpx based on the upstream URL
    # - content-length: will be recalculated by httpx
    # - connection: hop-by-hop header that shouldn't be forwarded
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length", "connection"}
    }
    return await request.body(), headers


async def _iter_bytes(upstream_stream: HttpxResponse) -> AsyncIterator[bytes]:
    """
    Iterate over upstream response bytes, handling SSE stream gracefully.

    This iterator handles Server-Sent Events (SSE) streaming from the upstream
    by catching read errors that occur when the upstream closes the connection.

    Args:
        upstream_stream: The httpx response stream from the upstream server.

    Yields:
        Bytes chunks from the upstream response.
    """
    try:
        async for chunk in upstream_stream.aiter_bytes():
            yield chunk
    except ReadError:
        # Normal termination when managed host closes SSE stream
        # This is expected behavior, not an error condition
        logger.debug("Managed host SSE stream closed", exc_info=True)
    finally:
        # Ensure connection is properly closed to prevent resource leaks
        await upstream_stream.aclose()


async def _proxy_request(request: Request, api_base_url: str, path: str = "") -> Response:
    """
    Proxy incoming requests to the upstream Axon host.

    Forwards API requests to a running Axon host instance. Special handling
    is applied for /api/events which uses Server-Sent Events (SSE) streaming.

    Args:
        request: The incoming FastAPI Request to proxy.
        api_base_url: Base URL of the upstream Axon host (e.g., http://localhost:8420).
        path: API path to proxy (appended to api_base_url/api/).

    Returns:
        Response from the upstream server, adapted for the client.
    """
    upstream = f"{api_base_url}/api/{path}".rstrip("/")
    body, headers = await _body_header(request)

    async with AsyncClient(timeout=Timeout(30.0, read=300.0)) as client:
        # Special handling for /api/events endpoint:
        # SSE streams require streaming response to forward events in real-time
        if request.url.path == "/api/events":
            upstream_request = client.build_request(
                request.method,
                upstream,
                params=request.query_params,
                headers=headers,
                content=body if body else None,
            )
            upstream_stream = await client.send(upstream_request, stream=True)
            return _streaming_response(upstream_stream)

        # Standard request handling: wait for complete response
        response = await client.request(
            request.method,
            upstream,
            params=request.query_params,
            headers=headers,
            content=body if body else None,
        )
        return _response(response)


def _streaming_response(response: HttpxResponse) -> StreamingResponse:
    """
    Convert an httpx response to a FastAPI StreamingResponse.

    Adapts the upstream response by forwarding status code, headers (excluding
    hop-by-hop headers), and streaming the body content.

    Args:
        response: The httpx Response from the upstream server.

    Returns:
        A FastAPI StreamingResponse ready to send to the client.
    """
    return StreamingResponse(
        _iter_bytes(response),
        status_code=response.status_code,
        headers={
            key: value
            for key, value in response.headers.items()
            if key.lower() not in {"content-length", "connection"}
        },
        media_type=response.headers.get("content-type"),
    )


def _response(response: HttpxResponse) -> Response:
    """
    Convert an httpx response to a FastAPI Response.

    Adapts the upstream response by forwarding status code, headers (excluding
    hop-by-hop headers), and the complete body content.

    Args:
        response: The complete httpx Response from the upstream server.

    Returns:
        A FastAPI Response ready to send to the client.
    """
    return Response(
        content=response.content,
        status_code=response.status_code,
        headers={
            key: value
            for key, value in response.headers.items()
            if key.lower() not in {"content-length", "connection"}
        },
        media_type=response.headers.get("content-type"),
    )


def create_ui_proxy_app(api_base_url: str, *, dev: bool = False) -> FastAPI:
    """
    Create a UI-only app that proxies API requests to an existing backend.

    This creates a lightweight FastAPI application that serves the frontend UI
    while proxying all API calls to a running Axon host. This is useful when
    the host was started without UI enabled but you want to access the UI.

    The proxy handles two types of requests differently:
    - Standard API requests: Forwarded and waited for response
    - /api/events: Streamed Server-Sent Events for real-time updates

    Args:
        api_base_url: Base URL of the upstream Axon host (e.g., http://localhost:8420).
        dev: If True, skip mounting the frontend (for development with Vite).

    Returns:
        A FastAPI app configured to proxy API requests to the upstream host.
    """
    app = FastAPI(title="Axon UI Proxy", description="UI proxy for a shared Axon backend")

    # Register proxy routes using functools.partial to bind api_base_url
    app.add_api_route(
        "/api",
        partial(_proxy_request, api_base_url=api_base_url),
        methods=["GET", "POST", "OPTIONS"],
    )
    app.add_api_route(
        "/api/{path:path}",
        partial(_proxy_request, api_base_url=api_base_url),
        methods=["GET", "POST", "OPTIONS"],
    )

    if not dev and FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    elif dev:
        logger.info("Dev mode: skipping static file mount (use Vite on :5173)")

    return app
