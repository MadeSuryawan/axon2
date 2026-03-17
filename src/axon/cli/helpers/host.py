"""Helper functions for Axon CLI operations."""

from asyncio import Event, Lock, gather
from asyncio import run as asyncio_run
from asyncio import sleep as asyncio_sleep
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from json import JSONDecodeError, dumps, loads
from logging import getLogger
from os import devnull, getpid, kill
from pathlib import Path
from shutil import rmtree
from subprocess import Popen
from sys import executable
from threading import Timer
from time import sleep, time
from urllib.error import URLError
from urllib.request import urlopen
from uuid import uuid4
from webbrowser import open as web_open

from anyio import create_task_group
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from fastapi import FastAPI
from mcp.client.streamable_http import streamable_http_client
from mcp.server.stdio import stdio_server
from rich.console import Console
from typer import Exit
from uvicorn import Config, Server
from uvicorn import run as uvicorn_run

from axon import __version__
from axon.core.ingestion.pipeline import PipelineResult, Pipelines
from axon.core.ingestion.watcher import Watcher, WatcherDeps
from axon.core.storage.kuzu_backend import KuzuBackend
from axon.mcp.server import set_lock, set_storage
from axon.runtime import AxonRuntime
from axon.web.app import create_app, create_ui_proxy_app

console = Console()
logger = getLogger(__name__)

# Default host configuration
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8420
DEFAULT_MANAGED_PORT = 8421

# Update check configuration
UPDATE_CHECK_INTERVAL_SECONDS = 60 * 60 * 24
UPDATE_CHECK_URL = "https://pypi.org/pypi/axoniq/json"
UPDATE_CHECK_SKIP_COMMANDS = {"mcp", "serve", "host"}


# ==================== Dataclasses for configuration ====================


@dataclass(slots=True)
class HostConfig:
    """Configuration for host processes."""

    port: int = DEFAULT_PORT
    bind: str = DEFAULT_HOST
    watch: bool = True
    timeout_seconds: float = 10.0
    managed: bool = False


# ==================== Storage Functions ====================


def _load_storage(repo_path: Path | None = None) -> KuzuBackend:
    """Load the KuzuDB backend for the given or current repo (read-only)."""
    target = (repo_path or Path.cwd()).resolve()
    db_path = target / ".axon" / "kuzu"
    if not db_path.exists():
        console.print(f"[red]Error:[/red] No index found at {target}. Run 'axon analyze' first.")
        raise Exit(code=1)

    storage = KuzuBackend()
    storage.initialize(db_path, read_only=True)
    return storage


def _initialize_writable_storage(
    repo_path: Path,
    *,
    auto_index: bool = True,
) -> tuple[KuzuBackend, Path, Path]:
    """
    Open the repo database in read-write mode.

    If *auto_index* is False and no index exists, raises Exit instead of
    running the pipeline — callers like ``ui`` should tell the user to run
    ``axon analyze .`` themselves.
    """
    axon_dir = repo_path / ".axon"
    db_path = axon_dir / "kuzu"

    if not auto_index and not (axon_dir / "meta.json").exists():
        console.print(
            "[red]Error:[/red] No index found. Run [cyan]axon analyze .[/cyan] first to index this codebase.",
        )
        raise Exit(code=1)

    axon_dir.mkdir(parents=True, exist_ok=True)

    storage = KuzuBackend()
    storage.initialize(db_path)

    if not (axon_dir / "meta.json").exists():
        console.print("[bold]Running initial index...[/bold]")
        pipelines = Pipelines(repo_path, storage, full=True, embeddings=True)
        pipelines.run_pipelines()
        result = pipelines.result
        meta = _build_meta(result, repo_path)
        meta_path = axon_dir / "meta.json"
        meta_path.write_text(dumps(meta, indent=2) + "\n", encoding="utf-8")
        try:
            _register_in_global_registry(meta, repo_path)
        except (RuntimeError, OSError, SystemError, PermissionError):
            logger.debug("Failed to register repo in global registry", exc_info=True)

    return storage, axon_dir, db_path


# ==================== Update Check Functions ====================


def _update_cache_path() -> Path:
    return Path.home() / ".axon" / "update-check.json"


def _parse_version_parts(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for raw_part in version.split("."):
        digits = "".join(ch for ch in raw_part if ch.isdigit())
        parts.append(int(digits or 0))
    return tuple(parts)


def _is_newer_version(candidate: str, current: str) -> bool:
    return _parse_version_parts(candidate) > _parse_version_parts(current)


def _read_update_cache() -> dict | None:
    cache_path = _update_cache_path()
    if not cache_path.exists():
        return None
    try:
        return loads(cache_path.read_text(encoding="utf-8"))
    except (JSONDecodeError, OSError):
        return None


def _write_update_cache(payload: dict) -> None:
    cache_path = _update_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(dumps(payload, indent=2) + "\n", encoding="utf-8")


def _fetch_latest_version() -> str | None:
    try:
        with urlopen(UPDATE_CHECK_URL, timeout=1.5) as response:  # noqa: S310
            payload = loads(response.read().decode("utf-8"))
            return str(payload["info"]["version"])
    except (KeyError, OSError, ValueError, URLError):
        return None


def _get_latest_version() -> str | None:
    now = int(time())
    cache = _read_update_cache()
    if cache is not None:
        checked_at = int(cache.get("checked_at", 0))
        latest = cache.get("latest_version")
        if latest and now - checked_at < UPDATE_CHECK_INTERVAL_SECONDS:
            return str(latest)

    latest = _fetch_latest_version()
    if latest is not None:
        _write_update_cache({"checked_at": now, "latest_version": latest})
    return latest


def maybe_notify_update(invoked_subcommand: str | None) -> None:
    if invoked_subcommand in UPDATE_CHECK_SKIP_COMMANDS:
        return
    if (latest := _get_latest_version()) and _is_newer_version(latest, __version__):
        console.print(
            f"[yellow]Update available:[/yellow] Axon {latest} "
            f"(current {__version__}). Run `pip install -U axoniq`.",
        )


# ==================== Registry Functions ====================


def _register_in_global_registry(meta: dict, repo_path: Path) -> None:
    """
    Write meta.json into ``~/.axon/repos/{slug}/`` for multi-repo discovery.

    Slug is ``{repo_name}`` if that slot is unclaimed or already belongs to
    this repo.  Falls back to ``{repo_name}-{sha256(path)[:8]}`` on collision.
    """
    registry_root = Path.home() / ".axon" / "repos"
    repo_name = repo_path.name

    candidate = registry_root / repo_name
    slug = repo_name
    if candidate.exists():
        existing_meta_path = candidate / "meta.json"
        try:
            existing = loads(existing_meta_path.read_text())
            if existing.get("path") != str(repo_path):
                short_hash = sha256(str(repo_path).encode()).hexdigest()[:8]
                slug = f"{repo_name}-{short_hash}"
        except (JSONDecodeError, OSError):
            rmtree(candidate, ignore_errors=True)  # Clean broken slot before claiming

    # Remove any stale entry for the same repo_path under a different slug.
    if registry_root.exists():
        for old_dir in registry_root.iterdir():
            if not old_dir.is_dir() or old_dir.name == slug:
                continue
            old_meta = old_dir / "meta.json"
            try:
                old_data = loads(old_meta.read_text())
                if old_data.get("path") == str(repo_path):
                    rmtree(old_dir, ignore_errors=True)
            except (JSONDecodeError, OSError):
                continue

    slot = registry_root / slug
    slot.mkdir(parents=True, exist_ok=True)

    registry_meta = dict(meta)
    registry_meta["slug"] = slug
    (slot / "meta.json").write_text(dumps(registry_meta, indent=2) + "\n", encoding="utf-8")


# ==================== Meta Build Functions ====================


def _build_meta(result: PipelineResult, repo_path: Path) -> dict:
    return {
        "version": __version__,
        "name": repo_path.name,
        "path": str(repo_path),
        "stats": {
            "files": result.files,
            "symbols": result.symbols,
            "relationships": result.relationships,
            "clusters": result.clusters,
            "flows": result.processes,
            "dead_code": result.dead_code,
            "coupled_pairs": result.coupled_pairs,
            "embeddings": result.embeddings,
        },
        "last_indexed_at": datetime.now(tz=UTC).isoformat(),
    }


# ==================== Host Meta Functions ====================


def _host_meta_path(repo_path: Path) -> Path:
    return repo_path / ".axon" / "host.json"


def _host_lease_dir(repo_path: Path) -> Path:
    return repo_path / ".axon" / "host-leases"


def _display_host(host: str) -> str:
    return "127.0.0.1" if host in {"0.0.0.0", "::"} else host  # noqa: S104


def _build_host_urls(host: str, port: int) -> tuple[str, str]:
    base = f"http://{_display_host(host)}:{port}"
    return base, f"{base}/mcp"


def _read_host_meta(repo_path: Path) -> dict | None:
    meta_path = _host_meta_path(repo_path)
    if not meta_path.exists():
        return None
    try:
        return loads(meta_path.read_text(encoding="utf-8"))
    except (JSONDecodeError, OSError):
        return None


def _write_host_meta(
    repo_path: Path,
    host_url: str,
    mcp_url: str,
    port: int,
    *,
    ui_enabled: bool,
) -> None:
    meta_path = _host_meta_path(repo_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": getpid(),
        "repo_path": str(repo_path),
        "host_url": host_url,
        "mcp_url": mcp_url,
        "port": port,
        "ui_enabled": ui_enabled,
        "leases_dir": str(_host_lease_dir(repo_path)),
    }
    meta_path.write_text(dumps(payload, indent=2) + "\n", encoding="utf-8")


def _clear_host_meta(repo_path: Path) -> None:
    meta_path = _host_meta_path(repo_path)
    if meta_path.exists():
        meta_path.unlink(missing_ok=True)


# ==================== Lease Management Functions ====================


def create_host_lease(repo_path: Path, lease_type: str) -> Path:
    lease_dir = _host_lease_dir(repo_path)
    lease_dir.mkdir(parents=True, exist_ok=True)
    lease_path = lease_dir / f"{getpid()}-{uuid4().hex}.json"
    payload = {
        "pid": getpid(),
        "type": lease_type,
        "created_at": time(),
    }
    lease_path.write_text(dumps(payload), encoding="utf-8")
    return lease_path


def remove_host_lease(lease_path: Path | None) -> None:
    if lease_path is not None:
        lease_path.unlink(missing_ok=True)


def _pid_is_alive(pid: int) -> bool:
    try:
        kill(pid, 0)
    except OSError:
        return False
    return True


def _count_live_host_leases(repo_path: Path) -> int:
    lease_dir = _host_lease_dir(repo_path)
    if not lease_dir.exists():
        return 0
    live = 0
    for lease_path in lease_dir.glob("*.json"):
        try:
            payload = loads(lease_path.read_text(encoding="utf-8"))
            pid = int(payload.get("pid", 0))
        except (ValueError, JSONDecodeError, OSError):
            lease_path.unlink(missing_ok=True)
            continue
        if _pid_is_alive(pid):
            live += 1
        else:
            lease_path.unlink(missing_ok=True)
    return live


def _is_host_alive(meta: dict, repo_path: Path) -> bool:
    host_url = meta.get("host_url")
    if not host_url:
        return False
    try:
        with urlopen(f"{host_url}/api/host", timeout=1.0) as response:  # noqa: S310
            if response.status != 200:
                return False
            payload = loads(response.read().decode("utf-8"))
            return payload.get("repoPath") == str(repo_path)
    except (OSError, ValueError, URLError):
        return False


def get_live_host_info(repo_path: Path) -> dict | None:
    meta = _read_host_meta(repo_path)
    if meta is None:
        return None
    if _is_host_alive(meta, repo_path):
        return meta
    return None


# ==================== Host Process Management ====================


def _start_host_background(
    repo_path: Path,
    *,
    port: int = DEFAULT_PORT,
    bind: str = DEFAULT_HOST,
    watch: bool = True,
    managed: bool = False,
) -> None:
    """Start a detached shared host process in the background."""
    command = [
        executable,
        "-m",
        "axon.cli.main",
        "host",
        "--port",
        str(port),
        "--bind",
        bind,
        "--no-open",
    ]
    if watch:
        command.append("--watch")
    else:
        command.append("--no-watch")
    if managed:
        command.append("--managed")
    with open(devnull, "wb") as d_null:
        Popen(
            command,
            cwd=repo_path,
            stdout=d_null,
            stderr=d_null,
            start_new_session=True,
        )


def ensure_host_running(
    repo_path: Path,
    config: HostConfig | None = None,
) -> dict:
    """Return live host metadata, starting the shared host if necessary."""
    if config is None:
        config = HostConfig()

    live_host = get_live_host_info(repo_path)
    if live_host is not None:
        return live_host

    _start_host_background(
        repo_path,
        port=config.port,
        bind=config.bind,
        watch=config.watch,
        managed=config.managed,
    )
    deadline = time() + config.timeout_seconds
    while time() < deadline:
        live_host = get_live_host_info(repo_path)
        if live_host is not None:
            return live_host
        sleep(0.2)
    details = "Timed out waiting for Axon host to start."
    raise RuntimeError(details)


# ==================== Version Callback ====================


def _version_callback(*, value: bool) -> None:
    if value:
        console.print(f"Axon v{__version__}")
        raise Exit()


# ==================== MCP Proxy ====================


async def proxy_stdio_to_http_mcp(mcp_url: str) -> None:
    """Bridge a local stdio MCP session to the shared HTTP MCP host."""
    async with (
        stdio_server() as (local_read, local_write),
        streamable_http_client(mcp_url) as (remote_read, remote_write, _),
    ):

        async def _forward(
            reader: MemoryObjectReceiveStream,
            writer: MemoryObjectSendStream,
        ) -> None:
            async with writer:
                async for message in reader:
                    await writer.send(message)

        async with create_task_group() as tg:
            tg.start_soon(_forward, local_read, remote_write)
            tg.start_soon(_forward, remote_read, local_write)


# ==================== Shared Host Runner ====================


# ==================== Host Existence Check ====================


def _check_existing_host(
    repo_path: Path,
    already_running_message: str,
    *,
    open_browser: bool,
    no_open: bool,
) -> bool:
    """
    Check if a live host is already running for the given repo.

    Returns True if a host is running (and was handled), False otherwise.
    """
    live_host = get_live_host_info(repo_path)
    if live_host is not None:
        console.print(already_running_message.format(url=live_host["host_url"]))
        if open_browser and not no_open:
            web_open(live_host["host_url"])
        return True
    return False


# ==================== Runtime and Storage Setup ====================


def _setup_runtime_and_storage(
    storage: KuzuBackend,
    repo_path: Path,
    host_url: str,
    mcp_url: str,
    *,
    watch: bool,
) -> tuple[AxonRuntime, Lock]:
    """Create and configure the AxonRuntime with storage and lock."""
    lock = Lock()
    runtime = AxonRuntime(
        storage=storage,
        repo_path=repo_path,
        watch=watch,
        lock=lock,
        host_url=host_url,
        mcp_url=mcp_url,
        owns_storage=True,
    )
    set_storage(storage)
    set_lock(lock)
    return runtime, lock


# ==================== Web Application Creation ====================


def _create_host_web_app(
    db_path: Path,
    repo_path: Path,
    runtime: AxonRuntime,
    host_url: str,
    mcp_url: str,
    *,
    watch: bool,
    dev: bool,
    expose_ui: bool,
) -> FastAPI:
    """Create and configure the web application for the host."""

    return create_app(
        db_path=db_path,
        repo_path=repo_path,
        watch=watch,
        dev=dev,
        runtime=runtime,
        mount_mcp=True,
        host_url=host_url,
        mcp_url=mcp_url,
        mount_frontend=expose_ui,
    )


# ==================== Browser Opening ====================


def _schedule_browser_open(host_url: str, *, open_browser: bool, no_open: bool) -> None:
    """Schedule browser to open after a short delay if requested."""
    if open_browser and not no_open:
        Timer(1.0, lambda: web_open(host_url)).start()


# ==================== Startup Messages ====================


def _print_startup_messages(
    host_url: str,
    mcp_url: str,
    *,
    watch: bool,
    dev: bool,
    announce_ui: bool,
    announce_mcp: bool,
) -> None:
    """Print startup status messages based on configuration."""
    if announce_ui:
        console.print(f"[bold green]Axon UI[/bold green] running at {host_url}")
    if announce_mcp:
        console.print(f"[dim]HTTP MCP endpoint:[/dim] {mcp_url}")
    if watch:
        console.print("[dim]File watching enabled[/dim]")
    if dev:
        console.print("[dim]Dev mode — proxying to Vite on :5173[/dim]")


# ==================== Async Host Runner ====================


async def _run_async_host(
    web_app: FastAPI,
    bind: str,
    port: int,
    repo_path: Path,
    storage: KuzuBackend,
    lock: Lock,
    *,
    watch: bool,
    managed: bool,
) -> None:
    """Run the async host server with optional watcher and managed shutdown."""
    config = Config(
        web_app,
        host=bind,
        port=port,
        log_level="warning",
    )
    server = Server(config)
    stop = Event()

    async def _serve() -> None:
        await server.serve()
        stop.set()

    async def _managed_shutdown() -> None:
        if not managed:
            return
        idle_started_at: float | None = None
        while not stop.is_set():
            live_leases = _count_live_host_leases(repo_path)
            if live_leases == 0:
                if idle_started_at is None:
                    idle_started_at = time()
                elif time() - idle_started_at >= 2.0:
                    server.should_exit = True
                    stop.set()
                    return
            else:
                idle_started_at = None
            await asyncio_sleep(0.5)

    tasks = [_serve()]
    if watch:
        deps = WatcherDeps(repo_path, storage, stop_event=stop, lock=lock)
        tasks.append(Watcher(deps).watch_repo())
    if managed:
        tasks.append(_managed_shutdown())
    await gather(*tasks)


# ==================== Host Cleanup ====================


def _cleanup_host(repo_path: Path, storage: KuzuBackend) -> None:
    """Clean up host resources on shutdown."""
    _clear_host_meta(repo_path)
    storage.close()


# ==================== Shared Host Runner ====================


# ==================== UI Helper Functions ====================


def check_db_exists(repo_path: Path) -> Path | None:
    """
    Check if the database exists and return the db_path, or None if not found.

    Args:
        repo_path: Path to the repository.

    Returns:
        Path to the database if it exists, None otherwise.
    """
    db_path = repo_path / ".axon" / "kuzu"
    return db_path if db_path.exists() else None


def run_ui_proxy_to_host(
    live_host: dict,
    port: int,
    *,
    dev: bool,
    no_open: bool,
) -> None:
    """
    Run a proxy UI that forwards requests to an existing host without UI enabled.

    This is used when a shared host is running but UI is not mounted.

    Args:
        live_host: Dictionary containing host info (host_url, etc.).
        port: Port to serve the proxy UI on.
        dev: Whether to run in dev mode (skip static file serving).
        no_open: Whether to skip auto-opening the browser.
    """

    proxy_app = create_ui_proxy_app(live_host["host_url"], dev=dev)
    host_url = f"http://{DEFAULT_HOST}:{port}"
    console.print(f"[bold green]Axon UI running at {host_url}")
    if not no_open:
        web_open(host_url)
    uvicorn_run(proxy_app, host=DEFAULT_HOST, port=port, log_level="warning")


def _create_standalone_ui_app(
    db_path: Path,
    repo_path: Path,
    *,
    watch: bool,
    dev: bool,
) -> FastAPI:
    """
    Create a standalone UI application (for direct mode).

    Args:
        db_path: Path to the KuzuDB database.
        repo_path: Path to the repository.
        watch: Whether to enable file watching.
        dev: Whether to run in dev mode.

    Returns:
        Configured FastAPI application for the UI.
    """

    return create_app(
        db_path=db_path,
        repo_path=repo_path,
        watch=watch,
        dev=dev,
    )


def _print_standalone_ui_startup(
    port: int,
    *,
    watch_files: bool,
    dev: bool,
    no_open: bool,
) -> str:
    """
    Print startup messages for standalone UI mode.

    Args:
        port: Port the UI is running on.
        watch_files: Whether file watching is enabled.
        dev: Whether dev mode is enabled.
        no_open: Whether browser auto-open is disabled.

    Returns:
        The URL that the UI is accessible at.
    """
    url = f"http://localhost:{port}"
    console.print(f"[bold green]Axon UI running at {url}")

    if watch_files:
        console.print("[dim]File watching enabled — graph updates on save")
    if dev:
        console.print("[dim]Dev mode — proxying to Vite on :5173[/dim]")

    if not no_open:
        Timer(1.5, lambda: web_open(url)).start()

    return url


async def _run_ui_with_watcher(
    web_app: FastAPI,
    port: int,
    repo_path: Path,
) -> None:
    """
    Run the UI server with file watching enabled.

    Args:
        web_app: The FastAPI application to run.
        port: Port to serve on.
        repo_path: Path to the repository for the watcher.
    """
    config = Config(web_app, host="127.0.0.1", port=port, log_level="warning")
    server = Server(config)
    stop = Event()

    async def _serve() -> None:
        await server.serve()
        stop.set()

    storage = web_app.state.storage
    await gather(
        _serve(),
        Watcher(
            WatcherDeps(repo_path, storage, stop_event=stop),
        ).watch_repo(),
    )


def run_standalone_ui(
    repo_path: Path,
    port: int,
    *,
    watch_files: bool,
    dev: bool,
    no_open: bool,
) -> None:
    """
    Run a standalone UI without connecting to a shared host.

    This is used when the --direct flag is set or no shared host is available.

    Args:
        repo_path: Path to the repository.
        port: Port to serve on.
        watch_files: Whether to enable file watching.
        dev: Whether to run in dev mode.
        no_open: Whether to skip auto-opening the browser.
    """

    db_path = check_db_exists(repo_path)
    if db_path is None:
        console.print(
            "[red]Error:[/red] No index found. Run [cyan]axon analyze .[/cyan] first to index this codebase.",
        )
        raise Exit(code=1)

    web_app = _create_standalone_ui_app(db_path, repo_path, watch=watch_files, dev=dev)
    _print_standalone_ui_startup(port, watch_files=watch_files, dev=dev, no_open=no_open)

    if watch_files:
        try:
            asyncio_run(_run_ui_with_watcher(web_app, port, repo_path))
        except KeyboardInterrupt:
            console.print("\n[b]UI stopped.")
    else:
        uvicorn_run(web_app, host="127.0.0.1", port=port, log_level="warning")


def check_live_host_for_ui(
    repo_path: Path,
    *,
    no_open: bool,
    dev: bool,
) -> bool:
    """
    Check if a live host exists and handle UI connection.

    This function checks for an existing shared host and:
    - If UI is enabled on the host, opens browser and returns True
    - If UI is not enabled, runs a proxy UI and returns True
    - If no host exists, returns False

    Args:
        repo_path: Path to the repository.
        no_open: Whether to skip auto-opening the browser.
        dev: Whether to run in dev mode (for proxy).

    Returns:
        True if a host (or proxy) was started, False if no host exists.
    """
    live_host = get_live_host_info(repo_path)
    if live_host is not None:
        if live_host.get("ui_enabled", True):
            # Host is running with UI - connect directly
            host_url = live_host["host_url"]
            console.print(f"[bold green]Axon UI available at {host_url}")
            if not no_open:
                web_open(host_url)
            return True
        else:
            # Host exists but no UI - run proxy
            run_ui_proxy_to_host(live_host, port=8420, dev=dev, no_open=no_open)
            return True
    return False


def run_shared_ui_host(
    repo_path: Path,
    port: int,
    *,
    watch_files: bool,
    dev: bool,
    no_open: bool,
) -> None:
    """
    Start a new shared host with UI enabled.

    Args:
        repo_path: Path to the repository.
        port: Port to serve on.
        watch_files: Whether to enable file watching.
        dev: Whether to run in dev mode.
        no_open: Whether to skip auto-opening the browser.
    """
    run_shared_host(
        port=port,
        bind=DEFAULT_HOST,
        no_open=no_open,
        watch=watch_files,
        dev=dev,
        managed=False,
        open_browser=True,
        announce_ui=True,
        announce_mcp=False,
        expose_ui=True,
        already_running_message="[bold green]Axon UI available at {url}",
        auto_index=False,
    )


def run_ui(
    port: int = 8420,
    *,
    no_open: bool = False,
    watch_files: bool = False,
    dev: bool = False,
    direct: bool = False,
) -> None:
    """
    Launch the Axon web UI.

    This function handles multiple scenarios:
    1. Direct mode: Run standalone UI without shared host
    2. Shared host with UI: Connect to existing host
    3. Shared host without UI: Run proxy UI
    4. No shared host: Start new shared host with UI

    Args:
        port: Port to serve the UI on (default: 8420).
        no_open: Don't auto-open browser when True.
        watch_files: Enable live file watching when True.
        dev: Proxy to Vite dev server for HMR when True.
        direct: Force standalone UI mode even if shared host exists.
    """
    repo_path = Path.cwd().resolve()

    if not direct:
        # Check if a shared host is already running
        if check_live_host_for_ui(repo_path, no_open=no_open, dev=dev):
            return

        # No shared host - start one with UI enabled
        run_shared_ui_host(repo_path, port, watch_files=watch_files, dev=dev, no_open=no_open)
        return

    # Direct mode - run standalone UI
    run_standalone_ui(repo_path, port, watch_files=watch_files, dev=dev, no_open=no_open)


def run_shared_host(
    port: int,
    bind: str,
    *,
    no_open: bool,
    watch: bool,
    dev: bool,
    managed: bool,
    open_browser: bool,
    announce_ui: bool,
    announce_mcp: bool,
    expose_ui: bool,
    already_running_message: str,
    auto_index: bool = True,
) -> None:
    """
    Run the shared Axon host with configurable UX messaging.

    This is kept for backward compatibility with the serve command.
    For UI-specific hosting, use run_shared_ui_host instead.

    Args:
        port: Port to serve on.
        bind: Host address to bind to.
        no_open: Don't auto-open browser when True.
        watch: Enable file watching when True.
        dev: Enable dev mode when True.
        managed: Enable managed shutdown when True.
        open_browser: Open browser on startup when True.
        announce_ui: Print UI URL when True.
        announce_mcp: Print MCP URL when True.
        expose_ui: Mount the frontend UI when True.
        already_running_message: Message template when host already exists.
        auto_index: Auto-index if no index exists when True.
    """
    repo_path = Path.cwd().resolve()

    # Check if host is already running
    if _check_existing_host(
        repo_path,
        already_running_message,
        open_browser=open_browser,
        no_open=no_open,
    ):
        return

    # Initialize storage
    storage, _, db_path = _initialize_writable_storage(repo_path, auto_index=auto_index)

    # Build URLs and setup runtime
    host_url, mcp_url = _build_host_urls(bind, port)
    runtime, lock = _setup_runtime_and_storage(storage, repo_path, host_url, mcp_url, watch=watch)

    # Create web application
    web_app = _create_host_web_app(
        db_path,
        repo_path,
        runtime,
        host_url,
        mcp_url,
        watch=watch,
        dev=dev,
        expose_ui=expose_ui,
    )

    # Schedule browser open
    _schedule_browser_open(host_url, open_browser=open_browser, no_open=no_open)

    # Print startup messages
    _print_startup_messages(
        host_url,
        mcp_url,
        watch=watch,
        dev=dev,
        announce_ui=announce_ui,
        announce_mcp=announce_mcp,
    )

    # Write host metadata
    _write_host_meta(repo_path, host_url, mcp_url, port, ui_enabled=expose_ui)

    # Run the async host
    try:
        asyncio_run(
            _run_async_host(
                web_app,
                bind,
                port,
                repo_path,
                storage,
                lock,
                watch=watch,
                managed=managed,
            ),
        )
    except KeyboardInterrupt:
        pass
    finally:
        _cleanup_host(repo_path, storage)
