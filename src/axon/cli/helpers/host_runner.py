"""
Helper functions for Axon CLI operations.

This module provides the core functionality for the Axon CLI host operations,
including host lifecycle management, UI launching, and MCP proxy handling.

Public API:
    - HostRunner: Main class for running Axon host services
    - HostConfig: Dataclass for host configuration
    - DEFAULT_MANAGED_PORT: Default port for managed host instances

Example:
    >>> from axon.cli.helpers.host import HostRunner, HostConfig
    >>> runner = HostRunner(port=8420, watch=True)
    >>> runner.run()
"""

from asyncio import Event, Lock, gather
from asyncio import run as asyncio_run
from asyncio import sleep as asyncio_sleep
from collections.abc import Callable
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
from rich import print as rprint
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


@dataclass(frozen=True)
class HostConfig:
    """
    Configuration for host processes.

    This dataclass holds the configuration options for starting an Axon host
    instance. It uses slots for memory efficiency and provides sensible defaults
    for all options.

    Attributes:
        port: The port number to bind the host to (default: 8420).
        bind: The IP address to bind to (default: "127.0.0.1").
        watch: Whether to enable file watching for automatic reindexing (default: True).
        timeout_seconds: Maximum time to wait for host startup (default: 10.0).
        managed: Whether to run in managed mode with automatic shutdown (default: False).

    Example:
        >>> config = HostConfig(port=9000, watch=False, managed=True)
        >>> print(config.port)
        9000
    """

    port: int = DEFAULT_PORT
    bind: str = DEFAULT_HOST
    watch: bool = True
    timeout_seconds: float = 10.0
    managed: bool = False


@dataclass(frozen=True)
class HostURLs:
    """
    URLs for the Axon host services.

    This dataclass groups the base URL and MCP endpoint URL together,
    as they are always used in combination.

    Attributes:
        host_url: The base URL of the host.
        mcp_url: The MCP endpoint URL.
    """

    host_url: str
    mcp_url: str


@dataclass(frozen=True)
class WebAppConfig:
    """
    Configuration for web application behavior.

    This dataclass groups configuration options that control how the
    web application behaves, including file watching, development mode,
    and UI exposure.

    Attributes:
        watch: Whether to enable file watching for live updates (default: True).
        dev: Whether to run in development mode with Vite HMR proxy (default: False).
        expose_ui: Whether to mount the frontend UI (default: True).
    """

    watch: bool = True
    dev: bool = False
    expose_ui: bool = True


@dataclass(frozen=True)
class StartupConfig:
    """
    Configuration for startup behavior and announcements.

    This dataclass groups options that control what information is
    displayed at startup and browser opening behavior.

    Attributes:
        watch: Whether file watching is enabled.
        dev: Whether dev mode is enabled.
        announce_ui: Whether to print the UI URL at startup.
        announce_mcp: Whether to print the MCP URL at startup.
        open_browser: Whether to open browser on startup.
    """

    watch: bool = True
    dev: bool = False
    announce_ui: bool = True
    announce_mcp: bool = False
    open_browser: bool = True


@dataclass(frozen=True)
class RunnerConfig:
    """
    Configuration for HostRunner behavior.

    This dataclass groups all configuration options for the HostRunner
    class, including server settings, UI behavior, and runtime options.

    Attributes:
        port: Port number to serve on.
        no_open: If True, skip auto-opening browser.
        watch: If True, enable file watching.
        dev: If True, enable dev mode with Vite HMR proxy.
        direct: If True, run standalone UI without shared host.
        managed: If True, run in managed mode with auto-shutdown.
    """

    port: int = DEFAULT_PORT
    no_open: bool = False
    watch: bool = True
    dev: bool = False
    direct: bool = False
    managed: bool = False


@dataclass(frozen=True)
class ServerConfig:
    """
    Configuration for the async server.

    Attributes:
        bind: Host address to bind to.
        port: Port to listen on.
    """

    bind: str = DEFAULT_HOST
    port: int = DEFAULT_PORT


@dataclass(frozen=True)
class RuntimeContext:
    """
    Runtime context for the host.

    This dataclass groups the runtime dependencies needed by the host,
    including repository path, storage backend, and synchronization lock.

    Attributes:
        repo_path: Path to the repository.
        storage: The storage backend.
        lock: Runtime lock for thread safety.
    """

    repo_path: Path
    storage: KuzuBackend
    lock: Lock


@dataclass(frozen=True)
class HostBehaviorConfig:
    """
    Configuration for host behavior options.

    This dataclass groups behavior-related configuration options
    that control how the host operates.

    Attributes:
        watch: Whether to enable file watching for live updates.
        dev: Whether to run in dev mode with Vite HMR proxy.
        managed: Whether to run in managed mode with auto-shutdown.
    """

    watch: bool = True
    dev: bool = False
    managed: bool = False


@dataclass(frozen=True)
class BrowserConfig:
    """
    Configuration for browser behavior.

    Attributes:
        no_open: If True, skip auto-opening browser.
        open_browser: If True, open browser on startup.
    """

    no_open: bool = False
    open_browser: bool = True


@dataclass(frozen=True)
class StartupDisplayConfig:
    """
    Configuration for startup display and announcements.

    Attributes:
        announce_ui: Whether to print the UI URL at startup.
        announce_mcp: Whether to print the MCP URL at startup.
        expose_ui: Whether to mount the frontend UI.
    """

    announce_ui: bool = True
    announce_mcp: bool = False
    expose_ui: bool = True


@dataclass(frozen=True)
class HostStartupConfig:
    """
    Configuration for host startup behavior.

    Attributes:
        already_running_message: Message template when host already exists.
        auto_index: Whether to auto-index if no index exists.
    """

    already_running_message: str = "[bold green]Axon UI available at {url}"
    auto_index: bool = True


# ==================== Private Helper Classes ====================


class _HostHelpers:
    """
    Private helper class containing internal utility functions.

    This class encapsulates all private helper functions used throughout the
    HostRunner implementation. These methods handle low-level operations such
    as storage initialization, host metadata management, lease tracking,
    and various utility functions.

    The class is organized into logically grouped sections:
    - Storage and Initialization: Database and storage setup
    - Update Checking: Version update notification logic
    - Registry Management: Global registry operations
    - Meta Building: Metadata construction
    - Host Metadata: Reading/writing host state files
    - Lease Management: Host lease tracking and cleanup
    - Process Management: Background process handling
    - Web Application: FastAPI app creation
    - UI Helpers: UI-specific utilities
    """

    # ==================== Storage and Initialization ====================

    @staticmethod
    def _load_storage(repo_path: Path | None = None) -> KuzuBackend:
        """
        Load the KuzuDB backend for the given or current repo (read-only).

        Args:
            repo_path: Optional path to the repository. If None, uses current
                      working directory.

        Returns:
            KuzuBackend: Initialized read-only storage backend.

        Raises:
            Exit: If no index exists at the specified path.

        Note:
            This function opens the database in read-only mode for operations
            that only require querying existing data.
        """
        target = (repo_path or Path.cwd()).resolve()
        db_path = target / ".axon" / "kuzu"
        if not db_path.exists():
            rprint(
                f"[red]Error:[/red] No index found at {target}. Run 'axon analyze' first.",
            )
            raise Exit(code=1)

        storage = KuzuBackend()
        storage.initialize(db_path, read_only=True)
        return storage

    def _initialize_writable_storage(
        self,
        repo_path: Path,
        *,
        auto_index: bool = True,
    ) -> tuple[KuzuBackend, Path, Path]:
        """
        Open the repo database in read-write mode.

        This function initializes the KuzuDB storage in read-write mode and
        optionally runs the initial indexing pipeline if no index exists.

        Args:
            repo_path: Path to the repository to initialize storage for.
            auto_index: If True and no index exists, automatically run the
                       indexing pipeline. If False and no index exists,
                       raises Exit (default: True).

        Returns:
            Tuple of (storage, axon_dir, db_path) for the initialized storage.

        Raises:
            Exit: If auto_index is False and no index exists.

        Note:
            Callers like ``ui`` should set auto_index=False when they want
            to prompt the user to run ``axon analyze .`` themselves.
        """
        axon_dir = repo_path / ".axon"
        db_path = axon_dir / "kuzu"

        if not auto_index and not (axon_dir / "meta.json").exists():
            rprint(
                "[red]Error:[/red] No index found. Run [cyan]axon analyze .[/cyan] first to index this codebase.",
            )
            raise Exit(code=1)

        axon_dir.mkdir(parents=True, exist_ok=True)

        storage = KuzuBackend()
        storage.initialize(db_path)

        if not (axon_dir / "meta.json").exists():
            rprint("[bold]Running initial index...[/bold]")
            pipelines = Pipelines(repo_path, storage, full=True, embeddings=True)
            pipelines.run_pipelines()
            result = pipelines.result
            meta = self._build_meta(result, repo_path)
            meta_path = axon_dir / "meta.json"
            meta_path.write_text(dumps(meta, indent=2) + "\n", encoding="utf-8")
            try:
                self._register_in_global_registry(meta, repo_path)
            except (RuntimeError, OSError, SystemError, PermissionError):
                logger.debug("Failed to register repo in global registry", exc_info=True)

        return storage, axon_dir, db_path

    # ==================== Update Checking ====================

    @staticmethod
    def _update_cache_path() -> Path:
        """
        Get the path to the update check cache file.

        Returns:
            Path to the update-check.json file in the user's config directory.
        """
        return Path.home() / ".axon" / "update-check.json"

    @staticmethod
    def _parse_version_parts(version: str) -> tuple[int, ...]:
        """
        Parse a version string into numeric parts.

        Extracts all numeric sequences from a version string and converts
        them to integers. Non-digit characters are ignored.

        Args:
            version: Version string (e.g., "1.2.3", "2.0.0-beta.1").

        Returns:
            Tuple of integers representing the version parts.

        Example:
            >>> _parse_version_parts("1.2.3")
            (1, 2, 3)
            >>> _parse_version_parts("2.0.0-beta.1")
            (2, 0, 0, 1)
        """
        parts: list[int] = []
        for raw_part in version.split("."):
            digits = "".join(ch for ch in raw_part if ch.isdigit())
            parts.append(int(digits or 0))
        return tuple(parts)

    def _is_newer_version(self, candidate: str, current: str) -> bool:
        """
        Check if candidate version is newer than current version.

        Args:
            candidate: The version to check (should be newer).
            current: The current installed version.

        Returns:
            True if candidate is newer than current, False otherwise.
        """
        return self._parse_version_parts(candidate) > self._parse_version_parts(
            current,
        )

    def _read_update_cache(self) -> dict | None:
        """
        Read the update check cache from disk.

        Returns:
            Dictionary containing cached update info, or None if cache
            doesn't exist or is invalid.
        """
        cache_path = self._update_cache_path()
        if not cache_path.exists():
            return None
        try:
            return loads(cache_path.read_text(encoding="utf-8"))
        except (JSONDecodeError, OSError):
            return None

    def _write_update_cache(self, payload: dict) -> None:
        """
        Write the update check cache to disk.

        Args:
            payload: Dictionary containing update check data to cache.
        """
        cache_path = self._update_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(dumps(payload, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _fetch_latest_version() -> str | None:
        """
        Fetch the latest Axon version from PyPI.

        Makes an HTTP request to PyPI's JSON API to retrieve the latest
        version number. Has a short timeout to avoid blocking.

        Returns:
            Latest version string, or None if the request fails.
        """
        try:
            with urlopen(UPDATE_CHECK_URL, timeout=1.5) as response:  # noqa: S310
                payload = loads(response.read().decode("utf-8"))
                return str(payload["info"]["version"])
        except (KeyError, OSError, ValueError, URLError):
            return None

    def _get_latest_version(self) -> str | None:
        """
        Get the latest Axon version, using cache if available.

        Checks the local cache first to avoid redundant network requests.
        If cache is missing or expired, fetches fresh data from PyPI.

        Returns:
            Latest version string, or None if unavailable.
        """
        now = int(time())
        cache = self._read_update_cache()
        if cache is not None:
            checked_at = int(cache.get("checked_at", 0))
            latest = cache.get("latest_version")
            if latest and now - checked_at < UPDATE_CHECK_INTERVAL_SECONDS:
                return str(latest)

        latest = self._fetch_latest_version()
        if latest is not None:
            self._write_update_cache({"checked_at": now, "latest_version": latest})
        return latest

    # ==================== Registry Management ====================

    @staticmethod
    def _register_in_global_registry(meta: dict, repo_path: Path) -> None:
        """
        Write meta.json into ``~/.axon/repos/{slug}/`` for multi-repo discovery.

        Registers the repository in the global registry for discovery across
        multiple repositories. Handles slug collisions by appending a hash.

        Args:
            meta: Metadata dictionary for the repository.
            repo_path: Path to the repository to register.

        Note:
            Slug is ``{repo_name}`` if that slot is unclaimed or already belongs to
            this repo. Falls back to ``{repo_name}-{sha256(path)[:8]}`` on collision.
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

    # ==================== Meta Building ====================

    @staticmethod
    def _build_meta(result: PipelineResult, repo_path: Path) -> dict:
        """
        Build metadata dictionary from pipeline results.

        Constructs a standardized metadata dictionary containing version info,
        repository details, and statistics from the indexing pipeline.

        Args:
            result: The pipeline result containing indexing statistics.
            repo_path: Path to the indexed repository.

        Returns:
            Dictionary containing repository metadata.
        """
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

    # ==================== Host Metadata ====================

    @staticmethod
    def _host_meta_path(repo_path: Path) -> Path:
        """
        Get the path to the host metadata file.

        Args:
            repo_path: Path to the repository.

        Returns:
            Path to the host.json file.
        """
        return repo_path / ".axon" / "host.json"

    @staticmethod
    def _host_lease_dir(repo_path: Path) -> Path:
        """
        Get the path to the host lease directory.

        Args:
            repo_path: Path to the repository.

        Returns:
            Path to the host-leases directory.
        """
        return repo_path / ".axon" / "host-leases"

    @staticmethod
    def _display_host(host: str) -> str:
        """
        Normalize host address for display.

        Converts binding addresses like "0.0.0.0" or "::" to "127.0.0.1"
        for display purposes, while keeping the original for binding.

        Args:
            host: The host address to normalize.

        Returns:
            Display-friendly host address.
        """
        return "127.0.0.1" if host in {"0.0.0.0", "::"} else host  # noqa: S104

    def _build_host_urls(self, host: str, port: int) -> tuple[str, str]:
        """
        Build the base URL and MCP URL for the host.

        Args:
            host: The host address.
            port: The port number.

        Returns:
            Tuple of (base_url, mcp_url).
        """
        base = f"http://{self._display_host(host)}:{port}"
        return base, f"{base}/mcp"

    def _read_host_meta(self, repo_path: Path) -> dict | None:
        """
        Read host metadata from disk.

        Args:
            repo_path: Path to the repository.

        Returns:
            Dictionary containing host metadata, or None if not found.
        """
        meta_path = self._host_meta_path(repo_path)
        if not meta_path.exists():
            return None
        try:
            return loads(meta_path.read_text(encoding="utf-8"))
        except (JSONDecodeError, OSError):
            return None

    def _write_host_meta(
        self,
        repo_path: Path,
        host_url: str,
        mcp_url: str,
        port: int,
        *,
        ui_enabled: bool,
    ) -> None:
        """
        Write host metadata to disk.

        Stores current host information including PID, URLs, and configuration
        for discovery by other processes.

        Args:
            repo_path: Path to the repository.
            host_url: Base URL of the host.
            mcp_url: MCP endpoint URL.
            port: Port the host is running on.
            ui_enabled: Whether the UI is enabled.
        """
        meta_path = self._host_meta_path(repo_path)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": getpid(),
            "repo_path": str(repo_path),
            "host_url": host_url,
            "mcp_url": mcp_url,
            "port": port,
            "ui_enabled": ui_enabled,
            "leases_dir": str(self._host_lease_dir(repo_path)),
        }
        meta_path.write_text(dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _clear_host_meta(self, repo_path: Path) -> None:
        """
        Remove host metadata file on shutdown.

        Args:
            repo_path: Path to the repository.
        """
        meta_path = self._host_meta_path(repo_path)
        if meta_path.exists():
            meta_path.unlink(missing_ok=True)

    # ==================== Lease Management ====================

    def create_host_lease(self, repo_path: Path, lease_type: str) -> Path:
        """
        Create a new host lease file.

        Creates a lease file that indicates an active connection to the
        shared host. Used for managed shutdown when all clients disconnect.

        Args:
            repo_path: Path to the repository.
            lease_type: Type of lease (e.g., "mcp", "ui").

        Returns:
            Path to the created lease file.
        """
        lease_dir = self._host_lease_dir(repo_path)
        lease_dir.mkdir(parents=True, exist_ok=True)
        lease_path = lease_dir / f"{getpid()}-{uuid4().hex}.json"
        payload = {
            "pid": getpid(),
            "type": lease_type,
            "created_at": time(),
        }
        lease_path.write_text(dumps(payload), encoding="utf-8")
        return lease_path

    @staticmethod
    def remove_host_lease(lease_path: Path | None) -> None:
        """
        Remove a host lease file.

        Args:
            lease_path: Path to the lease file to remove.
        """
        if lease_path is not None:
            lease_path.unlink(missing_ok=True)

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        """
        Check if a process ID is still running.

        Uses a zero signal to check process existence without actually
        terminating the process.

        Args:
            pid: Process ID to check.

        Returns:
            True if the process is alive, False otherwise.
        """
        try:
            kill(pid, 0)
        except OSError:
            return False
        return True

    def _count_live_host_leases(self, repo_path: Path) -> int:
        """
        Count the number of active host leases.

        Iterates through all lease files and counts those with still-running
        processes. Cleans up stale lease files.

        Args:
            repo_path: Path to the repository.

        Returns:
            Number of active leases.
        """
        lease_dir = self._host_lease_dir(repo_path)
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
            if self._pid_is_alive(pid):
                live += 1
            else:
                lease_path.unlink(missing_ok=True)
        return live

    @staticmethod
    def _is_host_alive(meta: dict, repo_path: Path) -> bool:
        """
        Check if a host is running by querying its health endpoint.

        Makes an HTTP request to the host's /api/host endpoint to verify
        it's running and serving the correct repository.

        Args:
            meta: Host metadata dictionary.
            repo_path: Expected repository path.

        Returns:
            True if host is alive and serving the correct repo, False otherwise.
        """
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

    def _get_live_host_info(self, repo_path: Path) -> dict | None:
        """
        Get information about a running host, if any.

        Checks for existing host metadata and verifies the host is alive.

        Args:
            repo_path: Path to the repository.

        Returns:
            Host metadata dictionary if host is running, None otherwise.
        """
        meta = self._read_host_meta(repo_path)
        if meta is None:
            return None
        if self._is_host_alive(meta, repo_path):
            return meta
        return None

    # ==================== Process Management ====================

    @staticmethod
    def _start_host_background(
        repo_path: Path,
        *,
        port: int = DEFAULT_PORT,
        bind: str = DEFAULT_HOST,
        watch: bool = True,
        managed: bool = False,
    ) -> None:
        """
        Start a detached shared host process in the background.

        Launches a new Axon host process as a background daemon using subprocess.

        Args:
            repo_path: Path to the repository.
            port: Port to bind to (default: DEFAULT_PORT).
            bind: Host address to bind to (default: DEFAULT_HOST).
            watch: Whether to enable file watching (default: True).
            managed: Whether to run in managed mode (default: False).
        """
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

    # ==================== Version Callback ====================

    @staticmethod
    def _version_callback(*, value: bool) -> None:
        """
        Handle version flag callback.

        Args:
            value: Whether the version flag was provided.
        """
        if value:
            rprint(f"Axon v{__version__}")
            raise Exit()

    # ==================== Web Application ====================

    @staticmethod
    def _setup_runtime_and_storage(
        storage: KuzuBackend,
        repo_path: Path,
        host_url: str,
        mcp_url: str,
        *,
        watch: bool,
    ) -> tuple[AxonRuntime, Lock]:
        """
        Create and configure the AxonRuntime with storage and lock.

        Initializes the runtime context for the host, including setting
        up the global storage and lock instances.

        Args:
            storage: The KuzuBackend storage instance.
            repo_path: Path to the repository.
            host_url: Base URL of the host.
            mcp_url: MCP endpoint URL.
            watch: Whether to enable file watching.

        Returns:
            Tuple of (AxonRuntime, Lock).
        """
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

    @staticmethod
    def _create_host_web_app(
        db_path: Path,
        repo_path: Path,
        runtime: AxonRuntime,
        urls: HostURLs,
        config: WebAppConfig,
    ) -> FastAPI:
        """
        Create and configure the web application for the host.

        Args:
            db_path: Path to the KuzuDB database.
            repo_path: Path to the repository.
            runtime: The AxonRuntime instance.
            urls: The host URLs (base URL and MCP URL).
            config: Web application configuration (watch, dev, expose_ui).

        Returns:
            Configured FastAPI application.
        """
        return create_app(
            db_path=db_path,
            repo_path=repo_path,
            watch=config.watch,
            dev=config.dev,
            runtime=runtime,
            mount_mcp=True,
            host_url=urls.host_url,
            mcp_url=urls.mcp_url,
            mount_frontend=config.expose_ui,
        )

    @staticmethod
    def _schedule_browser_open(host_url: str, *, open_browser: bool, no_open: bool) -> None:
        """
        Schedule browser to open after a short delay if requested.

        Uses a timer to delay browser opening, giving the server time to start.

        Args:
            host_url: URL to open in browser.
            open_browser: Whether to open browser (before no_open check).
            no_open: Whether to skip opening (final decision).
        """
        if open_browser and not no_open:
            Timer(1.0, lambda: web_open(host_url)).start()

    @staticmethod
    def _print_startup_messages(
        urls: HostURLs,
        config: StartupConfig,
    ) -> dict[bool, Callable[[], None]]:
        """
        Print startup status messages based on configuration.

        Args:
            urls: The host URLs (base URL and MCP URL).
            config: Startup configuration (watch, dev, announce_ui, announce_mcp, open_browser).
        """
        return {
            config.announce_ui: lambda: rprint(
                f"[bold green]Axon UI[/bold green] running at {urls.host_url}",
            ),
            config.announce_mcp: lambda: rprint(f"[dim]HTTP MCP endpoint:[/dim] {urls.mcp_url}"),
            config.watch: lambda: rprint("[dim]File watching enabled[/dim]"),
            config.dev: lambda: rprint("[dim]Dev mode — proxying to Vite on :5173[/dim]"),
        }

    # ==================== Async Host Runner ====================

    async def _run_async_host(
        self,
        web_app: FastAPI,
        server_config: ServerConfig,
        runtime_context: RuntimeContext,
        behavior_config: HostBehaviorConfig,
    ) -> None:
        """
        Run the async host server with optional watcher and managed shutdown.

        This is the main async entry point that coordinates the web server,
        optional file watcher, and managed shutdown logic.

        Args:
            web_app: The FastAPI application.
            server_config: Server configuration (bind address, port).
            runtime_context: Runtime context (repo_path, storage, lock).
            behavior_config: Behavior configuration (watch, managed).
        """
        config = Config(
            web_app,
            host=server_config.bind,
            port=server_config.port,
            log_level="warning",
        )
        server = Server(config)
        stop = Event()

        async def _serve() -> None:
            await server.serve()
            stop.set()

        async def _managed_shutdown() -> None:
            if not behavior_config.managed:
                return
            idle_started_at: float | None = None
            while not stop.is_set():
                live_leases = self._count_live_host_leases(runtime_context.repo_path)
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
        if behavior_config.watch:
            deps = WatcherDeps(
                runtime_context.repo_path,
                runtime_context.storage,
                stop_event=stop,
                lock=runtime_context.lock,
            )
            tasks.append(Watcher(deps).watch_repo())
        if behavior_config.managed:
            tasks.append(_managed_shutdown())
        await gather(*tasks)

    # ==================== Host Cleanup ====================

    def _cleanup_host(self, repo_path: Path, storage: KuzuBackend) -> None:
        """
        Clean up host resources on shutdown.

        Args:
            repo_path: Path to the repository.
            storage: The storage backend to close.
        """
        self._clear_host_meta(repo_path)
        storage.close()

    # ==================== UI Helper Functions ====================

    @staticmethod
    def _check_db_exists(repo_path: Path) -> Path | None:
        """
        Check if the database exists and return the db_path, or None if not found.

        Args:
            repo_path: Path to the repository.

        Returns:
            Path to the database if it exists, None otherwise.
        """
        db_path = repo_path / ".axon" / "kuzu"
        return db_path if db_path.exists() else None

    @staticmethod
    def _run_ui_proxy_to_host(
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
        rprint(f"[bold green]Axon UI running at {host_url}")
        if not no_open:
            web_open(host_url)
        uvicorn_run(proxy_app, host=DEFAULT_HOST, port=port, log_level="warning")

    @staticmethod
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

    @staticmethod
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
        rprint(f"[bold green]Axon UI running at {url}")

        if watch_files:
            rprint("[dim]File watching enabled — graph updates on save")
        if dev:
            rprint("[dim]Dev mode — proxying to Vite on :5173[/dim]")

        if not no_open:
            Timer(1.5, lambda: web_open(url)).start()

        return url

    @staticmethod
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

    def _run_standalone_ui(
        self,
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
        db_path = self._check_db_exists(repo_path)
        if db_path is None:
            rprint(
                "[red]Error:[/red] No index found. Run [cyan]axon analyze .[/cyan] first to index this codebase.",
            )
            raise Exit(code=1)

        web_app = self._create_standalone_ui_app(
            db_path,
            repo_path,
            watch=watch_files,
            dev=dev,
        )
        self._print_standalone_ui_startup(
            port,
            watch_files=watch_files,
            dev=dev,
            no_open=no_open,
        )

        if watch_files:
            try:
                asyncio_run(self._run_ui_with_watcher(web_app, port, repo_path))
            except KeyboardInterrupt:
                rprint("\n[b]UI stopped.")
        else:
            uvicorn_run(web_app, host="127.0.0.1", port=port, log_level="warning")

    def _check_live_host_for_ui(
        self,
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
        live_host = self._get_live_host_info(repo_path)
        if live_host is not None:
            if live_host.get("ui_enabled", True):
                # Host is running with UI - connect directly
                host_url = live_host["host_url"]
                rprint(f"[bold green]Axon UI available at {host_url}")
                if not no_open:
                    web_open(host_url)
                return True
            else:
                # Host exists but no UI - run proxy
                self._run_ui_proxy_to_host(live_host, port=8420, dev=dev, no_open=no_open)
                return True
        return False

    def _run_shared_ui_host(
        self,
        server_config: ServerConfig,
        browser_config: BrowserConfig,
        behavior_config: HostBehaviorConfig,
        startup_display_config: StartupDisplayConfig,
        startup_config: HostStartupConfig,
    ) -> None:
        """
        Start a new shared host with UI enabled.

        Args:
            server_config: Server configuration (bind address, port).
            browser_config: Browser configuration (no_open, open_browser).
            behavior_config: Behavior configuration (watch, dev, managed).
            startup_display_config: Startup display configuration.
            startup_config: Host startup configuration.
        """
        self._run_shared_host(
            server_config=server_config,
            browser_config=browser_config,
            behavior_config=behavior_config,
            startup_display_config=startup_display_config,
            startup_config=startup_config,
        )

    def _run_shared_host(
        self,
        server_config: ServerConfig,
        browser_config: BrowserConfig,
        behavior_config: HostBehaviorConfig,
        startup_display_config: StartupDisplayConfig,
        startup_config: HostStartupConfig,
    ) -> None:
        """
        Run the shared Axon host with configurable UX messaging.

        This is the core method for starting a shared host instance. It handles
        storage initialization, runtime setup, web app creation, and coordinates
        the async server lifecycle.

        Args:
            server_config: Server configuration (bind address, port).
            browser_config: Browser configuration (no_open, open_browser).
            behavior_config: Behavior configuration (watch, dev, managed).
            startup_display_config: Startup display configuration (announce_ui, announce_mcp, expose_ui).
            startup_config: Host startup configuration (already_running_message, auto_index).
        """
        repo_path = Path.cwd().resolve()
        port = server_config.port
        bind = server_config.bind
        no_open = browser_config.no_open
        open_browser = browser_config.open_browser
        watch = behavior_config.watch
        dev = behavior_config.dev
        # managed is passed to _run_async_host via behavior_config
        announce_ui = startup_display_config.announce_ui
        announce_mcp = startup_display_config.announce_mcp
        expose_ui = startup_display_config.expose_ui
        already_running_message = startup_config.already_running_message
        auto_index = startup_config.auto_index

        # Check if host is already running
        if self._check_existing_host(
            repo_path,
            already_running_message,
            open_browser=open_browser,
            no_open=no_open,
        ):
            return

        # Initialize storage
        storage, _, db_path = self._initialize_writable_storage(
            repo_path,
            auto_index=auto_index,
        )

        # Build URLs and setup runtime
        host_url, mcp_url = self._build_host_urls(bind, port)
        runtime, lock = self._setup_runtime_and_storage(
            storage,
            repo_path,
            host_url,
            mcp_url,
            watch=watch,
        )

        # Create web application
        urls = HostURLs(host_url=host_url, mcp_url=mcp_url)
        web_app_config = WebAppConfig(watch=watch, dev=dev, expose_ui=expose_ui)
        web_app = self._create_host_web_app(
            db_path,
            repo_path,
            runtime,
            urls,
            web_app_config,
        )

        # Schedule browser open
        self._schedule_browser_open(host_url, open_browser=open_browser, no_open=no_open)

        # Print startup messages
        startup_msg_config = StartupConfig(
            watch=watch,
            dev=dev,
            announce_ui=announce_ui,
            announce_mcp=announce_mcp,
        )
        messages = self._print_startup_messages(urls, startup_msg_config)
        for _bool, message in messages.items():
            if _bool:
                message()

        # Write host metadata
        self._write_host_meta(repo_path, host_url, mcp_url, port, ui_enabled=expose_ui)

        # Run the async host
        runtime_context = RuntimeContext(repo_path=repo_path, storage=storage, lock=lock)
        try:
            asyncio_run(
                self._run_async_host(
                    web_app,
                    server_config,
                    runtime_context,
                    behavior_config,
                ),
            )
        except KeyboardInterrupt:
            pass
        finally:
            self._cleanup_host(repo_path, storage)

    def _check_existing_host(
        self,
        repo_path: Path,
        already_running_message: str,
        *,
        open_browser: bool,
        no_open: bool,
    ) -> bool:
        """
        Check if a live host is already running for the given repo.

        Returns True if a host is running (and was handled), False otherwise.

        Args:
            repo_path: Path to the repository.
            already_running_message: Message template to display.
            open_browser: Whether to open browser if host exists.
            no_open: Override to skip browser opening.

        Returns:
            True if a host is running (and was handled), False otherwise.
        """
        live_host = self._get_live_host_info(repo_path)
        if live_host is not None:
            rprint(already_running_message.format(url=live_host["host_url"]))
            if open_browser and not no_open:
                web_open(live_host["host_url"])
            return True
        return False


# ==================== Public API - HostRunner Class ====================


class HostRunner(_HostHelpers):
    """
    Main class for running Axon host services.

    This class provides the primary API for launching and managing Axon host
    instances, including the web UI, MCP server, and background host processes.

    The HostRunner encapsulates all functionality needed to:
    - Launch the Axon web UI in various modes
    - Manage shared host processes
    - Handle MCP proxy connections
    - Check for and notify about updates

    Attributes:
        config: The runner configuration.

    Example:
        >>> runner = HostRunner(config=RunnerConfig(port=8420, watch=True, dev=False))
        >>> runner.run_ui()

        Or using the class method for MCP serving:
        >>> runner = HostRunner(config=RunnerConfig(port=8421, managed=True))
        >>> runner.ensure_host_running()
    """

    def __init__(
        self,
        config: RunnerConfig,
    ) -> None:
        """
        Initialize the HostRunner with configuration options.

        Args:
            config: Runner configuration containing port, no_open, watch, dev, direct, managed.
        """
        self.config = config

    # ==================== Update Notifications ====================

    def maybe_notify_update(self, invoked_subcommand: str | None) -> None:
        """
        Check for and notify about available updates.

        Checks if a newer version of Axon is available on PyPI and prints
        a notification message if one exists. Skips checking for certain
        commands that don't need it.

        Args:
            invoked_subcommand: The subcommand that was invoked. If None
                               or in SKIP_COMMANDS, no check is performed.

        Note:
            This function runs asynchronously and caches results to avoid
            excessive network requests. The cache expires after 24 hours.
        """
        if invoked_subcommand in UPDATE_CHECK_SKIP_COMMANDS:
            return
        if (latest := self._get_latest_version()) and self._is_newer_version(
            latest,
            __version__,
        ):
            rprint(
                f"[yellow]Update available:[/yellow] Axon {latest} "
                f"(current {__version__}). Run `pip install -U axoniq`.",
            )

    # ==================== Host Process Management ====================

    def ensure_host_running(
        self,
        repo_path: Path,
        config: HostConfig | None = None,
    ) -> dict:
        """
        Return live host metadata, starting the shared host if necessary.

        This method checks if a shared host is already running for the given
        repository. If not, it starts a new background host process and waits
        for it to become available.

        Args:
            repo_path: Path to the repository.
            config: Optional HostConfig for host settings. If None, uses defaults.

        Returns:
            Dictionary containing live host metadata including host_url, mcp_url,
            port, and other configuration details.

        Raises:
            RuntimeError: If the host fails to start within the timeout period.

        Example:
            >>> runner = HostRunner(port=8421, watch=True)
            >>> meta = runner.ensure_host_running(Path.cwd())
            >>> print(meta["host_url"])
            http://127.0.0.1:8421
        """
        if config is None:
            config = HostConfig()

        live_host = self._get_live_host_info(repo_path)
        if live_host is not None:
            return live_host

        self._start_host_background(
            repo_path,
            port=config.port,
            bind=config.bind,
            watch=config.watch,
            managed=config.managed,
        )
        deadline = time() + config.timeout_seconds
        while time() < deadline:
            live_host = self._get_live_host_info(repo_path)
            if live_host is not None:
                return live_host
            sleep(0.2)
        details = "Timed out waiting for Axon host to start."
        raise RuntimeError(details)

    # ==================== Lease Management ====================

    def create_host_lease(self, repo_path: Path, lease_type: str) -> Path:
        """
        Create a new host lease file.

        Creates a lease file that indicates an active connection to the
        shared host. Used for managed shutdown tracking.

        Args:
            repo_path: Path to the repository.
            lease_type: Type of lease (e.g., "mcp", "ui").

        Returns:
            Path to the created lease file.

        Note:
            Always call remove_host_lease when done to clean up.
        """
        return self.create_host_lease(repo_path, lease_type)

    def remove_host_lease(self, lease_path: Path | None) -> None:
        """
        Remove a host lease file.

        Args:
            lease_path: Path to the lease file to remove.
        """
        self.remove_host_lease(lease_path)

    # ==================== UI Launching ====================

    def run_ui(self) -> None:
        """
        Launch the Axon web UI.

        This method handles multiple scenarios for launching the UI:
        1. Direct mode: Run standalone UI without shared host
        2. Shared host with UI: Connect to existing host
        3. Shared host without UI: Run proxy UI
        4. No shared host: Start new shared host with UI

        The behavior depends on the configuration passed to the constructor:
        - If direct=True: Always runs standalone UI
        - If direct=False (default): Checks for existing shared host first

        Example:
            >>> runner = HostRunner(config=RunnerConfig(port=8420, watch=True, dev=False))
            >>> runner.run_ui()
        """
        repo_path = Path.cwd().resolve()

        if not self.config.direct:
            # Check if a shared host is already running
            if self._check_live_host_for_ui(
                repo_path,
                no_open=self.config.no_open,
                dev=self.config.dev,
            ):
                return

            # No shared host - start one with UI enabled
            server_config = ServerConfig(port=self.config.port)
            browser_config = BrowserConfig(no_open=self.config.no_open)
            behavior_config = HostBehaviorConfig(watch=self.config.watch, dev=self.config.dev)
            startup_display_config = StartupDisplayConfig()
            startup_config = HostStartupConfig()
            self._run_shared_ui_host(
                server_config=server_config,
                browser_config=browser_config,
                behavior_config=behavior_config,
                startup_display_config=startup_display_config,
                startup_config=startup_config,
            )
            return

        # Direct mode - run standalone UI
        self._run_standalone_ui(
            repo_path,
            self.config.port,
            watch_files=self.config.watch,
            dev=self.config.dev,
            no_open=self.config.no_open,
        )

    # ==================== MCP Proxy ====================

    @staticmethod
    async def proxy_stdio_to_http_mcp(mcp_url: str) -> None:
        """
        Bridge a local stdio MCP session to the shared HTTP MCP host.

        This async function creates a bidirectional bridge between a local
        stdio-based MCP client and the HTTP-based MCP server. It forwards
        messages in both directions concurrently.

        Args:
            mcp_url: The URL of the HTTP MCP endpoint to connect to.

        Example:
            >>> async def serve_mcp():
            ...     await HostRunner.proxy_stdio_to_http_mcp("http://127.0.0.1:8421/mcp")
        """
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
