"""UI helper for Axon CLI operations."""

from asyncio import Event, Lock, gather, get_running_loop, run
from asyncio import sleep as asyncio_sleep
from collections.abc import Callable
from logging import getLogger
from pathlib import Path
from threading import Timer
from webbrowser import open as web_open

from fastapi import FastAPI
from rich import print as rprint
from typer import Exit
from uvicorn import Config, Server
from uvicorn import run as uvicorn_run

from axon.cli.helpers.host_helpers.configs import (
    BrowserConfig,
    HostBehaviorConfig,
    HostStartupConfig,
    HostURLs,
    RuntimeContext,
    ServerConfig,
    StartupConfig,
    StartupDisplayConfig,
    WebAppConfig,
)
from axon.cli.helpers.host_helpers.host_state import HostStateHelper
from axon.cli.helpers.host_helpers.storage import StorageHelper
from axon.core.ingestion.watcher import Watcher, WatcherDeps
from axon.core.storage.kuzu_backend import KuzuBackend
from axon.mcp.server import set_lock, set_storage
from axon.runtime import AxonRuntime
from axon.web.app import create_app, create_ui_proxy_app

logger = getLogger(__name__)

# Constants
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8420


class UIRunner:
    """
    Handles UI launching and management.

    This class provides methods for running the Axon UI in various modes,
    including standalone mode, shared host mode, and proxy mode.
    """

    def __init__(
        self,
        storage_helper: StorageHelper | None = None,
        host_state_helper: HostStateHelper | None = None,
    ) -> None:
        """
        Initialize the UIRunner with optional helper instances.

        Args:
            storage_helper: Optional StorageHelper instance. If None, creates a new one.
            host_state_helper: Optional HostStateHelper instance. If None, creates a new one.
        """
        self._storage = storage_helper or StorageHelper()
        self._host_state = host_state_helper or HostStateHelper()

    # ==================== Web Application Setup ====================

    @staticmethod
    def setup_runtime_and_storage(
        storage: "KuzuBackend",
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

    @staticmethod
    def create_host_web_app(
        db_path: Path,
        repo_path: Path,
        runtime: AxonRuntime,
        urls: "HostURLs",
        config: "WebAppConfig",
    ) -> FastAPI:
        """Create and configure the web application for the host."""
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
    def schedule_browser_open(host_url: str, *, open_browser: bool, no_open: bool) -> None:
        """Schedule browser to open after a short delay if requested."""
        if open_browser and not no_open:
            Timer(1.0, lambda: web_open(host_url)).start()

    @staticmethod
    def print_startup_messages(
        urls: "HostURLs",
        config: "StartupConfig",
    ) -> dict[bool, Callable[[], None]]:
        """Print startup status messages based on configuration."""
        return {
            config.announce_ui: lambda: rprint(
                f"[bold green]Axon UI[/bold green] running at {urls.host_url}",
            ),
            config.announce_mcp: lambda: rprint(f"[dim]HTTP MCP endpoint:[/dim] {urls.mcp_url}"),
            config.watch: lambda: rprint("[dim]File watching enabled[/dim]"),
            config.dev: lambda: rprint("[dim]Dev mode — proxying to Vite on :5173[/dim]"),
        }

    # ==================== Async Host Runner ====================

    async def run_async_host(
        self,
        web_app: FastAPI,
        server_config: "ServerConfig",
        runtime_context: "RuntimeContext",
        behavior_config: "HostBehaviorConfig",
    ) -> None:
        """Run the async host server with optional watcher and managed shutdown."""
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
                live_leases = self.count_live_host_leases(runtime_context.repo_path)
                if live_leases == 0:
                    if idle_started_at is None:
                        idle_started_at = get_running_loop().time()
                    elif get_running_loop().time() - idle_started_at >= 2.0:
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

    # ==================== UI Helper Functions ====================

    @staticmethod
    def check_db_exists(repo_path: Path) -> Path | None:
        """Check if the database exists and return the db_path, or None if not found."""
        db_path = repo_path / ".axon" / "kuzu"
        return db_path if db_path.exists() else None

    @staticmethod
    def run_ui_proxy_to_host(
        live_host: dict,
        port: int,
        *,
        dev: bool,
        no_open: bool,
    ) -> None:
        """Run a proxy UI that forwards requests to an existing host without UI enabled."""
        proxy_app = create_ui_proxy_app(live_host["host_url"], dev=dev)
        host_url = f"http://{DEFAULT_HOST}:{port}"
        rprint(f"[bold green]Axon UI running at {host_url}")
        if not no_open:
            web_open(host_url)
        uvicorn_run(proxy_app, host=DEFAULT_HOST, port=port, log_level="warning")

    @staticmethod
    def create_standalone_ui_app(
        db_path: Path,
        repo_path: Path,
        *,
        watch: bool,
        dev: bool,
    ) -> FastAPI:
        """Create a standalone UI application (for direct mode)."""
        return create_app(
            db_path=db_path,
            repo_path=repo_path,
            watch=watch,
            dev=dev,
        )

    @staticmethod
    def print_standalone_ui_startup(
        port: int,
        *,
        watch_files: bool,
        dev: bool,
        no_open: bool,
    ) -> str:
        """Print startup messages for standalone UI mode."""
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
    async def run_ui_with_watcher(
        web_app: FastAPI,
        port: int,
        repo_path: Path,
    ) -> None:
        """Run the UI server with file watching enabled."""
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
        self,
        repo_path: Path,
        port: int,
        *,
        watch_files: bool,
        dev: bool,
        no_open: bool,
    ) -> None:
        """Run a standalone UI without connecting to a shared host."""
        db_path = self.check_db_exists(repo_path)
        if db_path is None:
            rprint(
                "[red]Error:[/red] No index found. Run [cyan]axon analyze .[/cyan] first to index this codebase.",
            )
            raise Exit(code=1)

        web_app = self.create_standalone_ui_app(
            db_path,
            repo_path,
            watch=watch_files,
            dev=dev,
        )
        self.print_standalone_ui_startup(
            port,
            watch_files=watch_files,
            dev=dev,
            no_open=no_open,
        )

        if watch_files:
            try:
                run(self.run_ui_with_watcher(web_app, port, repo_path))
            except KeyboardInterrupt:
                rprint("\n[b]UI stopped.")
        else:
            uvicorn_run(web_app, host="127.0.0.1", port=port, log_level="warning")

    def check_live_host_for_ui(
        self,
        repo_path: Path,
        *,
        no_open: bool,
        dev: bool,
    ) -> bool:
        """Check if a live host exists and handle UI connection."""

        live_host = self._host_state.get_live_host_info(repo_path)
        if live_host is not None:
            if live_host.get("ui_enabled", True):
                host_url = live_host["host_url"]
                rprint(f"[bold green]Axon UI available at {host_url}")
                if not no_open:
                    web_open(host_url)
                return True
            else:
                self.run_ui_proxy_to_host(live_host, port=8420, dev=dev, no_open=no_open)
                return True
        return False

    def run_shared_ui_host(
        self,
        server_config: "ServerConfig",
        browser_config: "BrowserConfig",
        behavior_config: "HostBehaviorConfig",
        startup_display_config: "StartupDisplayConfig",
        startup_config: "HostStartupConfig",
    ) -> None:
        """Start a new shared host with UI enabled."""
        self.run_shared_host(
            server_config=server_config,
            browser_config=browser_config,
            behavior_config=behavior_config,
            startup_display_config=startup_display_config,
            startup_config=startup_config,
        )

    def run_shared_host(
        self,
        server_config: "ServerConfig",
        browser_config: "BrowserConfig",
        behavior_config: "HostBehaviorConfig",
        startup_display_config: "StartupDisplayConfig",
        startup_config: "HostStartupConfig",
    ) -> None:
        """Run the shared Axon host with configurable UX messaging."""

        storage_helper = self._storage
        host_state = self._host_state

        repo_path = Path.cwd().resolve()
        port = server_config.port
        bind = server_config.bind
        no_open = browser_config.no_open
        open_browser = browser_config.open_browser
        watch = behavior_config.watch
        dev = behavior_config.dev
        announce_ui = startup_display_config.announce_ui
        announce_mcp = startup_display_config.announce_mcp
        expose_ui = startup_display_config.expose_ui
        already_running_message = startup_config.already_running_message
        auto_index = startup_config.auto_index

        if self.check_existing_host(
            repo_path,
            already_running_message,
            open_browser=open_browser,
            no_open=no_open,
        ):
            return

        storage, _, db_path = storage_helper.initialize_writable_storage(
            repo_path,
            auto_index=auto_index,
        )

        host_url, mcp_url = host_state.build_host_urls(bind, port)
        runtime, lock = self.setup_runtime_and_storage(
            storage,
            repo_path,
            host_url,
            mcp_url,
            watch=watch,
        )

        urls = HostURLs(host_url=host_url, mcp_url=mcp_url)
        web_app_config = WebAppConfig(watch=watch, dev=dev, expose_ui=expose_ui)
        web_app = self.create_host_web_app(
            db_path,
            repo_path,
            runtime,
            urls,
            web_app_config,
        )

        self.schedule_browser_open(host_url, open_browser=open_browser, no_open=no_open)

        startup_msg_config = StartupConfig(
            watch=watch,
            dev=dev,
            announce_ui=announce_ui,
            announce_mcp=announce_mcp,
        )
        messages = self.print_startup_messages(urls, startup_msg_config)
        for _bool, message in messages.items():
            if _bool:
                message()

        host_state.write_host_meta(repo_path, host_url, mcp_url, port, ui_enabled=expose_ui)

        runtime_context = RuntimeContext(repo_path=repo_path, storage=storage, lock=lock)
        try:
            run(
                self.run_async_host(
                    web_app,
                    server_config,
                    runtime_context,
                    behavior_config,
                ),
            )
        except KeyboardInterrupt:
            pass
        finally:
            host_state.cleanup_host(repo_path, storage)

    def check_existing_host(
        self,
        repo_path: Path,
        already_running_message: str,
        *,
        open_browser: bool,
        no_open: bool,
    ) -> bool:
        """Check if a live host is already running for the given repo."""

        live_host = self._host_state.get_live_host_info(repo_path)
        if live_host is not None:
            rprint(already_running_message.format(url=live_host["host_url"]))
            if open_browser and not no_open:
                web_open(live_host["host_url"])
            return True
        return False

    def count_live_host_leases(self, repo_path: Path) -> int:
        """Count active host leases (delegated to HostStateHelper)."""

        return self._host_state.count_live_host_leases(repo_path)
