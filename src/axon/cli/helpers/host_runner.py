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

from logging import getLogger
from pathlib import Path
from time import sleep, time

from anyio import create_task_group
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.client.streamable_http import streamable_http_client
from mcp.server.stdio import stdio_server
from rich import print as rprint

from axon import __version__
from axon.cli.helpers.host_helpers import (
    HostStateHelper,
    StorageHelper,
    UIRunner,
    UpdateChecker,
)
from axon.cli.helpers.host_helpers.configs import (
    BrowserConfig,
    HostBehaviorConfig,
    HostConfig,
    HostStartupConfig,
    RunnerConfig,
    ServerConfig,
    StartupDisplayConfig,
)

logger = getLogger(__name__)

UPDATE_CHECK_SKIP_COMMANDS = {"mcp", "serve", "host"}


class HostRunner:
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
        storage_helper: Helper for storage operations.
        update_checker: Helper for version checking.
        host_state_helper: Helper for host state management.
        ui_runner: Helper for UI operations.

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
        # Composition: Use helper classes instead of inheritance
        self._storage = StorageHelper()
        self._update_checker = UpdateChecker()
        self._host_state = HostStateHelper()
        self._ui_runner = UIRunner(
            storage_helper=self._storage,
            host_state_helper=self._host_state,
        )

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
        if (
            latest := self._update_checker.get_latest_version()
        ) and self._update_checker.is_newer_version(
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

        live_host = self._host_state.get_live_host_info(repo_path)
        if live_host is not None:
            return live_host

        self._host_state.start_host_background(
            repo_path,
            port=config.port,
            bind=config.bind,
            watch=config.watch,
            managed=config.managed,
        )
        deadline = time() + config.timeout_seconds
        while time() < deadline:
            live_host = self._host_state.get_live_host_info(repo_path)
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
        return self._host_state.create_host_lease(repo_path, lease_type)

    def remove_host_lease(self, lease_path: Path | None) -> None:
        """
        Remove a host lease file.

        Args:
            lease_path: Path to the lease file to remove.
        """
        self._host_state.remove_host_lease(lease_path)

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
            live_host = self._host_state.get_live_host_info(repo_path)
            if live_host is not None and self._ui_runner.check_live_host_for_ui(
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
            self._ui_runner.run_shared_ui_host(
                server_config=server_config,
                browser_config=browser_config,
                behavior_config=behavior_config,
                startup_display_config=startup_display_config,
                startup_config=startup_config,
            )
            return

        # Direct mode - run standalone UI
        self._ui_runner.run_standalone_ui(
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
