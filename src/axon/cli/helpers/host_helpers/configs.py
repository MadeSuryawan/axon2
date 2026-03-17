"""Shared dataclasses for host operations."""

from asyncio import Lock
from dataclasses import dataclass
from pathlib import Path

from axon.core.storage.kuzu_backend import KuzuBackend

# Default host configuration
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8420
DEFAULT_MANAGED_PORT = 8421


@dataclass(frozen=True)
class HostConfig:
    """Configuration for host processes."""

    port: int = DEFAULT_PORT
    bind: str = DEFAULT_HOST
    watch: bool = True
    timeout_seconds: float = 10.0
    managed: bool = False


@dataclass(frozen=True)
class HostURLs:
    """URLs for the Axon host services."""

    host_url: str
    mcp_url: str


@dataclass(frozen=True)
class WebAppConfig:
    """Configuration for web application behavior."""

    watch: bool = True
    dev: bool = False
    expose_ui: bool = True


@dataclass(frozen=True)
class StartupConfig:
    """Configuration for startup behavior and announcements."""

    watch: bool = True
    dev: bool = False
    announce_ui: bool = True
    announce_mcp: bool = False
    open_browser: bool = True


@dataclass(frozen=True)
class RunnerConfig:
    """Configuration for HostRunner behavior."""

    port: int = DEFAULT_PORT
    no_open: bool = False
    watch: bool = True
    dev: bool = False
    direct: bool = False
    managed: bool = False


@dataclass(frozen=True)
class ServerConfig:
    """Configuration for the async server."""

    bind: str = DEFAULT_HOST
    port: int = DEFAULT_PORT


@dataclass(frozen=True)
class RuntimeContext:
    """Runtime context for the host."""

    repo_path: Path
    storage: KuzuBackend
    lock: Lock


@dataclass(frozen=True)
class HostBehaviorConfig:
    """Configuration for host behavior options."""

    watch: bool = True
    dev: bool = False
    managed: bool = False


@dataclass(frozen=True)
class BrowserConfig:
    """Configuration for browser behavior."""

    no_open: bool = False
    open_browser: bool = True


@dataclass(frozen=True)
class StartupDisplayConfig:
    """Configuration for startup display and announcements."""

    announce_ui: bool = True
    announce_mcp: bool = False
    expose_ui: bool = True


@dataclass(frozen=True)
class HostStartupConfig:
    """Configuration for host startup behavior."""

    already_running_message: str = "[bold green]Axon UI available at {url}"
    auto_index: bool = True
