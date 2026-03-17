"""Host state helper for Axon CLI operations."""

from json import JSONDecodeError, dumps, loads
from logging import getLogger
from os import devnull, getpid, kill
from pathlib import Path
from subprocess import Popen
from sys import executable
from time import time
from urllib.error import URLError
from urllib.request import urlopen
from uuid import uuid4

from rich import print as rprint
from typer import Exit

from axon import __version__
from axon.cli.helpers.host_helpers.configs import DEFAULT_HOST, DEFAULT_PORT
from axon.core.storage.kuzu_backend import KuzuBackend

logger = getLogger(__name__)


class HostStateHelper:
    """
    Manages host metadata, leases, and lifecycle.

    This class provides methods for handling host state files, lease
    management for managed shutdown, and checking host availability.
    """

    # ==================== Host Metadata ====================

    @staticmethod
    def _host_meta_path(repo_path: Path) -> Path:
        """Get the path to the host metadata file."""
        return repo_path / ".axon" / "host.json"

    @staticmethod
    def _host_lease_dir(repo_path: Path) -> Path:
        """Get the path to the host lease directory."""
        return repo_path / ".axon" / "host-leases"

    @staticmethod
    def _display_host(host: str) -> str:
        """Normalize host address for display."""
        return "127.0.0.1" if host in {"0.0.0.0", "::"} else host  # noqa: S104

    def build_host_urls(self, host: str, port: int) -> tuple[str, str]:
        """Build the base URL and MCP URL for the host."""
        base = f"http://{self._display_host(host)}:{port}"
        return base, f"{base}/mcp"

    def read_host_meta(self, repo_path: Path) -> dict | None:
        """Read host metadata from disk."""
        meta_path = self._host_meta_path(repo_path)
        if not meta_path.exists():
            return None
        try:
            return loads(meta_path.read_text(encoding="utf-8"))
        except (JSONDecodeError, OSError):
            return None

    def write_host_meta(
        self,
        repo_path: Path,
        host_url: str,
        mcp_url: str,
        port: int,
        *,
        ui_enabled: bool,
    ) -> None:
        """Write host metadata to disk."""
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

    def clear_host_meta(self, repo_path: Path) -> None:
        """Remove host metadata file on shutdown."""
        meta_path = self._host_meta_path(repo_path)
        if meta_path.exists():
            meta_path.unlink(missing_ok=True)

    # ==================== Lease Management ====================

    def create_host_lease(self, repo_path: Path, lease_type: str) -> Path:
        """Create a new host lease file."""
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
        """Remove a host lease file."""
        if lease_path is not None:
            lease_path.unlink(missing_ok=True)

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        """Check if a process ID is still running."""
        try:
            kill(pid, 0)
        except OSError:
            return False
        return True

    def count_live_host_leases(self, repo_path: Path) -> int:
        """Count the number of active host leases."""
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
        """Check if a host is running by querying its health endpoint."""
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

    def get_live_host_info(self, repo_path: Path) -> dict | None:
        """Get information about a running host, if any."""
        meta = self.read_host_meta(repo_path)
        if meta is None:
            return None
        if self._is_host_alive(meta, repo_path):
            return meta
        return None

    # ==================== Process Management ====================

    @staticmethod
    def start_host_background(
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

    # ==================== Version Callback ====================

    @staticmethod
    def version_callback(*, value: bool) -> None:
        """Handle version flag callback."""
        if value:
            rprint(f"Axon v{__version__}")
            raise Exit()

    # ==================== Host Cleanup ====================

    def cleanup_host(self, repo_path: Path, storage: KuzuBackend) -> None:
        """Clean up host resources on shutdown."""
        self.clear_host_meta(repo_path)
        storage.close()
