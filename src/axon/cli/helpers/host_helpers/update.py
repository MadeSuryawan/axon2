"""Update checker helper for Axon CLI operations."""

from json import JSONDecodeError, dumps, loads
from pathlib import Path
from time import time
from urllib.error import URLError
from urllib.request import urlopen

# Constants
UPDATE_CHECK_INTERVAL_SECONDS = 60 * 60 * 24  # 24 hours
UPDATE_CHECK_URL = "https://pypi.org/pypi/axoniq/json"


class UpdateChecker:
    """
    Handles version checking and notifications.

    This class provides methods for checking if a newer version of Axon
    is available on PyPI and caching the results.
    """

    # ==================== Update Cache Path ====================

    @staticmethod
    def _update_cache_path() -> Path:
        """Get the path to the update check cache file."""
        return Path.home() / ".axon" / "update-check.json"

    # ==================== Version Parsing ====================

    @staticmethod
    def _parse_version_parts(version: str) -> tuple[int, ...]:
        """Parse a version string into numeric parts."""
        parts: list[int] = []
        for raw_part in version.split("."):
            digits = "".join(ch for ch in raw_part if ch.isdigit())
            parts.append(int(digits or 0))
        return tuple(parts)

    def is_newer_version(self, candidate: str, current: str) -> bool:
        """Check if candidate version is newer than current version."""
        return self._parse_version_parts(candidate) > self._parse_version_parts(current)

    # ==================== Cache Management ====================

    def _read_update_cache(self) -> dict | None:
        """Read the update check cache from disk."""
        cache_path = self._update_cache_path()
        if not cache_path.exists():
            return None
        try:
            return loads(cache_path.read_text(encoding="utf-8"))
        except (JSONDecodeError, OSError):
            return None

    def _write_update_cache(self, payload: dict) -> None:
        """Write the update check cache to disk."""
        cache_path = self._update_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(dumps(payload, indent=2) + "\n", encoding="utf-8")

    # ==================== Version Fetching ====================

    @staticmethod
    def _fetch_latest_version() -> str | None:
        """Fetch the latest Axon version from PyPI."""
        try:
            with urlopen(UPDATE_CHECK_URL, timeout=1.5) as response:  # noqa: S310
                payload = loads(response.read().decode("utf-8"))
                return str(payload["info"]["version"])
        except (KeyError, OSError, ValueError, URLError):
            return None

    def get_latest_version(self) -> str | None:
        """Get the latest Axon version, using cache if available."""
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
