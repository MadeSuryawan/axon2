"""Model configuration management for Axon embeddings."""

from contextlib import suppress
from json import JSONDecodeError, dumps, loads
from logging import getLogger
from os import replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

logger = getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
LARGE_MODEL = "BAAI/bge-large-en-v1.5"
MODEL_OPTIONS = {
    "small": DEFAULT_MODEL,
    "s": DEFAULT_MODEL,
    "large": LARGE_MODEL,
    "l": LARGE_MODEL,
}
MODEL_CHOICES = list(MODEL_OPTIONS.keys())
CONFIG_FILE_NAME = "config.json"


def _get_config_path(repo_path: Path) -> Path:
    """Get the path to the config file for the given repository."""
    axon_dir = repo_path / ".axon"
    return axon_dir / CONFIG_FILE_NAME


def _load_config(repo_path: Path) -> dict[str, Any]:
    """Load the config file for the given repository."""
    config_path = _get_config_path(repo_path)
    if not config_path.exists():
        return {}

    try:
        return loads(config_path.read_text(encoding="utf-8"))
    except (JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load config from {config_path}: {e}")
        return {}


def _save_config(repo_path: Path, config: dict[str, Any]) -> None:
    """Save the config file for the given repository using atomic write."""
    config_path = _get_config_path(repo_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: write to temp file then rename (safe for concurrent access)
    tmp_path: Path | None = None
    try:
        # Write to a temporary file in the same directory
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=config_path.parent,
            delete=False,
        ) as tmp_file:
            tmp_file.write(dumps(config, indent=2, ensure_ascii=False) + "\n")
            tmp_path = Path(tmp_file.name)

        # Atomic rename (on POSIX systems, this is atomic; on Windows, it's close enough)
        replace(tmp_path, config_path)
    except OSError as e:
        logger.warning(f"Failed to save config to {config_path}: {e}")
        # Clean up temp file if it exists
        if tmp_path is not None and tmp_path.exists():
            with suppress(OSError):
                tmp_path.unlink()


def get_model_name(path: Path | None = None) -> str | None:
    """
    Get the configured model name for the given repository.

    Args:
        path: Path to the repository.

    Returns:
        The model name if configured, None otherwise.
    """
    repo_path = Path.cwd().resolve() if not path else path.resolve()
    config = _load_config(repo_path)
    return config.get("model_name")


def set_model_name(model_name: str, repo_path: Path) -> None:
    """
    Set the model name for the given repository.

    Args:
        model_name: The model name to set. Must be a non-None string.
        repo_path: Path to the repository.
    """
    config = _load_config(repo_path)
    config["model_name"] = model_name
    _save_config(repo_path, config)


def clear_model_name(repo_path: Path | None = None) -> None:
    """
    Clear the model name setting for the given repository.

    Args:
        repo_path: Path to the repository. If None, uses current working directory.
    """
    if repo_path is None:
        repo_path = Path.cwd()

    config = _load_config(repo_path)
    config.pop("model_name", None)
    _save_config(repo_path, config)


def get_model_for_embedding(repo_path: Path | None = None) -> str:
    """
    Get the model to use for embedding generation.

    This function checks for a stored model configuration and falls back
    to the default model if none is configured or if repo_path is not provided.

    Args:
        repo_path: Path to the repository. If None, returns the default model.

    Returns:
        The model name to use for embeddings.
    """
    if repo_path is None:
        # No repo_path provided, use default
        return DEFAULT_MODEL

    if configured_model := get_model_name(repo_path):
        return configured_model
    return DEFAULT_MODEL
