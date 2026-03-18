"""Axon configuration — ignore patterns and language detection."""

from axon.config.ignore import DEFAULT_IGNORE_PATTERNS, load_gitignore, should_ignore
from axon.config.languages import SUPPORTED_EXTENSIONS, get_language, is_supported
from axon.config.model_config import (
    clear_model_name,
    get_model_for_embedding,
    get_model_name,
    set_model_name,
)

__all__ = [
    "DEFAULT_IGNORE_PATTERNS",
    "SUPPORTED_EXTENSIONS",
    "get_language",
    "is_supported",
    "load_gitignore",
    "should_ignore",
    "get_model_name",
    "set_model_name",
    "get_model_for_embedding",
    "clear_model_name",
]
