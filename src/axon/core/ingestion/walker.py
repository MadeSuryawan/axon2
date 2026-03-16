"""File system walker for discovering and reading source files in a repository."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from os import walk
from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired
from subprocess import run as subprocess_run

from axon.config.ignore import DEFAULT_IGNORE_PATTERNS, should_ignore
from axon.config.languages import get_language, is_supported

_PRUNE_DIRS = frozenset(
    p for p in DEFAULT_IGNORE_PATTERNS if ("*" not in p and "." not in p) or p.startswith(".")
) | frozenset(
    {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".env",
        "dist",
        "build",
        ".idea",
        ".vscode",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".eggs",
        ".axon",
    },
)

cmmnd = ["git", "ls-files", "--cached", "--others", "--exclude-standard"]


@dataclass
class FileEntry:
    """A source file discovered during walking."""

    path: str  # relative path from repo root (e.g., "src/auth/validate.py")
    content: str  # full file content
    language: str  # "python", "typescript", "javascript"


def _discover_via_git(repo_path: Path, gitignore_patterns: list[str] | None) -> list[Path] | None:
    try:
        result = subprocess_run(
            cmmnd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        if result.returncode != 0:
            return
    except (TimeoutExpired, FileNotFoundError):
        return

    return _process_git_output(result, repo_path, gitignore_patterns)


def _process_git_output(
    result: CompletedProcess[str],
    repo_path: Path,
    gitignore_patterns: list[str] | None,
) -> list[Path]:
    discovered: list[Path] = []
    for line in result.stdout.splitlines():
        if not (striped := line.strip()):
            continue
        if should_ignore(striped, gitignore_patterns):
            continue
        full = repo_path / striped
        if is_supported(full):
            discovered.append(full)
    return discovered


def _discover_via_walk(repo_path: Path, gitignore_patterns: list[str] | None) -> list[Path]:
    discovered: list[Path] = []

    for dirpath, dirnames, filenames in walk(repo_path, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in _PRUNE_DIRS and not d.endswith(".egg-info")]

        for fname in filenames:
            full = Path(dirpath) / fname
            try:
                relative = full.relative_to(repo_path)
            except ValueError:
                continue

            if should_ignore(str(relative), gitignore_patterns):
                continue
            if not is_supported(full):
                continue
            discovered.append(full)

    return discovered


def discover_files(repo_path: Path, gitignore_patterns: list[str] | None = None) -> list[Path]:
    """
    Discover supported source file paths without reading their content.

    Walks *repo_path* recursively and returns paths that are not ignored and
    have a supported language extension.  Useful for incremental indexing where
    you want to check paths before reading.

    Parameters
    ----------
    repo_path:
        Root directory of the repository to walk.
    gitignore_patterns:
        Optional list of gitignore-style patterns (e.g. from
        :func:`axon.config.ignore.load_gitignore`).

    Returns
    -------
    list[Path]
        List of absolute :class:`Path` objects for each discovered file.
    """
    if result := _discover_via_git((resolved := repo_path.resolve()), gitignore_patterns):
        return result

    return _discover_via_walk(resolved, gitignore_patterns)


def read_file(repo_path: Path, file_path: Path) -> FileEntry | None:
    """
    Read a single file and return a :class:`FileEntry`, or ``None`` on failure.

    Returns ``None`` when the file cannot be decoded as UTF-8 (binary files),
    when the file is empty, or when an OS-level error occurs.
    """
    relative = file_path.relative_to(repo_path)

    try:
        if not (content := file_path.read_text(encoding="utf-8")):
            return
    except (UnicodeDecodeError, ValueError, OSError):
        return

    if not (language := get_language(file_path)):
        return

    return FileEntry(
        path=str(relative),
        content=content,
        language=language,
    )


def walk_repo(
    repo_path: Path,
    gitignore_patterns: list[str] | None = None,
    max_workers: int = 8,
) -> list[FileEntry]:
    """
    Walk a repository and return all supported source files with their content.

    Discovers files using the same filtering logic as :func:`discover_files`,
    then reads their content in parallel using a :class:`ThreadPoolExecutor`.

    Parameters
    ----------
    repo_path:
        Root directory of the repository to walk.
    gitignore_patterns:
        Optional list of gitignore-style patterns (e.g. from
        :func:`axon.config.ignore.load_gitignore`).
    max_workers:
        Maximum number of threads for parallel file reading.  Defaults to 8.

    Returns
    -------
    list[FileEntry]
        Sorted (by path) list of :class:`FileEntry` objects for every
        discovered source file.
    """
    repo_path = repo_path.resolve()
    file_paths = discover_files(repo_path, gitignore_patterns)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(lambda fp: read_file(repo_path, fp), file_paths)

    return sorted([entry for entry in results if entry], key=lambda e: e.path)
