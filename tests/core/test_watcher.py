"""Tests for the watch mode module (watcher.py)."""

from asyncio import run as asyncio_run
from collections.abc import Generator
from os import environ
from pathlib import Path
from shutil import which
from subprocess import run as subprocess_run

import pytest

from axon.core.graph.model import NodeLabel
from axon.core.ingestion.pipeline import reindex_files, run_pipeline
from axon.core.ingestion.walker import FileEntry, read_file
from axon.core.ingestion.watcher import (
    Watcher,
    WatcherDeps,
)
from axon.core.storage.kuzu_backend import KuzuBackend

GIT_BIN = which("git") or "git"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_repo(tmp_path: Path) -> Path:
    """Create a small Python repository for watcher tests."""
    src = tmp_path / "src"
    src.mkdir()

    (src / "app.py").write_text(
        "def hello():\n    return 'hello'\n",
        encoding="utf-8",
    )

    (src / "utils.py").write_text(
        "def helper():\n    pass\n",
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture()
def storage(tmp_path: Path) -> Generator[KuzuBackend]:
    """Provide an initialised KuzuBackend for testing."""
    db_path = tmp_path / "test_db"
    backend = KuzuBackend()
    backend.initialize(db_path)
    yield backend
    backend.close()


# ---------------------------------------------------------------------------
# Tests: read_file
# ---------------------------------------------------------------------------


class TestReadFileEntry:
    """read_file reads a file and returns a FileEntry."""

    def test_reads_python_file(self, tmp_repo: Path) -> None:
        entry = read_file(tmp_repo, tmp_repo / "src" / "app.py")

        assert entry is not None
        assert entry.path == "src/app.py"
        assert entry.language == "python"
        assert "hello" in entry.content

    def test_returns_none_for_unsupported(self, tmp_repo: Path) -> None:
        readme = tmp_repo / "README.md"
        readme.write_text("# readme", encoding="utf-8")

        entry = read_file(tmp_repo, readme)

        assert entry is None

    def test_returns_none_for_missing(self, tmp_repo: Path) -> None:
        entry = read_file(tmp_repo, tmp_repo / "nonexistent.py")

        assert entry is None

    def test_returns_none_for_empty(self, tmp_repo: Path) -> None:
        empty = tmp_repo / "empty.py"
        empty.write_text("", encoding="utf-8")

        entry = read_file(tmp_repo, empty)

        assert entry is None


# ---------------------------------------------------------------------------
# Tests: reindex_files (pipeline function)
# ---------------------------------------------------------------------------


class TestReindexFiles:
    """reindex_files() correctly removes old nodes and adds new ones."""

    def test_reindex_updates_content(
        self,
        tmp_repo: Path,
        storage: KuzuBackend,
    ) -> None:
        # Initial full index.
        run_pipeline(tmp_repo, storage)

        # Verify initial node exists.
        node = storage.get_node("function:src/app.py:hello")
        assert node is not None
        assert "hello" in node.content

        # Modify the file.
        (tmp_repo / "src" / "app.py").write_text(
            "def hello():\n    return 'goodbye'\n",
            encoding="utf-8",
        )

        # Re-read and reindex.
        entry = FileEntry(
            path="src/app.py",
            content=(tmp_repo / "src" / "app.py").read_text(),
            language="python",
        )
        reindex_files([entry], tmp_repo, storage)

        # Verify updated node.
        node = storage.get_node("function:src/app.py:hello")
        assert node is not None
        assert "goodbye" in node.content


# ---------------------------------------------------------------------------
# Tests: Watcher
# ---------------------------------------------------------------------------


class TestRepositoryWatcher:
    """Tests for the Watcher class."""

    def test_reindexes_changed_files(
        self,
        tmp_repo: Path,
        storage: KuzuBackend,
    ) -> None:
        watcher = Watcher(WatcherDeps(repo_path=tmp_repo, storage=storage))

        # Modify a file.
        app_path = tmp_repo / "src" / "app.py"
        app_path.write_text(
            "def hello():\n    return 'updated'\n",
            encoding="utf-8",
        )

        count, paths = watcher._reindex_files([app_path])

        assert count == 1
        node = storage.get_node("function:src/app.py:hello")
        assert node is not None
        assert "updated" in node.content
        assert "src/app.py" in paths

    def test_skips_ignored_files(
        self,
        tmp_repo: Path,
        storage: KuzuBackend,
    ) -> None:
        watcher = Watcher(WatcherDeps(repo_path=tmp_repo, storage=storage))
        run_pipeline(tmp_repo, storage)

        # Create a file in an ignored directory.
        cache_dir = tmp_repo / "__pycache__"
        cache_dir.mkdir()
        cached = cache_dir / "module.cpython-311.pyc"
        cached.write_bytes(b"\x00")

        count, _paths = watcher._reindex_files([cached])

        assert count == 0

    def test_handles_deleted_files(
        self,
        tmp_repo: Path,
        storage: KuzuBackend,
    ) -> None:
        watcher = Watcher(WatcherDeps(repo_path=tmp_repo, storage=storage))
        run_pipeline(tmp_repo, storage)

        # File exists in storage but is now deleted from disk.
        deleted_path = tmp_repo / "src" / "app.py"
        assert storage.get_node("file:src/app.py:") is not None

        deleted_path.unlink()

        count, paths = watcher._reindex_files([deleted_path])

        assert count == 0
        assert "src/app.py" in paths
        # Node should be gone from storage.
        assert storage.get_node("file:src/app.py:") is None

    def test_last_change_time_updates_on_reindex(
        self,
        tmp_repo: Path,
        storage: KuzuBackend,
    ) -> None:
        watcher = Watcher(WatcherDeps(repo_path=tmp_repo, storage=storage))
        assert watcher._last_change_time == 0.0

        app_path = tmp_repo / "src" / "app.py"
        # We need to call the async wrapper or set up the state.
        asyncio_run(watcher._reindex_changed_paths([app_path]))

        assert watcher._last_change_time > 0.0

    def test_changed_paths_cleared_after_processing(
        self,
        tmp_repo: Path,
        storage: KuzuBackend,
    ) -> None:
        watcher = Watcher(WatcherDeps(repo_path=tmp_repo, storage=storage))
        app_path = tmp_repo / "src" / "app.py"
        watcher._changed_paths.append(app_path)

        # This is what Watcher.watch does:
        if watcher._changed_paths:
            batch = watcher._changed_paths.copy()
            watcher._changed_paths.clear()
            asyncio_run(watcher._reindex_changed_paths(batch))

        assert len(watcher._changed_paths) == 0


class TestGetHeadSha:
    """_get_head_sha returns the current git HEAD."""

    def test_returns_sha_in_git_repo(self, tmp_repo: Path, storage: KuzuBackend) -> None:
        subprocess_run([GIT_BIN, "init"], cwd=tmp_repo, capture_output=True, check=True)
        subprocess_run([GIT_BIN, "add", "."], cwd=tmp_repo, capture_output=True, check=True)
        env = {
            **environ,
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "t@t",
        }
        subprocess_run(
            [GIT_BIN, "commit", "-m", "init"],
            cwd=tmp_repo,
            capture_output=True,
            env=env,
            check=True,
        )
        watcher = Watcher(WatcherDeps(repo_path=tmp_repo, storage=storage))
        sha = watcher._get_head_sha()
        assert sha is not None
        assert len(sha) == 40

    def test_returns_none_outside_git_repo(self, tmp_path: Path, storage: KuzuBackend) -> None:
        watcher = Watcher(WatcherDeps(repo_path=tmp_path, storage=storage))
        sha = watcher._get_head_sha()
        assert sha is None


class TestRunIncrementalGlobalPhases:
    """Integration test: stale synthetic nodes are not re-persisted."""

    def test_no_stale_synthetic_nodes_after_rerun(
        self,
        tmp_repo: Path,
        storage: KuzuBackend,
    ) -> None:
        run_pipeline(tmp_repo, storage, full=True, embeddings=False)

        watcher = Watcher(WatcherDeps(repo_path=tmp_repo, storage=storage))
        watcher._dirty_files = {"src/app.py"}

        # First incremental run.
        watcher._run_incremental_global_phases()
        graph1 = storage.load_graph()
        comm_count_1 = len(list(graph1.get_nodes_by_label(NodeLabel.COMMUNITY)))

        # Second incremental run.
        watcher._run_incremental_global_phases()
        graph2 = storage.load_graph()
        comm_count_2 = len(list(graph2.get_nodes_by_label(NodeLabel.COMMUNITY)))

        assert comm_count_2 == comm_count_1
