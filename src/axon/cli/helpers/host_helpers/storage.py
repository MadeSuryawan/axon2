"""Storage helper for Axon CLI operations."""

from datetime import UTC, datetime
from hashlib import sha256
from json import JSONDecodeError, dumps, loads
from logging import getLogger
from pathlib import Path
from shutil import rmtree

from rich import print as rprint
from typer import Exit

from axon import __version__
from axon.core.ingestion.pipeline import PipelineResult, Pipelines
from axon.core.storage.kuzu_backend import KuzuBackend

logger = getLogger(__name__)


class StorageHelper:
    """
    Handles storage initialization and management.

    This class provides methods for loading and initializing the KuzuDB
    storage backend, as well as building metadata and registry operations.
    """

    # ==================== Storage and Initialization ====================

    def load_storage(self, repo_path: Path | None = None) -> KuzuBackend:
        """Load the KuzuDB backend for the given or current repo (read-only)."""
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

    def initialize_writable_storage(
        self,
        repo_path: Path,
        *,
        auto_index: bool = True,
    ) -> tuple[KuzuBackend, Path, Path]:
        """Open the repo database in read-write mode."""
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
            meta = self.build_meta(result, repo_path)
            meta_path = axon_dir / "meta.json"
            meta_path.write_text(dumps(meta, indent=2) + "\n", encoding="utf-8")
            try:
                self.register_in_global_registry(meta, repo_path)
            except (RuntimeError, OSError, SystemError, PermissionError):
                logger.debug("Failed to register repo in global registry", exc_info=True)

        return storage, axon_dir, db_path

    # ==================== Registry Management ====================

    @staticmethod
    def register_in_global_registry(meta: dict, repo_path: Path) -> None:
        """Write meta.json into ~/.axon/repos/{slug}/ for multi-repo discovery."""

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
                rmtree(candidate, ignore_errors=True)

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
    def build_meta(result: PipelineResult, repo_path: Path) -> dict:
        """Build metadata dictionary from pipeline results."""

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
