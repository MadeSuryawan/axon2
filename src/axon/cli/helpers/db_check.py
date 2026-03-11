from asyncio import Event, Lock, gather
from contextlib import suppress
from datetime import UTC, datetime
from hashlib import sha256
from json import JSONDecodeError, dumps, loads
from logging import getLogger
from pathlib import Path
from shutil import rmtree
from sys import stderr
from typing import Any

from mcp.server.stdio import stdio_server
from rich import print as rprint
from typer import Exit, Option

from axon import __version__
from axon.core.ingestion.pipeline import PipelineResult, Pipelines
from axon.core.ingestion.watcher import Watcher, WatcherDeps
from axon.core.storage.kuzu_backend import KuzuBackend
from axon.mcp.server import server as mcp_server
from axon.mcp.server import set_lock, set_storage

logger = getLogger(__name__)


# Boolean defaults to avoid FBT003 errors
_FALSE = False
_TRUE = True


def version_callback(*, value: bool) -> None:
    """Print the version and exit."""
    if value:
        rprint(f"Axon v{__version__}")
        raise Exit()


async def start_serve(
    repo_path: Path,
    storage: KuzuBackend,
) -> None:
    lock = Lock()
    set_storage(storage)
    set_lock(lock)
    stop = Event()

    deps = WatcherDeps(repo_path, storage, stop_event=stop, lock=lock)

    async with stdio_server() as (read, write):

        async def _mcp_then_stop() -> None:
            await mcp_server.run(
                read,
                write,
                mcp_server.create_initialization_options(),
            )
            stop.set()

        await gather(
            _mcp_then_stop(),
            Watcher(deps).watch(),
        )


def get_kuzu(db_path: Path, *, read_only: bool = False) -> KuzuBackend:
    """Return a KuzuBackend initialised at *db_path*."""
    storage = KuzuBackend()
    storage.initialize(db_path, read_only=read_only)
    rprint("[b green]KuzuDB initialised")
    return storage


def get_path(path: Path | None = None) -> tuple[Path, Path, Path]:
    """Return (repo_path, axon_dir, db_path) for the given path."""
    repo_path = Path.cwd().resolve() if not path else path.resolve()
    if not repo_path.is_dir():
        rprint(f"[red]Error:[/red] {repo_path} is not a directory.")
        raise Exit(code=1)

    axon_dir = repo_path / ".axon"
    axon_dir.mkdir(parents=True, exist_ok=True)
    db_path = axon_dir / "kuzu"

    return repo_path, axon_dir, db_path


def load_storage(repo_path: Path | None = None) -> KuzuBackend:
    """Load the KuzuDB backend for the given or current repo."""

    target = (repo_path or Path.cwd()).resolve()
    db_path = target / ".axon" / "kuzu"
    if not db_path.exists():
        rprint(
            f"[red]Error:[/red] No index found at {target}. Run 'axon analyze' first.",
        )
        raise Exit(code=1)

    return get_kuzu(db_path, read_only=True)


def report(result: PipelineResult) -> None:

    rprint()
    rprint("[bold green]Indexing complete.[/bold green]")
    rprint(f"  Files:          {result.files}")
    rprint(f"  Symbols:        {result.symbols}")
    rprint(f"  Relationships:  {result.relationships}")
    if result.clusters > 0:
        rprint(f"  Clusters:       {result.clusters}")
    if result.processes > 0:
        rprint(f"  Flows:          {result.processes}")
    if result.dead_code > 0:
        rprint(f"  Dead code:      {result.dead_code}")
    if result.coupled_pairs > 0:
        rprint(f"  Coupled pairs:  {result.coupled_pairs}")
    if result.embeddings > 0:
        rprint(f"  Embeddings:     {result.embeddings}")
    rprint(f"  Duration:       {result.duration_seconds:.2f}s")


def process_meta(
    axon_dir: Path,
    repo_path: Path,
    storage: KuzuBackend,
    *,
    no_embeddings: bool = Option(
        _FALSE,
        "--no-embeddings",
        help="Skip vector embedding generation.",
    ),
) -> PipelineResult:
    """Check if meta.json exists, and if not, run initial indexing."""

    pipelines = Pipelines(repo_path, storage, full=True, embeddings=not no_embeddings)
    pipelines.run_pipelines()

    meta = _build_meta(pipelines.result, repo_path)
    meta_path = axon_dir / "meta.json"
    meta_path.write_text(dumps(meta, indent=2) + "\n", encoding="utf-8")

    try:
        _register_in_global_registry(meta, repo_path)
    except (RuntimeError, OSError, SystemError):
        logger.debug("Failed to register repo in global registry", exc_info=True)

    return pipelines.result


def check_meta_json(
    axon_dir: Path,
    repo_path: Path,
    storage: KuzuBackend,
    *,
    no_embeddings: bool,
) -> None:
    if not (axon_dir / "meta.json").exists():
        rprint("[b yellow]Un-initialized repo, running initial index....", file=stderr)
        report(process_meta(axon_dir, repo_path, storage, no_embeddings=no_embeddings))


def _build_meta(result: PipelineResult, repo_path: Path) -> dict[str, Any]:
    """Build the meta.json dict from a pipeline result."""
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


def _register_in_global_registry(meta: dict, repo_path: Path) -> None:
    """
    Write meta.json into ``~/.axon/repos/{slug}/`` for multi-repo discovery.

    Slug is ``{repo_name}`` if that slot is unclaimed or already belongs to
    this repo.  Falls back to ``{repo_name}-{sha256(path)[:8]}`` on collision.
    """
    registry_root = Path.home() / ".axon" / "repos"
    repo_name = repo_path.name
    candidate = registry_root / repo_name

    slug = _get_slug(repo_name, candidate, repo_path)
    _remove_stale_entry(registry_root, slug, repo_path)

    slot = registry_root / slug
    slot.mkdir(parents=True, exist_ok=True)

    registry_meta = dict(meta)
    registry_meta["slug"] = slug
    (slot / "meta.json").write_text(dumps(registry_meta, indent=2) + "\n", encoding="utf-8")


def _get_slug(repo_name: str, candidate: Path, repo_path: Path) -> str:
    """Repository metadata."""
    slug = repo_name
    if candidate.exists():
        existing_meta_path = candidate / "meta.json"
        try:
            existing = loads(existing_meta_path.read_text())
            if existing.get("path") != str(repo_path):
                short_hash = sha256(str(repo_path).encode()).hexdigest()[:8]
                slug = f"{repo_name}-{short_hash}"
        except (JSONDecodeError, OSError):
            rmtree(candidate, ignore_errors=True)  # Clean broken slot before claiming
    return slug


def _remove_stale_entry(registry_root: Path, slug: str, repo_path: Path) -> None:
    """Remove any stale entry for the same repo_path under a different slug."""

    if not registry_root.exists():
        return

    for old_dir in registry_root.iterdir():
        if not old_dir.is_dir() or old_dir.name == slug:
            continue
        old_meta = old_dir / "meta.json"
        with suppress(Exception):
            old_data = loads(old_meta.read_text())
            if old_data.get("path") != str(repo_path):
                continue
            rmtree(old_dir, ignore_errors=True)
