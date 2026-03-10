"""Axon CLI — Graph-powered code intelligence engine."""

from asyncio import Event, Lock, gather
from asyncio import run as asyncio_run
from contextlib import suppress
from datetime import UTC, datetime
from hashlib import sha256
from json import JSONDecodeError, dumps, loads
from logging import basicConfig, getLogger
from pathlib import Path
from shutil import rmtree
from sys import platform, stderr
from typing import Any

from mcp.server.stdio import stdio_server
from rich import print as rprint
from rich.logging import RichHandler
from rich.traceback import install
from typer import Argument, Exit, Option, Typer, confirm

from axon import __version__
from axon.core.diff import diff_branches, format_diff
from axon.core.ingestion.pipeline import PipelineResult, Pipelines
from axon.core.ingestion.watcher import Watcher, WatcherDeps
from axon.core.storage.kuzu_backend import KuzuBackend
from axon.mcp.server import main as mcp_main
from axon.mcp.server import server as mcp_server
from axon.mcp.server import set_lock, set_storage
from axon.mcp.tools import Tools

if platform in ("win32", "cygwin", "cli"):
    with suppress(ImportError):
        # will not installed on macOS/Linux, ignored Pyrefly(missing-import) error
        from winloop import install as winloop_install  # type: ignore[import]

        winloop_install()
        rprint("[b blue]Event loop policy: [b green]winloop (Windows)")
else:
    with suppress(ImportError):
        # will not installed on Windows, ignored Pyrefly(missing-import) error
        from uvloop import install as uvloop_install  # type: ignore[import]

        uvloop_install()
        rprint("[b blue]Event loop policy: [b green]uvloop (Linux/macOS)")

basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="%X",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = getLogger("rich")
install()


# Module-level singletons to avoid typer calls in function defaults
_REPO_OPTION = Option(
    None,
    "--repo",
    "-r",
    help="Path to the repository (defaults to auto-discovery).",
)

_PATH_ARG = Argument(Path("."), help="Path to the repository to index.")

# Boolean defaults to avoid FBT003 errors
_FALSE = False
_TRUE = True
_tools = Tools()


def _get_kuzu(db_path: Path, *, read_only: bool = False) -> KuzuBackend:
    """Return a KuzuBackend initialised at *db_path*."""
    storage = KuzuBackend()
    storage.initialize(db_path, read_only=read_only)
    rprint("[b green]KuzuDB initialised")
    return storage


def _get_path(path: Path | None = None) -> tuple[Path, Path, Path]:
    """Return (repo_path, axon_dir, db_path) for the given path."""
    repo_path = Path.cwd().resolve() if not path else path.resolve()
    if not repo_path.is_dir():
        rprint(f"[red]Error:[/red] {repo_path} is not a directory.")
        raise Exit(code=1)

    axon_dir = repo_path / ".axon"
    axon_dir.mkdir(parents=True, exist_ok=True)
    db_path = axon_dir / "kuzu"

    return repo_path, axon_dir, db_path


def _load_storage(repo_path: Path | None = None) -> KuzuBackend:
    """Load the KuzuDB backend for the given or current repo."""

    target = (repo_path or Path.cwd()).resolve()
    db_path = target / ".axon" / "kuzu"
    if not db_path.exists():
        rprint(
            f"[red]Error:[/red] No index found at {target}. Run 'axon analyze' first.",
        )
        raise Exit(code=1)

    return _get_kuzu(db_path, read_only=True)


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


def _report(result: PipelineResult) -> None:

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


def _process_meta(
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


def _check_meta_json(
    axon_dir: Path,
    repo_path: Path,
    storage: KuzuBackend,
    *,
    no_embeddings: bool,
) -> None:
    if not (axon_dir / "meta.json").exists():
        rprint("[b yellow]Un-initialize repo, running initial index....", file=stderr)
        _process_meta(axon_dir, repo_path, storage, no_embeddings=no_embeddings)
        rprint("[b green]Index complete.")


app = Typer(
    name="axon",
    help="Axon — Graph-powered code intelligence engine.",
    no_args_is_help=True,
)


def _version_callback(*, value: bool) -> None:
    """Print the version and exit."""
    if value:
        rprint(f"Axon v{__version__}")
        raise Exit()


@app.callback()
def main(
    *,
    version: bool | None = Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Axon — Graph-powered code intelligence engine."""


@app.command()
def analyze(
    path: Path = _PATH_ARG,
    *,
    full: bool = Option(_FALSE, "--full", help="Perform a full re-index."),
    no_embeddings: bool = Option(
        _FALSE,
        "--no-embeddings",
        help="Skip vector embedding generation.",
    ),
) -> None:
    """Index a repository into a knowledge graph."""

    repo_path, axon_dir, db_path = _get_path(path)
    rprint(f"[b green]Indexing [b magenta]{repo_path}")
    storage = _get_kuzu(db_path)
    _report(_process_meta(axon_dir, repo_path, storage, no_embeddings=no_embeddings))
    storage.close()


@app.command()
def status() -> None:
    """Show index status for current repository."""
    repo_path = Path.cwd().resolve()
    meta_path = repo_path / ".axon" / "meta.json"

    if not meta_path.exists():
        rprint(f"[red]Error:[/red] No index found at {repo_path}. Run 'axon analyze' first.")
        raise Exit(code=1)

    meta = loads(meta_path.read_text(encoding="utf-8"))
    stats = meta.get("stats", {})

    rprint(f"[bold]Index status for[/bold] {repo_path}")
    rprint(f"  Version:        {meta.get('version', '?')}")
    rprint(f"  Last indexed:   {meta.get('last_indexed_at', '?')}")
    rprint(f"  Files:          {stats.get('files', '?')}")
    rprint(f"  Symbols:        {stats.get('symbols', '?')}")
    rprint(f"  Relationships:  {stats.get('relationships', '?')}")

    if stats.get("clusters", 0) > 0:
        rprint(f"  Clusters:       {stats['clusters']}")
    if stats.get("flows", 0) > 0:
        rprint(f"  Flows:          {stats['flows']}")
    if stats.get("dead_code", 0) > 0:
        rprint(f"  Dead code:      {stats['dead_code']}")
    if stats.get("coupled_pairs", 0) > 0:
        rprint(f"  Coupled pairs:  {stats['coupled_pairs']}")


@app.command(name="list")
def list_repos() -> None:
    """List all indexed repositories."""

    result = _tools.handle_list_repos()
    rprint(result)


@app.command()
def clean(
    *,
    force: bool = Option(
        _FALSE,
        "--force",
        "-f",
        help="Skip confirmation prompt.",
    ),
) -> None:
    """Delete index for current repository."""
    repo_path = Path.cwd().resolve()
    axon_dir = repo_path / ".axon"

    if not axon_dir.exists():
        rprint(f"[red]Error:[/red] No index found at {repo_path}. Nothing to clean.")
        raise Exit(code=1)

    if not force and not confirm(f"Delete index at {axon_dir}?"):
        rprint("Aborted.")
        raise Exit()

    rmtree(axon_dir)
    rprint(f"[green]Deleted[/green] {axon_dir}")


@app.command()
def query(
    q: str = Argument(..., help="Search query for the knowledge graph."),
    limit: int = Option(20, "--limit", "-n", help="Maximum number of results."),
) -> None:
    """Search the knowledge graph."""

    storage = _load_storage()
    result = _tools.handle_query(storage, q, limit=limit)
    rprint(result)
    storage.close()


@app.command()
def context(
    name: str = Argument(..., help="Symbol name to inspect."),
) -> None:
    """Show 360-degree view of a symbol."""

    storage = _load_storage()
    result = _tools.handle_context(storage, name)
    rprint(result)
    storage.close()


@app.command()
def impact(
    target: str = Argument(..., help="Symbol to analyze blast radius for."),
    depth: int = Option(3, "--depth", "-d", min=1, max=10, help="Traversal depth (1-10)."),
) -> None:
    """Show blast radius of changing a symbol."""

    storage = _load_storage()
    result = _tools.handle_impact(storage, target, depth=depth)
    rprint(result)
    storage.close()


@app.command(name="dead-code")
def dead_code() -> None:
    """List all detected dead code."""

    storage = _load_storage()
    result = _tools.handle_dead_code(storage)
    rprint(result)
    storage.close()


@app.command()
def cypher(
    query: str = Argument(..., help="Raw Cypher query to execute."),
) -> None:
    """Execute raw Cypher against the knowledge graph."""

    storage = _load_storage()
    result = _tools.handle_cypher(storage, query)
    rprint(result)
    storage.close()


@app.command()
def setup(
    *,
    claude: bool = Option(
        _FALSE,
        "--claude",
        help="Configure MCP for Claude Code.",
    ),
    cursor: bool = Option(
        _FALSE,
        "--cursor",
        help="Configure MCP for Cursor.",
    ),
) -> None:
    """Configure MCP for Claude Code / Cursor."""
    mcp_config = {
        "command": "axon",
        "args": ["serve", "--watch"],
    }

    if claude or (not claude and not cursor):
        rprint("[bold]Claude Code[/bold]")
        rprint("Add to .mcp.json in your project root:")
        rprint(dumps({"mcpServers": {"axon": mcp_config}}, indent=2))
        rprint("\n[dim]Or run: claude mcp add axon -- axon serve --watch[/dim]")

    if cursor or (not claude and not cursor):
        rprint("[bold]Add to your Cursor MCP config:[/bold]")
        rprint(dumps({"axon": mcp_config}, indent=2))


@app.command()
def watch(
    *,
    no_embeddings: bool = Option(
        _FALSE,
        "--no-embeddings",
        help="Skip vector embedding generation.",
    ),
) -> None:
    """Watch mode — re-index on file changes."""

    repo_path, axon_dir, db_path = _get_path()
    storage = _get_kuzu(db_path)

    _check_meta_json(axon_dir, repo_path, storage, no_embeddings=no_embeddings)

    try:
        deps = WatcherDeps(repo_path, storage)
        rprint(
            f"[b blue]Watching [b magenta]{repo_path}[/b magenta] for changes. [white](Ctrl+C to stop)",
        )
        asyncio_run(Watcher(deps).watch())
    except KeyboardInterrupt:
        rprint("\n[bold]Watch stopped.[/bold]")
    finally:
        storage.close()


@app.command()
def diff(
    branch_range: str = Argument(
        ...,
        help="Branch range for comparison (e.g. main..feature).",
    ),
) -> None:
    """Structural branch comparison."""

    repo_path = Path.cwd().resolve()
    try:
        result = diff_branches(repo_path, branch_range)
    except (ValueError, RuntimeError) as exc:
        rprint(f"[red]Error:[/red] {exc}")
        raise Exit(code=1) from exc

    rprint(format_diff(result))


@app.command()
def mcp() -> None:
    """Start MCP server (stdio transport)."""

    asyncio_run(mcp_main())


async def _run(
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


@app.command()
def serve(
    *,
    watch: bool = Option(
        _FALSE,
        "--watch",
        "-w",
        help="Enable file watching with auto-reindex.",
    ),
    full: bool = Option(_FALSE, "--full", help="Perform a full re-index."),
    no_embeddings: bool = Option(
        _FALSE,
        "--no-embeddings",
        help="Skip vector embedding generation.",
    ),
) -> None:
    """Start MCP server, optionally with live file watching."""

    if not watch:
        asyncio_run(mcp_main())
        return

    repo_path, axon_dir, db_path = _get_path()
    storage = _get_kuzu(db_path)

    _check_meta_json(axon_dir, repo_path, storage, no_embeddings=no_embeddings)

    try:
        rprint(
            f"[b blue]Serving and watching [b magenta]{repo_path}[/b magenta] "
            "for changes. [white](Ctrl+D to stop)",
        )
        asyncio_run(_run(repo_path, storage))
    except KeyboardInterrupt:
        pass
    finally:
        storage.close()


if __name__ == "__main__":
    app()
