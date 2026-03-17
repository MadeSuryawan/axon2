"""Axon CLI — Graph-powered code intelligence engine."""

from asyncio import run as asyncio_run
from contextlib import suppress
from json import dumps, loads
from logging import ERROR, WARNING, basicConfig, getLogger
from pathlib import Path
from shutil import rmtree
from sys import platform

from rich import print as rprint
from rich.logging import RichHandler
from rich.traceback import install
from typer import Argument, Exit, Option, Typer, confirm

from axon.cli.helpers.db_check import (
    _FALSE,
    check_meta_json,
    get_kuzu,
    get_path,
    load_storage,
    process_meta,
    report,
    start_serve,
    version_callback,
)
from axon.core.diff import diff_branches, format_diff
from axon.core.ingestion.watcher import Watcher, WatcherDeps
from axon.mcp.server import main as mcp_main
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
getLogger("httpcore").setLevel(ERROR)
getLogger("httpx").setLevel(ERROR)
getLogger("fastembed").setLevel(WARNING)
getLogger("filelock").setLevel(WARNING)
getLogger("pygments").setLevel(WARNING)
install()


app = Typer(
    name="axon",
    help="Axon — Graph-powered code intelligence engine.",
    no_args_is_help=True,
)

_tools = Tools()

_PATH_ARG = Argument(Path("."), help="Path to the repository to index.")


@app.callback()
def main(
    *,
    version: bool | None = Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
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

    repo_path, axon_dir, db_path = get_path(path)
    rprint(f"[b green]Indexing [b magenta]{repo_path}")
    storage = get_kuzu(db_path)
    report(process_meta(axon_dir, repo_path, storage, no_embeddings=no_embeddings))
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

    storage = load_storage()
    result = _tools.handle_query(storage, q, limit=limit)
    rprint(result)
    storage.close()


@app.command()
def context(
    name: str = Argument(..., help="Symbol name to inspect."),
) -> None:
    """Show 360-degree view of a symbol."""

    storage = load_storage()
    result = _tools.handle_context(storage, name)
    rprint(result)
    storage.close()


@app.command()
def impact(
    target: str = Argument(..., help="Symbol to analyze blast radius for."),
    depth: int = Option(3, "--depth", "-d", min=1, max=10, help="Traversal depth (1-10)."),
) -> None:
    """Show blast radius of changing a symbol."""

    storage = load_storage()
    result = _tools.handle_impact(storage, target, depth=depth)
    rprint(result)
    storage.close()


@app.command(name="dead-code")
def dead_code() -> None:
    """List all detected dead code."""

    storage = load_storage()
    result = _tools.handle_dead_code(storage)
    rprint(result)
    storage.close()


@app.command()
def cypher(
    query: str = Argument(..., help="Raw Cypher query to execute."),
) -> None:
    """Execute raw Cypher against the knowledge graph."""

    storage = load_storage()
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

    repo_path, axon_dir, db_path = get_path()
    storage = get_kuzu(db_path)

    check_meta_json(axon_dir, repo_path, storage, no_embeddings=no_embeddings)

    try:
        asyncio_run(Watcher(WatcherDeps(repo_path, storage)).watch_repo())
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

    repo_path, axon_dir, db_path = get_path()
    storage = get_kuzu(db_path)

    check_meta_json(axon_dir, repo_path, storage, no_embeddings=no_embeddings)

    try:
        rprint(
            f"[b blue]Serving and watching [b magenta]{repo_path}[/b magenta] "
            "for changes. [white](Ctrl+D to stop)",
        )
        asyncio_run(start_serve(repo_path, storage))
    except KeyboardInterrupt:
        pass
    finally:
        storage.close()


if __name__ == "__main__":
    app()
