"""Axon CLI — Graph-powered code intelligence engine."""

from asyncio import Event, gather
from asyncio import run as asyncio_run
from contextlib import suppress
from json import dumps, loads
from logging import ERROR, WARNING, basicConfig, getLogger
from pathlib import Path
from shutil import rmtree
from sys import platform
from threading import Timer

from rich import print as rprint
from rich.logging import RichHandler
from rich.traceback import install
from typer import Argument, Context, Exit, Option, Typer, confirm
from uvicorn import Config, Server
from uvicorn import run as uvicorn_run

from axon.cli.helpers.db_check import (
    _FALSE,
    check_meta_json,
    get_kuzu,
    get_path,
    load_storage,
    process_meta,
    report,
    version_callback,
)
from axon.cli.helpers.host import (
    DEFAULT_HOST,
    DEFAULT_MANAGED_PORT,
    HostConfig,
    create_host_lease,
    ensure_host_running,
    get_live_host_info,
    maybe_notify_update,
    proxy_stdio_to_http_mcp,
    remove_host_lease,
    run_shared_host,
    web_open,
)
from axon.core.diff import diff_branches, format_diff
from axon.core.ingestion.watcher import Watcher, WatcherDeps
from axon.mcp.server import main as mcp_main
from axon.mcp.tools import MCPTools
from axon.web.app import create_app, create_ui_proxy_app

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

mcp_tools = MCPTools()

_PATH_ARG = Argument(Path("."), help="Path to the repository to index.")


@app.callback()
def main(
    ctx: Context,
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
    maybe_notify_update(ctx.invoked_subcommand)


@app.command()
def analyze(
    path: Path = _PATH_ARG,
    *,
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
        rprint(
            "[red]Error: No index found.[/red] Run [cyan]axon analyze .[/cyan] first to index this codebase.",
        )
        raise Exit(code=1)

    meta = loads(meta_path.read_text(encoding="utf-8"))
    stats = meta.get("stats", {})

    rprint(f"[bold]Index status for {repo_path}")
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

    rprint(mcp_tools.handle_list_repos())


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
    repo_path, axon_dir, _ = get_path()

    if not axon_dir.exists():
        rprint(f"[red]Error: No index found at {repo_path}. Nothing to clean.")
        raise Exit(code=1)

    if not force and not confirm(f"Delete index at {axon_dir}?"):
        rprint("Aborted.")
        raise Exit()

    rmtree(axon_dir)
    rprint(f"[green]Deleted {axon_dir}")


@app.command()
def query(
    q: str = Argument(..., help="Search query for the knowledge graph."),
    limit: int = Option(20, "--limit", "-n", help="Maximum number of results."),
) -> None:
    """Search the knowledge graph."""

    storage = load_storage()
    rprint(mcp_tools.handle_query(storage, q, limit=limit))
    storage.close()


@app.command()
def context(
    name: str = Argument(..., help="Symbol name to inspect."),
) -> None:
    """Show 360-degree view of a symbol."""

    storage = load_storage()
    rprint(mcp_tools.handle_context(storage, name))
    storage.close()


@app.command()
def impact(
    target: str = Argument(..., help="Symbol to analyze blast radius for."),
    depth: int = Option(3, "--depth", "-d", min=1, max=10, help="Traversal depth (1-10)."),
) -> None:
    """Show blast radius of changing a symbol."""

    storage = load_storage()
    rprint(mcp_tools.handle_impact(storage, target, depth=depth))
    storage.close()


@app.command(name="dead-code")
def dead_code() -> None:
    """List all detected dead code."""

    storage = load_storage()
    rprint(mcp_tools.handle_dead_code(storage))
    storage.close()


@app.command()
def cypher(
    query: str = Argument(..., help="Raw Cypher query to execute."),
) -> None:
    """Execute raw Cypher against the knowledge graph."""

    storage = load_storage()
    rprint(mcp_tools.handle_cypher(storage, query))
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
    stdio_config = {
        "command": "axon",
        "args": ["serve", "--watch"],
    }

    if claude or (not claude and not cursor):
        rprint("[bold]Claude Code")
        rprint("Add to .mcp.json in your project root:")
        rprint(dumps({"mcpServers": {"axon": stdio_config}}, indent=2))
        rprint("\n[dim]Or run: claude mcp add axon -- axon serve --watch")

    if cursor or (not claude and not cursor):
        rprint("[bold]Add to your Cursor MCP config:")
        rprint(dumps({"axon": stdio_config}, indent=2))


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
        rprint("\n[bold]Watch stopped.")
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
        rprint(format_diff(diff_branches(repo_path, branch_range)))
    except (ValueError, RuntimeError) as exc:
        rprint(f"[red]Error: {exc}")
        raise Exit(code=1) from exc


@app.command()
def mcp() -> None:
    """Start MCP server (stdio transport)."""

    asyncio_run(mcp_main())


@app.command()
def serve(
    *,
    watch: bool = Option(_FALSE, "--watch", "-w", help="Enable file watching with auto-reindex."),
) -> None:
    """Start MCP server, optionally with live file watching."""
    if not watch:
        asyncio_run(mcp_main())
        return

    repo_path = Path.cwd().resolve()
    lease_path: Path | None = None
    try:
        live_host = ensure_host_running(
            repo_path,
            HostConfig(port=DEFAULT_MANAGED_PORT, watch=True, managed=True),
        )
        lease_path = create_host_lease(repo_path, "mcp")
    except RuntimeError as exc:
        rprint(f"[red]Error: {exc}")
        raise Exit(code=1) from exc

    try:
        asyncio_run(proxy_stdio_to_http_mcp(live_host["mcp_url"]))
    finally:
        remove_host_lease(lease_path)


@app.command()
def ui(
    port: int = Option(8420, "--port", "-p", help="Port to serve on."),
    *,
    no_open: bool = Option(_FALSE, "--no-open", help="Don't auto-open browser."),
    watch_files: bool = Option(_FALSE, "--watch", "-w", help="Enable live file watching."),
    dev: bool = Option(_FALSE, "--dev", help="Proxy to Vite dev server for HMR."),
    direct: bool = Option(
        _FALSE,
        "--direct",
        help="Force standalone UI mode even if a shared Axon host is already running.",
    ),
) -> None:
    """Launch the Axon web UI."""
    repo_path = Path.cwd().resolve()
    if not direct:
        live_host = get_live_host_info(repo_path)
        if live_host is not None:
            if live_host.get("ui_enabled", True):
                rprint(f"[b green]Axon UI available at {live_host['host_url']}")
                if not no_open:
                    web_open(live_host["host_url"])
                return

            proxy_app = create_ui_proxy_app(live_host["host_url"], dev=dev)
            rprint(f"[b green]Axon UI running at http://{DEFAULT_HOST}:{port}")
            if not no_open:
                web_open(f"http://{DEFAULT_HOST}:{port}")
            uvicorn_run(proxy_app, host=DEFAULT_HOST, port=port, log_level="warning")
            return

        run_shared_host(
            port=port,
            bind=DEFAULT_HOST,
            no_open=no_open,
            watch=watch_files,
            dev=dev,
            managed=_FALSE,
            open_browser=True,
            announce_ui=True,
            announce_mcp=_FALSE,
            expose_ui=True,
            already_running_message="[b green]Axon UI available at {url}",
            auto_index=_FALSE,
        )
        return

    db_path = repo_path / ".axon" / "kuzu"
    if not db_path.exists():
        rprint(
            "[red]Error: No index found. Run [cyan]axon analyze . first to index this codebase.",
        )
        raise Exit(code=1)

    web_app = create_app(
        db_path=db_path,
        repo_path=repo_path,
        watch=watch_files,
        dev=dev,
    )

    if not no_open:
        url = f"http://localhost:{port}"
        Timer(1.5, lambda: web_open(url)).start()

    rprint(f"[b green]Axon UI running at http://localhost:{port}")
    if watch_files:
        rprint("[dim]File watching enabled — graph updates on save")
    if dev:
        rprint("[dim]Dev mode — proxying to Vite on :5173")

    if watch_files:

        async def _run() -> None:
            config = Config(web_app, host="127.0.0.1", port=port, log_level="warning")
            server = Server(config)
            stop = Event()

            async def _serve() -> None:
                await server.serve()
                stop.set()

            await gather(
                _serve(),
                Watcher(
                    WatcherDeps(repo_path, web_app.state.storage, stop_event=stop),
                ).watch_repo(),
            )

        try:
            asyncio_run(_run())
        except KeyboardInterrupt:
            rprint("\n[b]UI stopped.")
    else:
        uvicorn_run(web_app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    app()
