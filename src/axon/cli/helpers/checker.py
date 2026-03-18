from contextlib import suppress
from datetime import UTC, datetime
from hashlib import sha256
from json import JSONDecodeError, dumps, loads
from logging import getLogger
from pathlib import Path
from shutil import rmtree
from sys import stderr
from threading import Event, Thread
from typing import Any

from rich import print as rprint
from rich.prompt import Prompt
from typer import Exit, Option

from axon import __version__
from axon.config.constants import SYSTEM_EXCEPTIONS
from axon.config.model_config import (
    DEFAULT_MODEL,
    LARGE_MODEL,
    MODEL_CHOICES,
    MODEL_OPTIONS,
    set_model_name,
)
from axon.core.ingestion.pipeline import PipelineResult, Pipelines
from axon.core.storage.kuzu_backend import KuzuBackend

logger = getLogger(__name__)


# Boolean defaults to avoid FBT003 errors
_FALSE = False
_TRUE = True


def version_callback(*, value: bool) -> None:
    """Print the version and exit."""
    if value:
        rprint(f"Axon v{__version__}")
        raise Exit()


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


def _run_thread() -> dict[str, Any]:
    """Run a thread to read input with timeout."""

    result: dict[str, Any] = {"value": None, "ready": Event()}

    def _read_input() -> None:
        try:
            result["value"] = Prompt.ask(
                "[bold blue]Enter your choice[/bold blue]",
                default="small",
                choices=MODEL_CHOICES,
                show_choices=False,
            )
        except (KeyboardInterrupt, EOFError):
            result["value"] = None
        finally:
            _event: Event = result["ready"]
            _event.set()

    # Start input thread
    input_thread = Thread(target=_read_input, daemon=True)
    input_thread.start()

    return result


def _choose_model(repo_path: Path, *, _timeout_seconds: float = 10.0) -> str:
    """
    Interactive model selection with validation and helpful feedback.

    Args:
        repo_path: Path to store the model configuration.
        _timeout_seconds: Timeout in seconds for user input (default: 30).

    Returns:
        The selected model name string.
    """

    # Display available models with descriptions
    rprint("\n[bold cyan]Select an embedding model:[/bold cyan]")
    rprint(f"  [1] [bold]small[/bold] - {DEFAULT_MODEL}")
    rprint("      → Faster inference, smaller memory footprint")
    rprint(f"  [2] [bold]large[/bold] - {LARGE_MODEL}")
    rprint("      → Higher quality embeddings, slower inference")
    rprint(
        f"\n[dim]Default: small (press Enter to accept, times out in {_timeout_seconds:.0f}s)[/dim]\n",
    )

    result = _run_thread()
    # Wait for input or timeout
    if not result["ready"].wait(timeout=_timeout_seconds):
        rprint(f"\n[yellow]⏱ Timeout! Using default model:[/yellow] {DEFAULT_MODEL}\n")
        return DEFAULT_MODEL

    try:
        if not (choice := result["value"]):
            # User pressed Ctrl+C or stdin was closed
            rprint(
                f"\n[yellow]Selection cancelled. Using default model:[/yellow] {DEFAULT_MODEL}\n",
            )
            return DEFAULT_MODEL

        # Map choice to model name
        model_name = MODEL_OPTIONS.get(choice.lower(), DEFAULT_MODEL)

        # Display confirmation
        if choice.lower() in ["small", "s"]:
            rprint(f"[b green]✓[/b green] Selected: [bold]{model_name}[/bold] (fast mode)")
        else:
            rprint(f"[b green]✓[/b green] Selected: [bold]{model_name}[/bold] (quality mode)")

        # Save the selection if repo_path is provided
        if repo_path:
            set_model_name(model_name, repo_path)
            rprint("[dim]Model preference saved to configuration[/dim]\n")

        return model_name

    except SYSTEM_EXCEPTIONS:
        # Handle any unexpected errors gracefully
        rprint(f"[yellow]Using default model: {DEFAULT_MODEL}[/yellow]")
        return DEFAULT_MODEL


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

    if not no_embeddings:
        _choose_model(repo_path)

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
