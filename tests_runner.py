from pathlib import Path

from pytest import TestReport
from pytest import main as pytest_main
from rich import print as rprint


class FailedTestPlugin:
    def __init__(self) -> None:
        self.failed_tests = []

    def pytest_runtest_logreport(self, report: TestReport) -> None:
        if report.when == "call" and report.failed:
            self.failed_tests.append(report.nodeid)


def run_tests(test_files: list[str]) -> list[str]:
    args = ["-p", "no:warnings"] + test_files
    plugin = FailedTestPlugin()
    pytest_main(args, plugins=[plugin])
    return plugin.failed_tests


def main() -> None:
    _stuck_tests = [
        "tests/core/test_kuzu_search.py::TestEmbeddingsAndVectorSearch::test_store_and_retrieve_by_vector",
        "tests/core/test_kuzu_search.py::TestEmbeddingsAndVectorSearch::test_vector_search_ranking",
        "tests/core/test_kuzu_search.py::TestEmbeddingsAndVectorSearch::test_vector_search_limit",
        "tests/core/test_kuzu_search.py::TestEmbeddingsAndVectorSearch::test_store_embeddings_upsert",
    ]
    tests_dir = Path.cwd().resolve() / "tests"
    # long running tests
    excluded = {
        # "test_kuzu_backend.py", # Done
        # "test_kuzu_search.py",  # Done
        # "test_watcher.py", # Done
        # "test_pipeline.py", # Done
        "test_full_pipeline.py",  # Done
    }
    test_files = [str(test) for test in tests_dir.rglob("test_*.py") if test.name in excluded]

    if failed_tests := run_tests(test_files):
        rprint(f"[red]Failed tests:[/red] {failed_tests}")
    else:
        rprint("[green]All tests passed[/green]")


if __name__ == "__main__":
    main()
