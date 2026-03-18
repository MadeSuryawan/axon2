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
    tests_dir = Path.cwd().resolve() / "tests"
    # a set of long running tests
    excluded = {
        # "test_kuzu_backend.py",
        "test_kuzu_search.py",
        # "test_watcher.py",
        # "test_pipeline.py",
        # "test_full_pipeline.py",
    }
    test_files = [str(test) for test in tests_dir.rglob("test_*.py") if test.name in excluded]

    if failed_tests := run_tests(test_files):
        rprint(f"[red]Failed tests:[/red] {failed_tests}")
    else:
        rprint("[green]All tests passed[/green]")


if __name__ == "__main__":
    main()
