"""Pytest configuration for arxiv-latex-reader tests."""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False,
        help="Run slow tests that require LLM API calls",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (require LLM API)")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="Need --run-slow to run LLM tests")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
