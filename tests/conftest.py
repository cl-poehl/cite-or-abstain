"""Shared pytest fixtures and options."""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--update-goldens",
        action="store_true",
        default=False,
        help="Rewrite golden files instead of comparing against them.",
    )


@pytest.fixture
def update_goldens(request) -> bool:
    return request.config.getoption("--update-goldens")
