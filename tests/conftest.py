"""pytest configuration: tests marked ``gpu`` are deselected unless ``--run-gpu`` is given."""
import pytest


def pytest_addoption(parser):
    parser.addoption("--run-gpu", action="store_true", default=False,
                     help="run tests marked gpu (they are skipped if CUDA is not available)")


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA device; deselected unless --run-gpu is given")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-gpu"):
        return
    deselected = [item for item in items if item.get_closest_marker("gpu")]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = [item for item in items if not item.get_closest_marker("gpu")]
