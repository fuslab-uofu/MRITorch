"""pytest configuration: tests marked ``gpu`` / ``matlab`` are deselected unless ``--run-gpu`` / ``--run-matlab`` is given."""
import pytest


def pytest_addoption(parser):
    parser.addoption("--run-gpu", action="store_true", default=False,
                     help="run tests marked gpu (they are skipped if CUDA is not available)")
    parser.addoption("--run-matlab", action="store_true", default=False,
                     help="run tests marked matlab (they are skipped if matlab is not on PATH)")


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA device; deselected unless --run-gpu is given")
    config.addinivalue_line("markers", "matlab: needs MATLAB; deselected unless --run-matlab is given")


def pytest_collection_modifyitems(config, items):
    skip = [m for m, opt in (("gpu", "--run-gpu"), ("matlab", "--run-matlab")) if not config.getoption(opt)]
    deselected = [item for item in items if any(item.get_closest_marker(m) for m in skip)]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = [item for item in items if item not in deselected]
