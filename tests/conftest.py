import multiprocessing
import sys

from mpi4py import MPI

import pytest
import numpy
import random
import tensorflow


pytest_plugins = ["tests.pytest_mpiexec_plugin"]


def pytest_configure(config):
    config.option.xmlpath = "junit-{}.xml".format(MPI.COMM_WORLD.Get_rank())


def pytest_addoption(parser):
    parser.addoption(
        "--enable_notebook_tests",
        action="store_true",
        dest="enable_notebook_tests",
        default=False,
        help="Enable tests for Jupyter notebook examples in docs/examples. "
        "These can take a long time and are disabled by default.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--enable_notebook_tests"):
        # --enable_notebook_tests given in cli: do not skip notebook tests
        return
    else:
        enable_notebook_tests = pytest.mark.skip(
            reason="needs --enable_notebook_tests option to run"
        )
        for item in items:
            if "notebook" in item.keywords:
                item.add_marker(enable_notebook_tests)


@pytest.fixture
def seeded_rng():
    random.seed(0)
    numpy.random.seed(0)
    tensorflow.random.set_seed(0)


@pytest.fixture(scope="module", autouse=True)
def pool_size():
    """
    Set the pool_size to 1 for MPI tests when start_method for multiprocessing
    is not fork.

    This replaces the old skip_non_fork fixture. We don't need to skip these
    tests completely, but we need to ensure that the pool_size is set to 1 so
    they don't launch any multiprocessing pools within the MPI environment.
    On windows, it seems like intel mpi and msmpi both have issues with fork,
    so we need to set the pool_size to 1 there as well.
    """
    if (multiprocessing.get_start_method() != "fork" and
            MPI.COMM_WORLD.Get_attr(MPI.APPNUM) is not None):
        return 1

    # OpenMPI has issues with fork, so we need to set the pool_size to 1
    if "Open MPI" in MPI.get_vendor()[0]:
        return 1

    # On Windows, we need to set the pool_size to 1 for intel mpi and msmpi
    elif sys.platform == "win32":
        return 1

    else:
        return 2
