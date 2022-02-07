
import multiprocessing

from mpi4py import MPI
import pytest
import numpy
import random
from brainiak.fcma.classifier import Classifier
import tensorflow


def pytest_configure(config):
    config.option.xmlpath = "junit-{}.xml".format(MPI.COMM_WORLD.Get_rank())


def pytest_addoption(parser):
    parser.addoption('--enable_notebook_tests', action='store_true', dest="enable_notebook_tests",
                      default=False, help="Enable tests for Jupyter notebook examples in docs/examples. "
                                          "These can take a long time and are disabled by default.")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--enable_notebook_tests"):
        # --enable_notebook_tests given in cli: do not skip notebook tests
        enable_notebook_tests = pytest.mark.skip(reason="needs --enable_notebook_tests option to run")
        for item in items:
            if "notebook" in item.keywords:
                item.add_marker(enable_notebook_tests)
    else:
        return

@pytest.fixture
def seeded_rng():
    random.seed(0)
    numpy.random.seed(0)
    tensorflow.random.set_seed(0)


skip_non_fork = pytest.mark.skipif(
    multiprocessing.get_start_method() != "fork"
    and MPI.COMM_WORLD.Get_attr(MPI.APPNUM) is not None,
    reason="MPI only works with multiprocessing fork start method.",
)
