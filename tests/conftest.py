import multiprocessing

from mpi4py import MPI
import pytest
import numpy
import random
import tensorflow


def pytest_configure(config):
    config.option.xmlpath = "junit-{}.xml".format(MPI.COMM_WORLD.Get_rank())


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
