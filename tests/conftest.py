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
