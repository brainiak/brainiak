from mpi4py import MPI


def pytest_configure(config):
    config.option.xmlpath = "junit-{}.xml".format(MPI.COMM_WORLD.Get_rank())
