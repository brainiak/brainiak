import os
import glob
import pytest

from testbook import testbook


nb_files = glob.glob("docs/examples/**/*.ipynb", recursive=True)

mpi_notebooks = ["htfa", "FCMA", "SRM"]

nb_tests = []
for f in nb_files:
    # Mark notebooks that need MPI to skip for now,
    # we are having some issues on della
    if any([nb in f for nb in mpi_notebooks]):
        nb_tests.append(
            pytest.param(
                f,
                marks=pytest.mark.skip(
                    "notebooks that require MPI are WIP on della"
                ),
            )
        )
    elif "rtcloud" in f:
        nb_tests.append(
            pytest.param(
                f, marks=pytest.mark.skip("rtcloud is failing on della")
            )
        )
    elif "Matrix-normal" in f:
        nb_tests.append(
            pytest.param(
                f,
                marks=pytest.mark.skip(
                    "Matrix-normal notebook is flaky, disabled for now"
                ),
            )
        )
    else:
        nb_tests.append(f)  # type: ignore


# Helper function to mark specific notebooks as expected failure.
def mark_xfail(nb, **kwargs):
    nb_index = None
    for i, nb_file in enumerate(nb_files):
        if nb in nb_file:
            nb_index = i

    if nb_index is None:
        raise ValueError(
            f"Cannot set notebook {nb} to xfail because it could not be found"
        )

    nb_files[nb_index] = pytest.param(nb, marks=pytest.mark.xfail(**kwargs))


# mark_xfail('rtcloud_notebook.ipynb',
#           reason="Needs to have a web server installed, "
#                  "will probably need to run this in "
#                  "singularity on della")


@pytest.fixture(autouse=True)
def chdir_back_to_root():
    """
    This fixture sets up and tears down state before each example is run.
    Certain examples require that they are run from the local directory in
    which they reside. This changes directory. It reverses this after the
    test finishes.
    """

    # Get the current directory before running the test
    cwd = os.getcwd()

    yield

    # After the test, we need chdir back to root of the repo
    os.chdir(cwd)


@pytest.mark.notebook
@pytest.mark.parametrize("notebook_file", nb_tests)
def test_notebook(notebook_file):

    os.chdir(os.path.dirname(notebook_file))

    with testbook(os.path.basename(notebook_file), execute=True, timeout=3600):
        pass
