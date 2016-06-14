Goals
=====

We're building a Python toolkit to support neuroscience. This effort is
complementary to others in this space, such as nilearn, nipy, and similar. We
are primarily working on new research efforts in processing and understanding
functional Magnetic Resonance Imaging (fMRI) data sets.

We provide high performance algorithms, typically implemented in C/C++, along
with convenient Python wrapper modules that make these advanced tools available
for easy use. These are some of the design goals that contributors should keep
in mind:

* We do not intend to duplicate existing functionality, either in the C++ side,
  or the Python side. For example, we do not provide any tools for parsing Nifti
  files, even though BrainIAK heavily depends on them. Nibabel already has
  perfectly good tools for this.

* We try to make the C++ libraries usable outside of Python as well, so that
  they could be used directly in your C++ project, or accessed from other
  languages. However this is a secondary goal, our primary goal is to produce a
  solid, highly usable, very high performance toolkit for Python.

* Every algorithm should be capable of running on a single machine, and if there
  is an appropriate distributed algorithm, it should also be capable of running
  at cluster scale. It is understood that the single-machine version of an
  algorithm will need to work with smaller datasets than the cluster version.



How to Contribute
=================

We use Pull Requests (PR's)
(https://help.github.com/categories/collaborating-on-projects-using-pull-requests/)
to make improvements to the repository. Please see the linked documentation for
information about how to create your own fork of the project, and generate pull
requests to submit your code for inclusion in the project. All pull requests
will be automatically tested as described below, so make sure you test them yourself before submitting them:

  # optional, but highly recommended: create a virtualenv to isolate tests

  virtualenv ../brainiak_pr_venv

  source ../brainiak_pr_venv/bin/activate


  # install developer dependencies

  pip install -r requirements-dev.txt


  # static analysis

  ./run-checks.sh


  # install brainiak in editable mode (required for testing)

  pip install -e .


  # run tests

  ./run-tests.sh


  # build documentation

  cd docs

  make

  cd -


  # optional: remove virtualenv

  deactivate

  rm -r ../brainiak_pr_venv



Supported Configurations
========================

BrainIAK provides greatest performance benefits when compiled with the Intel
C/C++ compiler, icc, though it will compile with both gcc and icc.

We currently support Linux and MacOS X.

The Intel Math Kernel Library (MKL) is required, as is MPI. We use MPICH for
compiling. The Intel Data Analytics Acceleration Library (DAAL) is also
required.

* MKL and DAAL are both available to everyone under free community license at
  this URL:
  https://registrationcenter.intel.com/en/forms/?productid=2558&licensetype=2
* The Intel C/C++ compiler is available for free to open source contributors
  here:
  https://registrationcenter.intel.com/en/forms/?licensetype=2&programID=opensource&productid=2302

Tools
=====

We primarily use PyCharm (or equivalently, IDEA with Python plugin). You're free
to use whatever you like to develop, but bear in mind that if you use the same
tools as the rest of the group, more people will be able to help if something
goes wrong.

The development requirements are listed in ``requirements-dev.txt``. You can install them with::

  pip install -r requirements-dev.txt

Standards
=========

* Python code should follow the SciKit-Learn coding standards
  (http://scikit-learn.org/stable/developers/contributing.html#coding-guidelines)
  with the exception that we target Python 3 only.
* Python docstrings should be formatted according to the NumPy docstring
  standard (https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt)
* C++ code should follow the Google C++ standards
  (https://google.github.io/styleguide/cppguide.html)
* All user-visible / public APIs should have technical documentation that
  explains what the code does, what its parameters mean, and what its return
  values can be, at a minimum. Run ``make`` in the ``docs`` directory to make
  sure you can build the documentation; this requires ``sphinx_rtd_theme``.
* All code should have repeatable automated unit tests, and most code should
  have integration tests as well.
* Where possible, transformations and classifiers should be made compatible
  with Scikit-learn Pipelines by implementing fit, transform and 
  fit_transform methods as described in the Scikit-learn documentation
  (http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
* Use ``run-checks.sh`` to check your code before submitting a pull request. It
  requires ``flake8``.

Testing
=======

Unit tests are small tests that execute very quickly, seconds or less. They
are the first line of defense against software errors, and you must
include some whenever you add code to BrainIAK. We use a tool called "pytest"
to run tests; look at http://pytest.org/latest/contents.html for its
documentation; you must also install ``pytest-cov``. You should put your tests
in a ``test_*.py`` file in the test folder, following the structure of the
``brainiak` folder. So for example, if you have your code in
``brainiak/functional_alignment/srm.py`` you should have tests in
``tests/functional_alignment/test_srm.py``.

You must install the package in editable mode using the ``-e`` flag of ``pip
install`` before running the tests.

You can run ``./run-tests.sh`` to run all the unit tests, or you can use the
``py.test <your-test-file.py>`` command to run your tests only, at a more
granular level.

Next to the test results, you will also see a code coverage report. New code
should have at least 90% coverage.

Note that you can only obtain test coverage data when the package is installed
in editable mode or the test command is called from the ``test`` directory. If
the package is installed normally and the test command is called from the
project root directory, the coverage program will fail to report the coverage
of the installed code, because it will look for the code in the current
directory, which is not executed.

Folder Layout
=============

Since BrainIAK is primarily published as a Python package, it is largely
organized according to the guidelines for Python package distribution:
http://python-packaging-user-guide.readthedocs.org/en/latest/distributing/

Python code goes in the ``brainiak`` folder, usually with a subfolder for each
major research initiative or algorithm.

Try to give subpackages a short, but still-as-meaningful-as-possible name.

For example, ``brainiak/topofactor`` might be a name for the folder for
topological factor analysis work.
