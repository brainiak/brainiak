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



How to contribute
=================

We use GitHub pull requests (PRs) to make improvements to the repository. You
should make a fork, create a new branch for each new feature you develop, and
make a PR to merge your branch into the master branch of the official
repository. There are several workflows you could follow. Here is a concise
step-by-step description of our recommended workflow:

1. Fork the official BrainIAK repository on GitHub.

2. Clone your fork::

     git clone https://github.com/yourgithubusername/brainiak

3. Add the official BrainIAK repository as the ``upstream`` remote::

     git remote add upstream https://github.com/brainiak/brainiak

4. Set the ``master`` branch to track the ``upstream`` remote::

     git fetch upstream
     git branch -u upstream/master

5. Whenever there are commits in the official repository, pull them to keep
   your ``master`` branch up to date::

     git pull --ff-only

6. Always create a new branch when you start working on a new feature; we only
   update the ``master`` branch via pull requests from feature branches; never
   commit directly to the ``master`` branch::

     git checkout -b new-feature

7. Make changes and commit them. Include a news fragment for the release notes
   in ``docs/newsfragments`` if your changes are visible to users (see `Pip's
   documentation`_ and our news types in ``pyproject.toml``).

8. Push your feature branch to your fork::

     git push --set-upstream origin new-feature  # only for the first push
     git push  # for all subsequent pushes

9. When your feature is ready, make a pull request on GitHub. After your
   feature is accepted, update your ``master`` branch and delete your feature
   branch::

     git checkout master
     git pull --ff-only
     git branch -d new-feature
     git push --delete origin new-feature  # or use delete button in GitHub PR

Please see the `GitHub help for collaborating on projects using issues and pull
requests`_ for more information.

.. _Pip's documentation:
   https://pip.pypa.io/en/latest/development/#adding-a-news-entry
.. _GitHub help for collaborating on projects using issues and pull requests:
   https://help.github.com/categories/collaborating-on-projects-using-issues-and-pull-requests/

All pull requests are automatically tested using the ``pr-check.sh`` script.
You should test you contributions yourself on your computer using
``pr-check.sh`` before creating a PR. The script performs several checks in a
Python virtual environment, which is isolated from your normal Python
environment for reproducibility.

During development, you may wish to run some of the individual checks in
``pr-check.sh`` repeatedly until you get everything right, without waiting for
the virtual environment to be set up every time. You can run the individual
checks from ``pr-check.sh`` using the steps bellow::

  # do not run this if using Anaconda, because Anaconda is not compatible with
  # venv; instead, look at pr-check.sh to see how to run the individual
  # checks that are part of pr-check.sh using Anaconda

  # optional, but highly recommended: create a virtual environment
  python3 -m venv ../brainiak_pr_venv
  source ../brainiak_pr_venv/bin/activate

  # install brainiak in editable mode and developer dependencies
  python3 -m pip install -U -r requirements-dev.txt

  # static analysis
  ./run-checks.sh

  # run tests
  ./run-tests.sh

  # build documentation
  cd docs
  make
  cd -

  # optional: remove the virtual environment, if you created one
  deactivate
  rm -r ../brainiak_pr_venv

When you are ready to submit your PR, run ``pr-check.sh`` even if you were
using the steps above to run the individual checks in ``pr-check.sh`` during
development.

If you want to obtain early feedback for your work, ask people to look at your
fork. Alternatively, you can open a PR before your work is ready; in this case,
you should start the PR title with ``WIP:``, to let people know your PR is work
in progress.


Tools
=====

We primarily use PyCharm (or equivalently, IDEA with Python plugin). You're free
to use whatever you like to develop, but bear in mind that if you use the same
tools as the rest of the group, more people will be able to help if something
goes wrong.

The development requirements are listed in ``requirements-dev.txt``. You can
install them with::

  python3 -m pip install -U -r requirements-dev.txt


Standards
=========

* Python code should follow the `Scikit-learn coding guidelines`_ with the
  exception that we target Python 3 only.

.. _Scikit-learn coding guidelines:
   http://scikit-learn.org/stable/developers/contributing.html#coding-guidelines

* Python docstrings should be formatted according to the NumPy docstring
  standard as implemented by the `Sphinx Napoleon extension`_ (see also the
  `Sphinx NumPy example`_). In particular, note that type annotations must
  follow `PEP 484`_. Please also read the `NumPy documentation guide`_, but
  note that we consider Sphinx authoritative.

.. _Sphinx Napoleon extension:
   http://www.sphinx-doc.org/en/stable/ext/napoleon.html
.. _Sphinx NumPy example:
   http://www.sphinx-doc.org/en/stable/ext/example_numpy.html
.. _PEP 484:
   https://www.python.org/dev/peps/pep-0484/
.. _NumPy documentation guide:
   https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

* C++ code should follow the `WebKit code style guidelines`_.

.. _WebKit code style guidelines:
   https://google.github.io/styleguide/cppguide.html

* All code exposed through public APIs must have documentation that explains
  what the code does, what its parameters mean, and what its return values can
  be, at a minimum.

* All code must have repeatable automated unit tests, and most code should
  have integration tests as well.

* Where possible, transformations and classifiers should be made compatible
  with Scikit-learn Pipelines by implementing ``fit``, ``transform`` and 
  ``fit_transform`` methods as described in the `Scikit-learn pipeline
  documentation`_.

.. _Scikit-learn pipeline documentation:
   http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

* All code using random numbers should allow reproducible execution using the
  `Scikit-learn random numbers guidelines`_.

.. _Scikit-learn random numbers guidelines:
   http://scikit-learn.org/stable/developers/contributing.html#random-numbers

* Use ``logging`` to record debug messages with a logger obtained using::

    logging.getLogger(__name__)

  Use ``warnings`` to show warning messages to users. Do not use ``print``. See
  the `Python Logging Tutorial`_ for details.

.. _Python Logging Tutorial:
   https://docs.python.org/3/howto/logging.html

* Create usage examples for new modules in the examples directory. Add a
  ``requirements.txt`` file to help users install the packages your examples
  require. If your example requires software that is not available in PyPI,
  document it in a ``README.rst`` and also update the BrainIAK Dockerfile.

* Remove the output of example Jupyter notebooks before committing them, using
  `nbstripout <https://pypi.python.org/pypi/nbstripout>`_.


Testing
=======

Unit tests are small tests that execute very quickly, seconds or less. They are
the first line of defense against software errors, and you must include some
whenever you add code to BrainIAK. We use a tool called "pytest" to run tests;
please read the `Pytest documentation`_.  You should put your tests in a
``test_*.py`` file in the test folder, following the structure of the
``brainiak`` folder. So for example, if you have your code in
``brainiak/funcalign/srm.py`` you should have tests in
``tests/funcalign/test_srm.py``. The unit tests for a subpackage should not
take more than one minute in total on our testing service, `Travis CI`_.

.. _Pytest documentation:
  http://pytest.org/latest/contents.html
.. _Travis CI:
  https://travis-ci.org

You must install the package in editable mode before running the tests::

    python3 -m pip install -e .

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

Folder layout
=============

Since BrainIAK is primarily published as a Python package, it is largely
organized according to the `Python Packaging User Guide`_.

.. _Python Packaging User Guide:
   https://packaging.python.org/distributing/

Python code goes in the ``brainiak`` package, usually with a subpackage for
each major research initiative. If an algorithm can be implemented in a single
module, place the module directly in the ``brainiak`` package, do not create a
subpackage.

Name subpackages and modules using short names describing their functionality,
e.g., ``tda`` for the subpackage containing topological data analysis work and
``htfa.py`` for the module implementing hierarchical topographical factor
analysis.


Making a release
================

This information is only of interest to the core contributors who have the
right to make releases.

1. Choose a release number, ``v``. We follow `Semantic Versioning
   <http://semver.org>`_, although we omit the patch number when it is 0.

2. Prepare the release notes::

       git checkout -b release-v<v>
       towncrier --version <v>
       ./pr-check.sh
       git commit -a -m "Add release notes for v<v>"
       git push --set-upstream origin release-v<v>
       <Create a PR; merge the PR.>
       git checkout master
       git pull --ff-only
       git branch -d release-v<v>

3. Tag the release::

       git tag v<v>

4. Create and test the distribution package::

       python3 setup.py sdist
       tar -xf dist/brainiak-<v>.tar.gz
       cd brainiak-<v>
       ./pr-check.sh
       cd -
       rm -r brainiak-<v>

5. Push release::

       git tag --push
       twine upload dist/brainiak-<v>.tar.gz
