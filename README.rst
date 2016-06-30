READ THIS FIRST
===============

For directions about how to contribute to the Brain Imaging Analysis Kit,
please see CONTRIBUTING.rst in the repository
(also visible nicely formatted at
https://github.com/IntelPNI/brainiak.rst)

The material below is directed at end users. This "READ THIS FIRST" section will be removed before our first release.

Brain Imaging Analysis Kit
==========================

The Brain Imaging Analysis Kit is a package of Python modules useful for neuroscience, primarily focused on
functional Magnetic Resonance Imaging (fMRI) analysis.

The package was originally created by a collaboration between Intel and the Princeton Neuroscience Institute (PNI).

To reduce verbosity, we may refer to the Brain Imaging Analysis Kit using the ``BrainIAK`` abbreviation. Whenever lowercase spelling is used (e.g., Python package name), we use ``brainiak``.

Requirements
============

We support Linux and MacOS with Python version 3.4 or higher. Most of the
dependencies will be installed automatically. However, a few need to be
installed manually.

Linux
-----

Install the following packages (Ubuntu 14.04 is used for the examples)::

    apt install build-essential cmake libgomp1 mpich python3-pip

Install up-to-date versions of ``pip`` and ``virtualenv``::

    # note that the command installed by apt is pip3, not pip
    pip3 install --user -U pip virtualenv

Note the ``--user`` flag, which instructs ``pip`` to not overwrite system
files. You must add ``$HOME/.local/bin`` to your ``$PATH`` to be able to run
the updated ``pip``, e.g., by adding the following line to ``$HOME/.profile``
and launching a new login shell (e.g., logout or execute ``bash -l``)::

    PATH="$HOME/.local/bin:$PATH"

MacOS
-----

Install the Xcode Command Line Tools::

    xcode-select --install

Install ``brew`` from https://brew.sh. Then install the following::

    brew install clang-omp cmake mpich python3

You must instruct programs to use ``clang-omp``. One way to do this, which
works for most programs, is setting the ``CC`` environment variable. You can
add the following lines to ``$HOME/.profile`` (for them to take effect, you
must logout or launch a new login shell, e.g., ``bash -l``)::

    CC=clang-omp
    CXX=clang-omp++

Install up-to-date versions of ``pip`` and ``virtualenv``::

    pip install -U pip virtualenv

Install
=======

In the future, the Brain Imaging Analysis Kit will be available on PyPI. For the moment, it must be installed from a Git repository.

Install directly from GitHub
----------------------------

To install directly from GitHub, do:

    pip install git+https://github.com/intelpni/brainiak.git

Or, if you have ssh keys installed, you can do:

    pip install git+ssh://github.com/intelpni/brainiak.git

Install from local clone
------------------------

If you prefer to install from a local clone of the repository, follow these
steps:

    git clone https://github.com/intelpni/brainiak

    cd brainiak

    pip install .

    ..
       To install via `pip`, execute the following at a command prompt::
       TODO
       pip install -U --user brainiak


Building documentation
----------------------

If desired, you can build the documentation yourself. Currently we don't have the docs hosted online, so this is a necessary step at the moment. Once the Brain Imaging Analysis Kit is made public, we'll host documentation at readthedocs.org, so most users will not need to build docs.

To build the documentation, you must have Sphinx installed (you may already have it, even if you've never heard of it). If you need to install, please follow directions from http://www.sphinx-doc.org/en/stable/install.html

Once you have sphinx installed, you can do (assuming you're already in the brainiak folder)

    cd docs

    make html

This will generate html documentation in the _build/html folder within the docs folder. _build/html/index.html is the starting page.


Links
=====

- Official source code repo: https://github.com/IntelPNI/brainiak
- HTML documentation (stable release): TODO (will be hosted on readthedocs.org once repository is made public)



Contribute
==========

Please read the contributor's guide at
https://github.com/IntelPNI/brainiak/blob/master/CONTRIBUTING.rst
