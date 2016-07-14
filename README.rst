Brain Imaging Analysis Kit
==========================

.. image:: https://badges.gitter.im/IntelPNI/brainiak.svg
   :alt: Join the chat at https://gitter.im/IntelPNI/brainiak
   :target: https://gitter.im/IntelPNI/brainiak?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

The Brain Imaging Analysis Kit is a package of Python modules useful for
neuroscience, primarily focused on functional Magnetic Resonance Imaging (fMRI)
analysis.

The package was originally created by a collaboration between Intel and the
Princeton Neuroscience Institute (PNI).

To reduce verbosity, we may refer to the Brain Imaging Analysis Kit using the
``BrainIAK`` abbreviation. Whenever lowercase spelling is used (e.g., Python
package name), we use ``brainiak``.


Requirements
============

We support Linux and MacOS with Python version 3.4 or higher. Most of the
dependencies will be installed automatically. However, a few need to be
installed manually.


Linux
-----

Install the following packages (Ubuntu 14.04 is used for the examples)::

    apt install build-essential cmake libgomp1 mpich python3-pip

Install updated version of the following Python packages::

    pip3 install --user -U pip virtualenv

Note the ``--user`` flag, which instructs Pip to not overwrite system
files. You must add ``$HOME/.local/bin`` to your ``$PATH`` to be able to run
the updated Pip, e.g., by adding the following line to ``$HOME/.profile``
and launching a new login shell (e.g., logout or execute ``bash -l``)::

    PATH="$HOME/.local/bin:$PATH"


MacOS
-----

Install the Xcode Command Line Tools::

    xcode-select --install

Install ``brew`` from https://brew.sh. If you already have ``brew``, examine
the output of the following command to make sure it is working::

    brew doctor

Then install the following::

    brew install clang-omp cmake mpich python3

You must instruct programs to use ``clang-omp``. One way to do this, which
works for most programs, is setting the ``CC`` environment variable. You can
add the following lines to ``$HOME/.profile`` (or ``$HOME/.bash_profile``, if
you have one). For them to take effect, you must logout or launch a new login
shell, e.g., ``bash -l``::

    export CC=clang-omp
    export CXX=clang-omp++

Install updated versions of the following Python packages::

    pip3 install -U pip virtualenv


Install
=======

In the future, the Brain Imaging Analysis Kit will be available on PyPI. For
the moment, it must be installed from a Git repository.


Install directly from GitHub
----------------------------

To install directly from GitHub, do::

    pip3 install git+https://github.com/intelpni/brainiak.git

Or, if you have ssh keys installed, you can do::

    pip3 install git+ssh://github.com/intelpni/brainiak.git


Install from local clone
------------------------

If you prefer to install from a local clone of the repository, follow these
steps::

    git clone https://github.com/intelpni/brainiak
    cd brainiak
    pip install .


Documentation
-------------

The documentation is available at::

    https://pythonhosted.org/brainiak


Contribute
==========

We welcome contributions. Please read the guide in `CONTRIBUTING.rst`_.

.. _CONTRIBUTING.rst:
   https://github.com/IntelPNI/brainiak/blob/master/CONTRIBUTING.rst
