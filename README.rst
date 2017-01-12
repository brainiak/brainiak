Brain Imaging Analysis Kit
==========================

.. image:: https://travis-ci.org/IntelPNI/brainiak.svg?branch=master
    :target: https://travis-ci.org/IntelPNI/brainiak

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

Install the following packages (Ubuntu 14.04 is used in these instructions)::

    apt install build-essential libgomp1 libmpich-dev mpich python3-dev \
        python3-pip

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

    brew install llvm cmake mpich python3

You must instruct programs to use this ``clang`` version at ``/usr/local/opt/llvm/bin``.
One way to do this, which
works for most programs, is setting the ``CC`` environment variable. You can
add the following lines to ``$HOME/.profile`` (or ``$HOME/.bash_profile``, if
you have one). For them to take effect, you must logout or launch a new login
shell, e.g., ``bash -l``::

    export CC=/usr/local/opt/llvm/bin/clang
    export CXX=/usr/local/opt/llvm/bin/clang++

In addition, you also need to specify the directories that the newly installed `clang`
will seek for compiling and linking::

    export LDFLAGS="-L/usr/local/opt/llvm/lib $LDFLAGS"
    export CPPFLAGS="-I/usr/local/opt/llvm/include $CPPFLAGS"

Install updated versions of the following Python packages::

    pip3 install -U pip virtualenv


Installing
==========

The Brain Imaging Analysis Kit is available on PyPI::

    pip3 install -U brainiak

Note that you may see a ``Failed building wheel for brainiak`` message (`issue
#61`_). Installation will proceed despite this failure. You can safely ignore it
as long as you see ``Successfully installed`` at the end.

.. _issue #61:
   https://github.com/IntelPNI/brainiak/issues/61

Documentation
=============

The documentation is available at `pythonhosted.org/brainiak`_.

.. _pythonhosted.org/brainiak:
    https://pythonhosted.org/brainiak


Contributing
============

We welcome contributions. Please read the guide in `CONTRIBUTING.rst`_.

.. _CONTRIBUTING.rst:
   https://github.com/IntelPNI/brainiak/blob/master/CONTRIBUTING.rst
