============
Installation
============

Conda
+++++

Make sure you have the ``conda-forge`` channel active, because some of our
dependencies are not available in the default channels.

Use our ``brainiak`` channel to install::

    conda install -c brainiak -c defaults -c conda-forge brainiak

Note that `~brainiak.funcalign.sssrm.SSSRM` currently uses Theano, which
requires the Xcode Command Line Tools on `MacOS`_.

Source
++++++

Requirements
============

We support Linux and MacOS with Python version 3.5 or higher. Most of the
dependencies will be installed automatically. However, a few need to be
installed manually.


Linux
-----

Install the following packages (Ubuntu 16.04 is used in these instructions)::

    apt install build-essential libgomp1 libmpich-dev mpich python3-dev \
        python3-pip python3-venv

Install updated version of the following Python packages::

    python3 -m pip install --user -U pip

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

Note that if your MacOS does not have ``gcc`` (a prerequisite of ``mpich``), 
it will take a long time (perhaps over an hour) to install ``gcc`` from source for you.

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

    export LDFLAGS="-L/usr/local/opt/llvm/lib "\
    "-Wl,-rpath,/usr/local/opt/llvm/lib $LDFLAGS"
    export CPPFLAGS="-I/usr/local/opt/llvm/include $CPPFLAGS"

Install updated versions of the following Python packages::

    python3 -m pip install -U pip


Installing and upgrading
========================

The Brain Imaging Analysis Kit is available on PyPI. You can install it (or
upgrade to the latest version) using the following command::

    python3 -m pip install -U brainiak

.. warning::
    Running `python setup.py install` might fail. It is recommended to install using pip.

Note that you may see a ``Failed building wheel for brainiak`` message (`issue
#61`_). Installation will proceed despite this failure. You can safely ignore it
as long as you see ``Successfully installed`` at the end.

.. _issue #61:
   https://github.com/brainiak/brainiak/issues/61

Until we reach version 1.0, we will only support the latest released version.
Therefore, if you have a problem with an older version, please upgrade to the
latest version before creating an issue on GitHub.
