Brain Imaging Analysis Kit
==========================

.. image:: https://travis-ci.org/brainiak/brainiak.svg?branch=master
    :target: https://travis-ci.org/brainiak/brainiak

.. image:: https://badges.gitter.im/brainiak/brainiak.svg
   :alt: Join the chat at https://gitter.im/brainiak/brainiak
   :target: https://gitter.im/brainiak/brainiak?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

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

Note that you may see a ``Failed building wheel for brainiak`` message (`issue
#61`_). Installation will proceed despite this failure. You can safely ignore it
as long as you see ``Successfully installed`` at the end.

.. _issue #61:
   https://github.com/brainiak/brainiak/issues/61

Until we reach version 1.0, we will only support the latest released version.
Therefore, if you have a problem with an older version, please upgrade to the
latest version before creating an issue on GitHub.


Docker
======

You can also test BrainIAK without installing it using Docker::

    docker pull brainiak/brainiak
    docker run -it -p 8888:8888 -v brainiak:/mnt --name demo brainiak/brainiak

To run Jupyter notebooks in the running container, try::

    python3 -m notebook --allow-root --no-browser --ip=0.0.0.0

Then visit http://localhost:8888 in your browser and enter the token. Protip:
run ``screen`` before running the notebook command.

Note that we do not support MPI execution using Docker containers and that performance will not be optimal.


Support
=======

If you have a question or feedback, chat with us on `Gitter
<https://gitter.im/brainiak/brainiak>`_ or email our list at
brainiak@googlegroups.com. If you find a problem with BrainIAK, you can also
`open an issue on GitHub <https://github.com/brainiak/brainiak/issues>`_.


Examples
========

We include BrainIAK usage examples in the examples directory of the code
repository, e.g., `funcalign/srm_image_prediction_example.ipynb
<https://github.com/brainiak/brainiak/blob/master/examples/funcalign/srm_image_prediction_example.ipynb>`_.

To run the examples, download an archive of the `latest BrainIAK release from
GitHub <https://github.com/brainiak/brainiak/releases>`_. Note that we only
support the latest release at this moment, so make sure to upgrade your
BrainIAK installation.


Documentation
=============

The documentation is available at `pythonhosted.org/brainiak`_.

.. _pythonhosted.org/brainiak:
    https://pythonhosted.org/brainiak


Contributing
============

We welcome contributions. Have a look at the issues labeled "`easy`_" for
starting contribution ideas. Please read the guide in `CONTRIBUTING.rst`_
first.

.. _easy:
   https://github.com/brainiak/brainiak/issues?q=is%3Aissue+is%3Aopen+label%3Aeasy
.. _CONTRIBUTING.rst:
   https://github.com/brainiak/brainiak/blob/master/CONTRIBUTING.rst


Citing
======

Please cite BrainIAK in your publications as: "Brain Imaging Analysis Kit,
http://brainiak.org." Additionally, if you use `RRIDs
<https://scicrunch.org/resolver>`_ to identify resources, please mention
BrainIAK as "Brain Imaging Analysis Kit, RRID:SCR_014824". Finally, please cite
the publications referenced in the documentation of the BrainIAK modules you
use, e.g., `SRM <http://pythonhosted.org/brainiak/brainiak.funcalign.html>`_.
