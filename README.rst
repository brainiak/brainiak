Brain Imaging Analysis Kit
==========================

.. image:: https://github.com/brainiak/brainiak/actions/workflows/ci.yml/badge.svg
    :alt: Status of GitHub Actions workflow
    :target: https://github.com/brainiak/brainiak/actions

.. image:: https://codecov.io/gh/brainiak/brainiak/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/brainiak/brainiak

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


Quickstart
==========

You can install BrainIAK wheels from PyPI on Linux, macOS, and Windows, e.g. using pip::

    python3 -m pip install brainiak

If you need MPI, see the installation requirements in `docs/installation`.

If you have `Conda <conda.io>`_, you can also use our Conda packages (not available for Windows). Installing BrainIAK will also install MPI::

    conda install -c brainiak -c conda-forge brainiak


Docker
======

You can also test BrainIAK without installing it using Docker::

    docker pull brainiak/brainiak
    docker run -it -p 8899:8899 brainiak/brainiak

Jupyter Notebook will start automatically; visit the URL shown in the Docker command output to access it. You can then run the BrainIAK examples or create new notebooks. You can also try a `sample example <http://127.0.0.1:8899/notebooks/examples/funcalign/rsrm_synthetic_reconstruction.ipynb>`_.

Note that we do not support MPI execution using Docker containers and that performance will not be optimal.


Support
=======

If you have a question or feedback, chat with us on `Matrix via Gitter
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

The documentation is available at http://brainiak.org/docs.


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
use, e.g., `SRM <http://brainiak.org/docs/brainiak.funcalign.html>`_.
