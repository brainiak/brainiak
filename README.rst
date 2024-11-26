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

If you have `Conda <conda.io>`_::

    conda install -c brainiak -c defaults -c conda-forge brainiak

Otherwise, or if you want to compile from source, install the requirements (see
`docs/installation`) and then install from PyPI::

    python3 -m pip install brainiak

Note that to use the ``brainiak.matnormal`` package, you need to install
additional dependencies. As of October 2020, the required versions are not
available as Conda packages, so you should install from PyPI, even when using
Conda::

    python3 -m pip install -U tensorflow tensorflow-probability

Note that we do not support Windows.


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
