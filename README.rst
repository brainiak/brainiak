READ THIS FIRST
===============

For directions about how to contribute to the Brain Imaging Analysis Kit,
please see CONTRIBUTING.rst in the repository
(also visible nicely formatted at
https://github.com/IntelPNI/brainiak.rst)

The material below is directed at end users. This "READ THIS FIRST" section will be removed before our first release.

Brain Imaging Analysis Kit
==========================

Brain Imaging Analysis Kit is a package of Python modules useful for neuroscience, primarily focused on
functional Magnetic Resonance Imaging (fMRI) analysis.

The package was originally created by a collaboration between Intel and the Princeton Neuroscience Institute (PNI).

To reduce verbosity, we refer to the Brain Imaging Analysis Kit as ``BrainIAK``. Whenever lowercase spelling is used (e.g., Python package name), we use ``brainiak``.

Requirements
============

BrainIAK requires Linux or MacOS X, and Python version 3.4 or higher.


Install
=======

BrainIAK will be available on PyPI once we finalize the open-sourcing process; for the moment it must be installed from GitHub.

Install directly from GitHub
----------------------------

To install directly from GitHub, do:

    pip install git+https://github.com/intelpni/brainiak.git

(note that you'll have to enter your username and password since this is
still a private repository).

Or if you have ssh keys installed, you can do:

    pip install git+ssh://github.com/intelpni/brainiak.git

and you won't have to enter username or password.

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
