READ THIS FIRST
===============

For directions about how to contribute to the toolkit, please see CONTRIBUTING.rst in the repository
(also visible nicely formatted at https://github.com/IntelPNI/toolkit/blob/master/CONTRIBUTING.rst)

The material below is directed at end users. This "READ THIS FIRST" section will be removed before our first release.

("toolkit" is a provisional name and will be replaced soon with a real one.
Some of the links and directions below won't work until we pick a name)

toolkit
=======

toolkit is a package of Python modules useful for neuroscience, primarily focused on
functional Magnetic Resonance Imaging (fMRI) analysis.

The package was originally created by a collaboration between Intel and the Princeton Neuroscience Institute (PNI).

Requirements
============

The toolkit requires Linux or MacOS X, and Python version 3.4 or higher.


Install
=======

The toolkit will be available on PyPy once the name has been finalized, but for the moment it must be installed from source.

Install from GitHub
-------------------

To install directly from GitHub, do:

    pip install git+https://github.com/intelpni/toolkit.git

(note that you'll have to enter your username and password since this is
still a private repository).

Or if you have ssh keys installed, you can do:

    pip install git+ssh://github.com/intelpni/toolkit.git

and you won't have to enter username or password.

Install from source clone
-------------------------

If you'd prefer to install from a local copy of the source, follow these steps:

To check out the source, do:

    git clone https://github.com/intelpni/toolkit


Then to install, do:

    cd toolkit

    pip install .

    ..
       To install via `pip`, execute the following at a command prompt::
       TODO
       pip install -U --user toolkit


Building documentation
----------------------

If desired, you can build the documentation yourself. Currently we don't have the docs hosted online, so this is a necessary step at the moment. Once the toolkit is made public, we'll host documentation at readthedocs.org, so most users will not need to build docs.

To build the documentation, you must have Sphinx installed (you may already have it, even if you've never heard of it). If you need to install, please follow directions from http://www.sphinx-doc.org/en/stable/install.html

Once you have sphinx installed, you can do (assuming you're already in the toolkit folder)

    cd docs

    make html

This will generate html documentation in the _build/html folder within the docs folder. _build/html/index.html is the starting page.


Links
=====

- Official source code repo: https://github.com/IntelPNI/toolkit
- HTML documentation (stable release): TODO (will be hosted on readthedocs.org once repository is made public)



Contribute
==========

Please read the contributor's guide at
https://github.com/IntelPNI/toolkit/blob/master/CONTRIBUTING.rst
