#!/bin/bash

# Install from PyPI because there is no current conda package for the
# following. Explicitly install dependencies with no conda package as well
# because otherwise conda-build does not include them in the output package.
PIP_NO_INDEX=False $PYTHON -m pip install "pymanopt"

$PYTHON -m pip install . -vv --no-deps --no-build-isolation
