#!/bin/bash

# Install pymanopt and tensorflow_probability via pip because there isn't a conda 
# package, and install tensorflow 2.3 from pip because tensorflow_probability
# requires it and latest in conda is 2.2. 
PIP_NO_INDEX=False $PYTHON -m pip install tensorflow>=2.3
PIP_NO_INDEX=False $PYTHON -m pip install pymanopt
PIP_NO_INDEX=False $PYTHON -m pip install tensorflow_probability


# NOTE: This is the recommended way to install packages
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
