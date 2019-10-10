#!/bin/bash

# Install pymanopt via pip because there isn't a conda package
PIP_NO_INDEX=False $PYTHON -m pip install pymanopt

# NOTE: This is the recommended way to install packages
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
