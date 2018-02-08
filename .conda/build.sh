#!/bin/bash

if [[ "$OSTYPE" == "darwin"* ]]; then
  export MACOSX_DEPLOYMENT_TARGET=10.9
  export CC=$(which clang)
  export CXX=$(which clang++)
fi

echo $PREFIX

# Install pymanopt via pip because there isn't a conda package
$PYTHON -m pip install pymanopt

# NOTE: This is the recommended way to install packages
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
