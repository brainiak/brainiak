#!/usr/bin/env bash

# Caust the script to exit if a single command fails.
set -e

# Show explicitly which commands are currently running.
set -x

# Install dependencies
yum install -y \
  mpich2-devel \
  libgomp

WHEEL_DIR=/brainiak/.whl
mkdir -p $WHEEL_DIR

# - Use the brainiak source stored in /brainiak via docker run command
# - Use head of mpi4py (we can pick some more suitable tag)
git clone https://bitbucket.org/mpi4py/mpi4py /mpi4py

for PYTHON in cp34-cp34m cp35-cp35m cp36-cp36m; do
  pushd /mpi4py
    git clean -f -f -x -d
    /opt/python/${PYTHON}/bin/python setup.py bdist_wheel
    auditwheel repair dist/*.whl
    mv wheelhouse/*.whl $WHEEL_DIR/
  popd

  git clean -f -f -x -d -e .whl
  /opt/python/${PYTHON}/bin/python -m pip install .
  /opt/python/${PYTHON}/bin/python setup.py bdist_wheel
  auditwheel repair dist/*.whl
  mv wheelhouse/*.whl $WHEEL_DIR/
done
