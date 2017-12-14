#!/usr/bin/env bash

# Caust the script to exit if a single command fails.
# set -e

# Show explicitly which commands are currently running.
set -x

# Install dependencies
yum install -y -q \
  mpich2-devel \
  libgomp

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)
WHEEL_DIR=$SCRIPT_DIR/../.whl
mkdir -p $WHEEL_DIR

# - Use the brainiak source stored in /brainiak via docker run command
# - Use head of mpi4py (we can pick some more suitable tag)
git clone -q https://bitbucket.org/mpi4py/mpi4py /mpi4py
pushd /mpi4py
git checkout 3.0.0
popd

for PYTHON in cp34-cp34m cp35-cp35m cp36-cp36m; do
  /opt/python/${PYTHON}/bin/python -m pip install -U pip

  pushd /mpi4py
    git clean -f -f -x -d -q
    /opt/python/${PYTHON}/bin/python setup.py -q bdist_wheel
    auditwheel repair dist/*.whl
    mv wheelhouse/*.whl $WHEEL_DIR/
  popd

  git clean -f -f -x -d -q -e .whl
  /opt/python/${PYTHON}/bin/python -m pip install -q .
  /opt/python/${PYTHON}/bin/python setup.py -q bdist_wheel
  auditwheel repair dist/*.whl
  mv wheelhouse/*.whl $WHEEL_DIR/
done
