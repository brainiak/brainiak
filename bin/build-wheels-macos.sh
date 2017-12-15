#!/usr/bin/env bash

# Cause the script to exit if a single command fails.
set -e

# Show explicitly which commands are currently running.
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)
WHEEL_DIR=$SCRIPT_DIR/../.whl
mkdir -p $WHEEL_DIR

git clone -q https://bitbucket.org/mpi4py/mpi4py
pushd mpi4py
git checkout 3.0.0
popd

for VERSION in $VERSIONS
do
  MAJOR=${VERSION%.*}
  PYTHON=python$MAJOR
  DELOCATE=$(dirname $PYTHON)/delocate-wheel

  git clean -f -f -x -d -q -e .whl -e mpi4py

  pushd mpi4py
    git clean -f -f -x -d -q
    $PYTHON setup.py bdist_wheel
    $DELOCATE dist/*.whl
    mv dist/*.whl $WHEEL_DIR/
  popd

  $PYTHON -m pip install -q .
  $PYTHON setup.py bdist_wheel
  $DELOCATE dist/*.whl
  mv dist/*.whl $WHEEL_DIR/

  $PYTHON -m pip uninstall -y .
  $PYTHON -m pip uninstall -y numpy cython pybind11 scipy mpi4py
done
