#!/usr/bin/env bash

# Caust the script to exit if a single command fails.
set -e

# Show explicitly which commands are currently running.
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)
WHEEL_DIR=$SCRIPT_DIR/../.whl

# Test whether we can install without any dependencies
# TODO: delete once we have setup.py setup correctly
for VERSION in $VERSIONS
do
  MAJOR=${VERSION%.*}
  PYTHON_DIR=$(cd $(dirname $(readlink $(which python$MAJOR))); pwd)
  PYTHON=$PYTHON_DIR/python$MAJOR
  WHEEL_VERSION=$(echo $MAJOR | tr -d '.')

  $PYTHON -m venv venv
  source venv/bin/activate

  # Find the appropriate wheel by grepping for the Python version.
  MPI4PY_WHEEL=$(find $WHEEL_DIR -type f -maxdepth 1 -print | grep "$WHEEL_VERSION" | grep mpi4py)

  # TODO: this will actually pick up both wheels since brainiak is in the path
  BRAINIAK_WHEEL=$(find $WHEEL_DIR -type f -maxdepth 1 -print | grep "$WHEEL_VERSION" | grep brainiak)

  $PYTHON -m pip install -q $MPI4PY_WHEEL
  $PYTHON -m pip install -q $BRAINIAK_WHEEL

  deactivate
  rm -rf venv
done

brew install mpich

# Test packages
for VERSION in $VERSIONS
do
  # TODO: refactor this out
  MAJOR=${VERSION%.*}
  PYTHON_DIR=$(cd $(dirname $(readlink $(which python$MAJOR))); pwd)
  PYTHON=$PYTHON_DIR/python$MAJOR
  WHEEL_DIR=$WHEEL_DIR PYTHON=$PYTHON $SCRIPT_DIR/../pr-check.sh
done
