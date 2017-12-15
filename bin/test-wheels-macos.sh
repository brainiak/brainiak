#!/usr/bin/env bash

# Caust the script to exit if a single command fails.
set -e

# Show explicitly which commands are currently running.
set -x


MACPYTHON_PY_PREFIX=/Library/Frameworks/Python.framework/Versions
PY_MMS=("3.4" "3.5" "3.6")

# This array is just used to find the right wheel.
PY_WHEEL_VERSIONS=("34" "35" "36")

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)
WHEEL_DIR=$SCRIPT_DIR/../.whl

# Test whether we can install without any dependencies
# TODO: delete once we have setup.py setup correctly
for ((i=0; i<${#PY_MMS[@]}; ++i)); do
   # TODO: delete this section once we implement travis stages
   PY_MM=${PY_MMS[i]}
   PY_WHEEL_VERSION=${PY_WHEEL_VERSIONS[i]}

   PYTHON_EXE=$MACPYTHON_PY_PREFIX/$PY_MM/bin/python$PY_MM
   PIP="$(dirname $PYTHON_EXE)/pip$PY_MM"

   # Find the appropriate wheel by grepping for the Python version.
   MPI4PY_WHEEL=$(find $WHEEL_DIR -type f -maxdepth 1 -print | grep "$PY_WHEEL_VERSION" | grep mpi4py)

   # TODO: this will actually pick up both wheels since brainiak is in the path
   BRAINIAK_WHEEL=$(find $WHEEL_DIR -type f -maxdepth 1 -print | grep "$PY_WHEEL_VERSION" | grep brainiak)

   $PIP install --upgrade pip

   $PIP install -q $MPI4PY_WHEEL
   $PIP install -q $BRAINIAK_WHEEL
done

brew install mpich
# Test packages
for ((i=0; i<${#PY_MMS[@]}; ++i)); do
   PY_MM=${PY_MMS[i]}
   WHEEL_DIR=$WHEEL_DIR \
      PYTHON=$MACPYTHON_PY_PREFIX/$PY_MM/bin/python$PY_MM \
      $SCRIPT_DIR/../pr-check.sh
done
