#!/usr/bin/env bash

# Caust the script to exit if a single command fails.
set -e

# Show explicitly which commands are currently running.
set -x


MACPYTHON_PY_PREFIX=/Library/Frameworks/Python.framework/Versions
PY_MMS=("3.4"
        "3.5"
        "3.6")
# This array is just used to find the right wheel.
PY_WHEEL_VERSIONS=("34"
                   "35"
                   "36")

for ((i=0; i<${#PY_MMS[@]}; ++i)); do
  PY_MM=${PY_MMS[i]}
  PY_WHEEL_VERSION=${PY_WHEEL_VERSIONS[i]}

  PYTHON_EXE=$MACPYTHON_PY_PREFIX/$PY_MM/bin/python$PY_MM
  PIP_CMD="$(dirname $PYTHON_EXE)/pip$PY_MM"

  # Find the appropriate wheel by grepping for the Python version.
  PYTHON_WHEEL=$(find $ROOT_DIR/../.whl -type f -maxdepth 1 -print | grep -m1 "$PY_WHEEL_VERSION")

  # TODO: Install MPI wheel

  # Install the wheel.
  $PIP_CMD install $PYTHON_WHEEL

  # Run a simple test script to make sure that the wheel works.
  # $PYTHON_EXE $TEST_SCRIPT
done
