#!/usr/bin/env bash

# Caust the script to exit if a single command fails.
set -e

# Show explicitly which commands are currently running.
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)
WHEEL_DIR=$SCRIPT_DIR/../.whl

# Test install
for PYTHON in cp34-cp34m cp35-cp35m cp36-cp36m; do
   MPI4PY_WHEEL=$(find $WHEEL_DIR -type f | grep $PYTHON | grep mpi4py)
   BRAINIAK_WHEEL=$(find $WHEEL_DIR -type f | grep $PYTHON | grep brainiak)

   PIP=/opt/python/${PYTHON}/bin/pip

   $PIP install -q $MPI4PY_WHEEL
   $PIP install -q $BRAINIAK_WHEEL
done

# Separate tests into a separate loop because we want to make sure
# we can install brainiak without installing mpich, but require mpiexec during tests

# Install dependencies
yum install -y -q \
  mpich2-devel \
  libgomp

for PYTHON in cp34-cp34m cp35-cp35m cp36-cp36m; do
  PYTHON_EXE=/opt/python/$PYTHON/bin/python $SCRIPT_DIR/../pr-check.sh
done
