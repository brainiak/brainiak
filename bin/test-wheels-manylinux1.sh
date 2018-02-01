#!/usr/bin/env bash

# Caust the script to exit if a single command fails.
set -e

# Show explicitly which commands are currently running.
set -x

# If we're downloading from repository, skip
if [ -z $PYPI_REPOSITORY_URL ]
then
   SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)
   WHEEL_DIR=$SCRIPT_DIR/../dist

   # Test whether we can install without any dependencies
   # TODO: delete once we have setup.py setup correctly
   for VERSION in cp34-cp34m cp35-cp35m cp36-cp36m; do
     # MPI4PY_WHEEL=$(find $WHEEL_DIR -type f | grep $VERSION | grep mpi4py | grep manylinux)
     BRAINIAK_WHEEL=$(find $WHEEL_DIR -type f | grep $VERSION | grep brainiak | grep manylinux)

     PYTHON=/opt/python/${VERSION}/bin/python3

     git clean -f -f -x -d -q -e dist

     $PYTHON -m venv venv
     source venv/bin/activate

     # $PYTHON -m pip install -q $MPI4PY_WHEEL
     $PYTHON -m pip install -q $BRAINIAK_WHEEL

     deactivate
     rm -rf venv
   done
fi

# Separate tests into a separate loop because we want to make sure
# we can install brainiak without installing mpich, but require mpiexec during tests
./bin/install-test-deps-manylinux1.sh
for VERSION in cp34-cp34m cp35-cp35m cp36-cp36m; do
  git clean -f -f -x -d -q -e dist

  if [ -z $PYPI_REPOSITORY_URL ]
  then
     mpi_command=mpiexec.hydra \
       WHEEL_DIR=$WHEEL_DIR \
       PYTHON=/opt/python/$VERSION/bin/python \
       $SCRIPT_DIR/pr-check.sh
  else
     mpi_command=mpiexec.hydra \
       PYTHON=/opt/python/$VERSION/bin/python \
       $SCRIPT_DIR/pr-check.sh
  fi
done
