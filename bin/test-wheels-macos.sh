#!/usr/bin/env bash

# Caust the script to exit if a single command fails.
set -e

# Show explicitly which commands are currently running.
set -x

if [ -z $PYPI_REPOSITORY_URL ]
then
   SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)
   WHEEL_DIR=$SCRIPT_DIR/../dist

   # Test whether we can install without any dependencies
   # TODO: delete once we have setup.py setup correctly
   for VERSION in $VERSIONS
   do
     MAJOR=${VERSION%.*}
     PYTHON_DIR=$(dirname $(realpath $(which python$MAJOR)))
     PYTHON=$PYTHON_DIR/python$MAJOR
     WHEEL_VERSION=$(echo $MAJOR | tr -d '.')

     git clean -f -f -x -d -q -e dist

     pushd ..
     $PYTHON -m venv venv
     source venv/bin/activate
     popd

     # Find the appropriate wheel by grepping for the Python version.
     # MPI4PY_WHEEL=$(find $WHEEL_DIR -type f -maxdepth 1 -print | grep "$WHEEL_VERSION" | grep mpi4py | grep macosx)

     # TODO: this will actually pick up both wheels since brainiak is in the path
     BRAINIAK_WHEEL=$(find $WHEEL_DIR -type f -maxdepth 1 -print | grep "$WHEEL_VERSION" | grep brainiak | grep macosx)

     # $PYTHON -m pip install $MPI4PY_WHEEL
     $PYTHON -m pip install $BRAINIAK_WHEEL

     deactivate
     rm -rf ../venv
   done
fi

brew install mpich

# Test packages
for VERSION in $VERSIONS
do
  # TODO: refactor this out
  MAJOR=${VERSION%.*}
  PYTHON_DIR=$(dirname $(realpath $(which python$MAJOR)))
  PYTHON=$PYTHON_DIR/python$MAJOR
  git clean -f -f -x -d -q -e dist

  if [ -z $PYPI_REPOSITORY_URL ]
  then
     WHEEL_DIR=$WHEEL_DIR PYTHON=$PYTHON $SCRIPT_DIR/pr-check.sh
  else
     PYTHON=$PYTHON $SCRIPT_DIR/pr-check.sh
  fi
done
