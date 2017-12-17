#!/usr/bin/env bash

set -ex

# Test mpi4py wheels from s3
for VERSION in $VERSIONS
do
   MAJOR=${VERSION%.*}
   PYTHON=$(realpath $(which python$MAJOR))
   PYTHON=$PYTHON ./bin/test-local-install.sh
done
