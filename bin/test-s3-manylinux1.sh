#!/usr/bin/env bash

set -ex

# Test mpi4py wheels from s3
for VERSION in cp34-cp34m cp35-cp35m cp36-cp36m
do
   PYTHON=/opt/python/${VERSION}/bin/python3
   PYTHON=$PYTHON ./bin/test-local-install.sh
done
