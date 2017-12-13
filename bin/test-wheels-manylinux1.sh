#!/usr/bin/env bash

# Caust the script to exit if a single command fails.
set -e

# Show explicitly which commands are currently running.
set -x

export WHEEL_DIR=/brainiak/.whl

for PYTHON in cp34-cp34m cp35-cp35m cp36-cp36m; do
   # - Find mpi wheel
   # - Find brainiak wheel
   # - Install both wheels
done
