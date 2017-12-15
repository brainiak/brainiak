#!/usr/bin/env bash

# Caust the script to exit if a single command fails.
set -e

# Show explicitly which commands are currently running.
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)
WHEEL_DIR=$SCRIPT_DIR/../.whl

# Test whether we can install without any dependencies
# TODO: delete once we have setup.py setup correctly
for PYTHON in cp34-cp34m cp35-cp35m cp36-cp36m; do
  MPI4PY_WHEEL=$(find $WHEEL_DIR -type f | grep $PYTHON | grep mpi4py)
  BRAINIAK_WHEEL=$(find $WHEEL_DIR -type f | grep $PYTHON | grep brainiak)

  PIP=/opt/python/${PYTHON}/bin/pip

  $PIP install -q $MPI4PY_WHEEL
  $PIP install -q $BRAINIAK_WHEEL
done

# Install and configure ssh so we can ssh locally for MPI tests
yum install -y -q openssh-server
service sshd start

ssh-keygen -f ~/.ssh/id_rsa -t rsa -N '' -b 4096
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

echo "Host *" >> ~/.ssh/config
echo "  StrictHostKeyChecking no" >> ~/.ssh/config

# Separate tests into a separate loop because we want to make sure
# we can install brainiak without installing mpich, but require mpiexec during tests

# Install dependencies
yum install -y -q mpich2-devel

for VERSION in cp34-cp34m cp35-cp35m cp36-cp36m; do
  mpi_command=mpiexec.hydra \
    WHEEL_DIR=$WHEEL_DIR \
    PYTHON=/opt/python/$VERSION/bin/python \
    $SCRIPT_DIR/../pr-check.sh
done
