#!/usr/bin/env bash

set -ex

# Install and configure ssh so we can ssh locally for MPI tests
yum install -y -q openssh-server mpich2
service sshd start

ssh-keygen -f ~/.ssh/id_rsa -t rsa -N '' -b 4096
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

echo "Host *" >> ~/.ssh/config
echo "  StrictHostKeyChecking no" >> ~/.ssh/config
