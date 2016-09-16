#!/bin/sh

#  Copyright 2016 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

set -e

pip freeze | grep -qi /brainiak || {
    echo "You must install brainiak in editable mode using \"pip install -e\""`
        `" before calling "$(basename "$0")
    exit 1
}

mpi_command=mpiexec

if [ ! -z $SLURM_NODELIST ]
then
    mpi_command=srun
fi

$mpi_command -n 2 coverage run -m pytest
coverage combine
coverage report
coverage html
coverage xml
