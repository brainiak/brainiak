#!/bin/bash

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

# Check readiness for pull request

set -ev

if [ ! -f brainiak/__init__.py ]
then
    echo "Run "$(basename "$0")" from the root of the BrainIAK repository"
    exit 1
fi

basedir=$(pwd)

function create_virtualenv_venv {
    virtualenv ../$1
}

function activate_virtualenv_venv {
    source ../$1/bin/activate
}

function deactivate_virtualenv_venv {
    deactivate
}

function remove_virtualenv_venv {
    rm -r ../$1
}

function create_conda_venv {
    conda create -n $1 --yes python=3
}

function activate_conda_venv {
    source activate $1
    # Anaconda does not provide pip3 (at least on our Jenkins machine,
    # Metacortex), so we install it in the virtual environment.
    pip install -I pip
}

function deactivate_conda_venv {
    source deactivate
}

function remove_conda_venv {
    conda env remove -n $1 --yes
}

function exit_with_error {
    echo $1 >&2
    exit 1
}

function exit_with_error_and_venv {
    $deactivate_venv
    cd $basedir
    rm -f .coverage.*
    $remove_venv $venv
    exit_with_error "$1"
}

if [ $(which conda) ]
then
    export PYTHONNOUSERSITE=True
    create_venv=create_conda_venv
    activate_venv=activate_conda_venv
    deactivate_venv=deactivate_conda_venv
    remove_venv=remove_conda_venv
    ignore_installed="--ignore-installed"
elif [ $(which virtualenv) ]
then
    create_venv=create_virtualenv_venv
    activate_venv=activate_virtualenv_venv
    deactivate_venv=deactivate_virtualenv_venv
    remove_venv=remove_virtualenv_venv
else
    echo "Cannot find virtualenv or conda."
    echo "You must install one of them or test manually."
    exit 1
fi

# optional, but highly recommended: create a virtualenv to isolate tests
venv=$(mktemp -u brainiak_pr_venv_XXXXX) || \
    exit_with_error "mktemp -u error"
$create_venv $venv || {
    exit_with_error "virtualenv creation failed"
}
$activate_venv $venv || {
    $remove_venv $venv
    exit_with_error "virtualenv activation failed"
}

# install brainiak in editable mode (required for testing)
pip3 install $ignore_installed -U -e . || \
    exit_with_error_and_venv "pip3 failed to install BrainIAK"

# install developer dependencies
pip3 install $ignore_installed -U -r requirements-dev.txt || \
    exit_with_error_and_venv "pip3 failed to install requirements"

# static analysis
./run-checks.sh || \
    exit_with_error_and_venv "run-checks failed"

# run tests
./run-tests.sh || \
    exit_with_error_and_venv "run-tests failed"

# build documentation
cd docs
export THEANO_FLAGS='device=cpu,floatX=float64'
git clean -Xf .
if [ ! -z $SLURM_NODELIST ]
then
    make_wrapper="srun -n 1"
fi
$make_wrapper make || {
    cd -
    exit_with_error_and_venv "make docs failed"
}
cd -

# optional: remove virtualenv
$deactivate_venv
$remove_venv $venv

echo "Check successful"
