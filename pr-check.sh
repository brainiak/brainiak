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

set -e

# If we pass PYTHON_EXE, then we have a wheel and we don't want environment
if [ -z $PYTHON_EXE ]; then
   PYTHON_EXE=python3
   PYTHON_WHL=0
   USE_VENV=1
else
   PYTHON_WHL=1
   USE_VENV=0
fi
echo $PYTHON_EXE

if [ ! -f brainiak/__init__.py ]
then
    echo "Run "$(basename "$0")" from the root of the BrainIAK hierarchy."
    exit 1
fi

basedir=$(pwd)

function create_venv_venv {
    $PYTHON_EXE -m venv ../$1
}

function activate_venv_venv {
    source ../$1/bin/activate
}

function deactivate_venv_venv {
    deactivate
}

function remove_venv_venv {
    rm -r ../$1
}

function create_conda_venv {
    conda create -n $1 --yes python=3
}

function activate_conda_venv {
    source activate $1
    # Pip may update setuptools while installing BrainIAK requirements and
    # break the Conda cached package, which breaks subsequent runs.
    conda install --yes -f setuptools
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
    if [ $USE_VENV -eq 1 ]; then
        $deactivate_venv
        cd $basedir
        rm -f .coverage.*
        $remove_venv $venv
    fi
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
else
    create_venv=create_venv_venv
    activate_venv=activate_venv_venv
    deactivate_venv=deactivate_venv_venv
    remove_venv=remove_venv_venv
fi

# Check if running in an sdist
if git ls-files --error-unmatch pr-check.sh 2> /dev/null
then
    git clean -Xf .
else
    sdist_mode="--sdist-mode"
fi

if [ $USE_VENV -eq 1 ]; then
   venv=$(mktemp -u brainiak_pr_venv_XXXXX) || \
       exit_with_error "mktemp -u error"
   $create_venv $venv || {
       exit_with_error "Virtual environment creation failed."
   }
   $activate_venv $venv || {
       $remove_venv $venv
       exit_with_error "Virtual environment activation failed."
   }
fi

# install brainiak in editable mode (required for testing)
# brainiak will also be installed together with the developer dependencies, but
# we install it first here to check that installation succeeds without the
# developer dependencies.
$PYTHON_EXE -m pip install -q numpy scipy cython pybind11

if [ $PYTHON_WHL -eq 0 ]; then
   $PYTHON_EXE -m pip install -q $ignore_installed -U -e . || \
       exit_with_error_and_venv "Failed to install BrainIAK."
fi

# install developer dependencies
$PYTHON_EXE -m pip install -q $ignore_installed -U -r requirements-dev.txt || \
    exit_with_error_and_venv "Failed to install development requirements."

# static analysis
./run-checks.sh || \
    exit_with_error_and_venv "run-checks failed"

# run tests
PYTHON_EXE=$PYTHON_EXE ./run-tests.sh $sdist_mode || \
    exit_with_error_and_venv "run-tests failed"

# build documentation
cd docs
export THEANO_FLAGS='device=cpu,floatX=float64,blas.ldflags=-lblas'

if [ ! -z $SLURM_NODELIST ]
then
    make_wrapper="srun -n 1"
fi
$make_wrapper make || {
    cd -
    exit_with_error_and_venv "make docs failed"
}
cd -

if [ $USE_VENV -eq 1 ]; then
  $deactivate_venv
  $remove_venv $venv
fi

echo "pr-check finished successfully."
