#!/bin/bash
# Check readiness for pull request

set -e

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
    $remove_venv $venv
    exit_with_error "$1"
}

if [ $(which conda) ]
then
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

# install developer dependencies
pip install $ignore_installed -U -r requirements-dev.txt || \
    exit_with_error_and_venv "pip failed to install requirements"

# static analysis
./run-checks.sh || \
    exit_with_error_and_venv "run-checks failed"

# install brainiak in editable mode (required for testing)
pip install $ignore_installed -U -e . || \
    exit_with_error_and_venv "pip failed to install BrainIAK"

# run tests
./run-tests.sh || \
    exit_with_error_and_venv "run-tests failed"

# build documentation
cd docs
make || {
    cd -
    exit_with_error_and_venv "make docs failed"
}
cd -

# optional: remove virtualenv
$deactivate_venv
$remove_venv $venv

echo "Check successful"
