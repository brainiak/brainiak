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

# Check whether we are running on Princeton's della compute cluster.
is_della=false
if [[ $(hostname -s) == della* ]];
then
    is_della=true
fi

# Set this to where brainiak example notebooks and their datasets should be stored when running.
EXAMPLE_NOTEBOOKS_DIR=docs/examples

# Check if we are running on della.princeton.edu
if [[ "$is_della" == true ]]; then
    echo "Running on della, load required modules"

    # Load some modules we will need on della
    module load anaconda3/2023.3

    # Load openmpi and turn off infiniband
    # module load openmpi/gcc/2.0.2/64
    module load openmpi/gcc/4.1.2
    export MPICC=$(which mpicc)
    export OMPI_MCA_btl="vader,self,tcp"

    # Issues with pip using tmp on della
    export TMPDIR=/scratch/gpfs/dmturner/tmp

fi

if [ ! -f src/brainiak/__init__.py ]
then
    echo "Run "$(basename "$0")" from the root of the BrainIAK hierarchy."
    exit 1
fi

basedir=$(pwd)

function create_venv_venv {
    python3 -m venv ../$1
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
    conda create -n $1 --yes python=3.8
}

function activate_conda_venv {
    source activate $1
    # Pip may update setuptools while installing BrainIAK requirements and
    # break the Conda cached package, which breaks subsequent runs.
    conda install --yes -f setuptools

    if [[ "$is_della" == true ]]; then
        # On della, we need to set LD_LIBRARY_PATH to the conda environment libs explicitly.
        # This is because when importing tensorflow the system libstd++ is picked
        # up instead of the conda one. This causes subsequent GLIB version errors
        # when trying to load compiled modules
        export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
    fi

}

function deactivate_conda_venv {

    if [[ "$is_della" == true ]]; then
        conda deactivate
    else
        source deactivate
    fi
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

if [ -z "$IGNORE_CONDA" ] && [ "$(which conda)" ]
then
    export PYTHONNOUSERSITE=True
    create_venv=create_conda_venv
    activate_venv=activate_conda_venv
    deactivate_venv=deactivate_conda_venv
    remove_venv=remove_conda_venv
    #ignore_installed="--ignore-installed"
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

venv=$(mktemp -u brainiak_pr_venv_XXXXX) || \
    exit_with_error "mktemp -u error"
$create_venv $venv || {
    exit_with_error "Virtual environment creation failed."
}
$activate_venv $venv || {
    $remove_venv $venv
    exit_with_error "Virtual environment activation failed."
}


if [[ "$is_della" == true ]]; then
    # We need to fetch any data needed for running notebook examples
    # Update our data cache with any download_data.sh scripts found in the repo
    BRAINIAK_EXAMPLES_DATA_CACHE_DIR=/scratch/gpfs/dmturner/brainiak_tests/brainiak-example-data
    echo "Copying download_data.sh scripts to brainiak-example-data cache"
    rsync -av --prune-empty-dirs --include="*/" --include="download_data.sh" --exclude="*" $EXAMPLE_NOTEBOOKS_DIR/ $BRAINIAK_EXAMPLES_DATA_CACHE_DIR/

    # Download any data, this should only trigger downloads for new datasets since download_data.sh should check if the data exists.
    echo "Executing download_data scripts in cache directory"
    pushd .
    cd $BRAINIAK_EXAMPLES_DATA_CACHE_DIR
    bash download_data.sh
    popd

    echo "Updating the working repo with any data downloaded into the cache"
    rsync -av $BRAINIAK_EXAMPLES_DATA_CACHE_DIR/ $EXAMPLE_NOTEBOOKS_DIR/

    # Skip upgrading pip, this was causing failures on della, not sure why.

    # Install mpi4py first, no cache director
    pip install mpi4py --no-cache-dir || \
        exit_with_error_and_venv "Failed to install mpi4py."

else
    python3 -m pip install -U pip || \
        exit_with_error_and_venv "Failed to update Pip."
fi

# install brainiak in editable mode (required for testing)
# Install with all dependencies (testing, documentation, examples, etc.)
python3 -m pip install $ignore_installed -U \
    -v --config-settings=cmake.verbose=true --config-settings=logging.level=INFO \
    -e .[all] || \
    exit_with_error_and_venv "Failed to install BrainIAK."


# static analysis, skip on della for now, failing for numpy 1.20 typing issues I think
if [[ "$is_della" == false ]]; then
    ./run-checks.sh || exit_with_error_and_venv "run-checks failed"
fi

# run tests
if [[ "$is_della" == true ]]; then
    echo "Running on della head node, need to request time on a compute node"
    export BRAINIAKDEV_MPI_COMMAND=srun
    salloc -t 03:00:00 -N 1 -n 16 sh run-tests.sh $sdist_mode || \
        exit_with_error_and_venv "run-tests failed"
else
    ./run-tests.sh $sdist_mode || \
        exit_with_error_and_venv "run-tests failed"
fi



# build documentation, only if not della
if [[ "$is_della" == true ]]; then
    echo "Skipping docs build on della"
else
    cd docs

    if [ ! -z $SLURM_NODELIST ]
    then
        if [[ "$is_della" == true ]]; then
            make_wrapper="srun -t 00:30:00 -N 1 -n 4 "
        else
            make_wrapper="srun -n 1"
        fi
    fi
    $make_wrapper make || {
        cd -
        exit_with_error_and_venv "make docs failed"
    }
    cd -
fi

$deactivate_venv
$remove_venv $venv

echo "pr-check finished successfully."
