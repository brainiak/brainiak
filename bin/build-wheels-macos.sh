#!/usr/bin/env bash

# Cause the script to exit if a single command fails.
# set -e

# Show explicitly which commands are currently running.
set -x

MACPYTHON_URL=https://www.python.org/ftp/python
MACPYTHON_PY_PREFIX=/Library/Frameworks/Python.framework/Versions
DOWNLOAD_DIR=python_downloads

PY_VERSIONS=("3.4.4"
             "3.5.3"
             "3.6.1")
PY_INSTS=("python-3.4.4-macosx10.6.pkg"
          "python-3.5.3-macosx10.6.pkg"
          "python-3.6.1-macosx10.6.pkg")
PY_MMS=("3.4"
        "3.5"
        "3.6")

mkdir -p $DOWNLOAD_DIR

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)
WHEEL_DIR=$SCRIPT_DIR/../.whl
mkdir -p $WHEEL_DIR

git clone -q https://bitbucket.org/mpi4py/mpi4py

for ((i=0; i<${#PY_VERSIONS[@]}; ++i)); do
  PY_VERSION=${PY_VERSIONS[i]}
  PY_INST=${PY_INSTS[i]}
  PY_MM=${PY_MMS[i]}

  git clean -f -f -x -d -q -e .whl -e $DOWNLOAD_DIR -e mpi4py

  # Install Python.
  INST_PATH=python_downloads/$PY_INST
  curl $MACPYTHON_URL/$PY_VERSION/$PY_INST > $INST_PATH
  sudo installer -pkg $INST_PATH -target /

  PYTHON_EXE=$MACPYTHON_PY_PREFIX/$PY_MM/bin/python$PY_MM
  PIP_CMD="$(dirname $PYTHON_EXE)/pip$PY_MM"

  $PIP_CMD install -q wheel delocate numpy cython pybind11 scipy

  pushd mpi4py
    git clean -f -f -x -d -q
    env ARCHFLAGS="-arch x86_64" $PYTHON_EXE setup.py bdist_wheel
    $MACPYTHON_PY_PREFIX/$PY_MM/bin/delocate-wheel dist/*.whl
    mv dist/*.whl $WHEEL_DIR/
  popd

  env ARCHFLAGS="-arch x86_64" $PIP_CMD install -q .
  env ARCHFLAGS="-arch x86_64" $PYTHON_EXE setup.py bdist_wheel
  $MACPYTHON_PY_PREFIX/$PY_MM/bin/delocate-wheel dist/*.whl
  mv dist/*.whl $WHEEL_DIR/

  $PIP_CMD uninstall -y .
  $PIP_CMD uninstall -y numpy cython pybind11 scipy mpi4py

done

echo $WHEEL_DIR
