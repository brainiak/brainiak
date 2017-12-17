#!/usr/bin/env bash

set -ex

# Install package from source in four ways
installs=(
   "pip install ."
   "pip install -e ."

   # TODO: Figure out why these don't work
   # "python3 setup.py install"
   # "python3 setup.py develop"
)

if [ -z $PYTHON ]
then
   PYTHON=python3
fi

for install in "${installs[@]}"
do
   $PYTHON -m venv venv
   source venv/bin/activate
   $install
   deactivate
   rm -rf venv
done
