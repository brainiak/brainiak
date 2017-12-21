#!/usr/bin/env bash


$PYTHON -m pip install -q .
$PYTHON setup.py -q bdist_wheel -d tmp

 platform="unknown"
 unamestr="$(uname)"
 if [[ "$unamestr" == "Linux" ]]
 then
    auditwheel repair tmp/*.whl -w $WHEEL_DIR
 elif [[ "$unamestr" == "Darwin" ]]
 then
    $DELOCATE tmp/*.whl
    mv tmp/*.whl $WHEEL_DIR
 else
    echo "Unrecognized platform."
    exit 1
 fi
