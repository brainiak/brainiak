#!/bin/sh

pip freeze | grep -q /brainiak || {
    echo "You must install brainiak in editable mode using \"pip install -e\""`
        `" before calling "$(basename "$0")
    exit 1
}

py.test --cov=brainiak
