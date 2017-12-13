#!/usr/bin/env bash

docker run --rm -w /brainiak -v `pwd`:/brainiak -ti quay.io/pypa/manylinux1_x86_64 ./bin/build-wheels-manylinux1.sh
