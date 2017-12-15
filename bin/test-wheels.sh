#!/usr/bin/env bash

docker run --rm -w /src -v `pwd`:/src -ti brainiak/manylinux ./bin/test-wheels-manylinux1.sh
