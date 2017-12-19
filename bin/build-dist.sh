#!/usr/bin/env bash

docker run --rm -w /src -v `pwd`:/src -ti brainiak/manylinux ./bin/build-dist-manylinux1.sh
