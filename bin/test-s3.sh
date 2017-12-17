#!/usr/bin/env bash

docker run --rm -w /src -v `pwd`:/src -ti brainiak/manylinux ./bin/test-s3-manylinux1.sh
