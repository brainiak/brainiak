#!/usr/bin/env bash

docker run --rm -v `pwd`:/brainiak -ti quay.io/xhochy/arrow_manylinux1_x86_64_base ./brainiak/bin/build-wheels-manylinux1.sh
