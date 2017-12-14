#!/usr/bin/env bash

docker run --rm -w /src -v `pwd`:/src -ti quay.io/pypa/manylinux1_x86_64 /bin/bash
