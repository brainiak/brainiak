#!/usr/bin/env bash

for VERSION in $VERSIONS
do
   # Download source
   wget https://github.com/python/cpython/archive/v$VERSION.tar.gz
   tar zxf v$VERSION

   # Install
   pushd cpython-$VERSION
      ./configure --enable-shared
      make -j
      make install
   popd

   PYTHON=python${VERSION%.*}
   $PYTHON -m pip install -U pip wheel

   # Clean up
   rm -rf cpython-$VERSION v$VERSION
done
