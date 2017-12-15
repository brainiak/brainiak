#!/usr/bin/env bash

MACPYTHON_URL=https://www.python.org/ftp/python
DOWNLOAD_DIR=python_downloads
mkdir -p $DOWNLOAD_DIR

for VERSION in $VERSIONS
do
  PKG="python-$VERSION-macosx10.6.pkg"
  MAJOR=${VERSION%.*}

  # Install Python.
  INST_PATH=$DOWNLOAD_DIR/$PKG
  curl $MACPYTHON_URL/$VERSION/$PKG > $INST_PATH
  sudo installer -pkg $INST_PATH -target /

  pip$MAJOR install -q -U pip delocate
done
