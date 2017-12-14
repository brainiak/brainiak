#!/bin/bash

#  Copyright 2016 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

set -ex
set -o pipefail

if [ $PYTHON_EXE == 'python3' ]; then
  RST_LINT=rst-lint
  TOWNCRIER=towncrier
else
  RST_LINT=$(dirname $PYTHON_EXE)/rst-lint
  TOWNCRIER=$(dirname $PYTHON_EXE)/towncrier
fi

$PYTHON_EXE -m flake8 --config setup.cfg brainiak
$PYTHON_EXE -m flake8 --config tests/.flake8 tests
$PYTHON_EXE -m mypy --ignore-missing-imports brainiak tests/[!_]*
$RST_LINT ./*.rst | { grep -v "is clean.$" || true; }
$TOWNCRIER --version=100 --draft > /dev/null 2>&1 \
    || echo "Error assembling news fragments using towncrier."

echo "run-checks finished successfully."
