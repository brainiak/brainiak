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

set -e
set -o pipefail

flake8 --config setup.cfg brainiak
flake8 --config tests/.flake8 tests
mypy --ignore-missing-imports brainiak tests/[!_]*
rst-lint ./*.rst | { grep -v "is clean.$" || true; }
towncrier --version=100 --draft > /dev/null 2>&1 \
    || echo "Error assembling news fragments using towncrier."

echo "run-checks finished successfully."
