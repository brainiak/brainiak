#!/bin/bash

set -e
set -o pipefail

flake8 --ignore=W503 --max-complexity=10 brainiak
rst-lint *.rst | { grep -v "is clean.$" || true; }
