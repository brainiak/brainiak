#!/bin/sh

#the number of process is configurable, 6 is just an example
NUM_PROCESS=${1:-6}
echo "About to run command 'mpirun -np ${NUM_PROCESS} python3 htfa_cv_example.py'"
mpirun -np ${NUM_PROCESS} python3 htfa_cv_example.py 
