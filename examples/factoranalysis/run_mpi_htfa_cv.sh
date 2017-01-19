#!/bin/sh

#the number of process is configurable, 6 is just an example
NUM_PROCESS=6
mpirun -np ${NUM_PROCESS} python3 htfa_cv_example.py 
