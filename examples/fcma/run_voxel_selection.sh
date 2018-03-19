#!/bin/sh

# make sure OMP_NUM_THREADS is specified

mpirun -np 2 python3 voxel_selection.py face_scene bet.nii.gz face_scene/mask.nii.gz face_scene/fs_epoch_labels.npy 12 18
