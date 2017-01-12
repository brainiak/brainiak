#!/bin/sh

mpirun -np 2 python voxel_selection.py face_scene bet.nii.gz face_scene/mask.nii.gz face_scene/fs_epoch_labels.npy 12 18
