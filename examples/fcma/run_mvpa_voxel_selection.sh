#!/bin/sh
mpirun -np 1 python mvpa_voxel_selection.py face_scene bet.nii.gz face_scene/visual_top_mask.nii.gz face_scene/fs_epoch_labels.npy 18
