"""Initialization routines for the Shared Response Model (SRM) algorithms
"""

# Authors: Po-Hsuan Chen (Princeton Neuroscience Institute) and Javier Turek (Intel Labs), 2015

import numpy as np


def init_w_transforms(method, data, features, rand_seed):
    # Initializes the SRM the mappings (Wi matrices)
    # Methods (in method):
    # 1 - Orthogonal random matrix per subject
    # 2 - Submatrix of the Identity matrix
    # rand_seed is a seed for the random number generator in numpy

    np.random.seed(rand_seed)

    w = []
    subjects = len(data)
    voxels = np.zeros(subjects)

    if method == 1:
        # Set Wi to a random orthogonal voxels by TRs matrix
        for subject in range(subjects):
            voxels = data[subject].shape[0]
            rnd_matrix = np.mat(np.random.random((voxels, features)))
            q, r = np.linalg.qr(rnd_matrix)
            w.append(q)
    else:
        # Set Wi to a subset of columns of the identity matrix of size voxels by TRs
        for subject in range(subjects):
            voxels = data[subject].shape[0]
            identity = np.identity(voxels)
            w.append(identity[:, :features])

    return w, voxels

