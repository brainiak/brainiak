"""Optimization algorithm for the deterministic Shared Response Model (SRM)

This implementation is based on the work:
[1] "A Reduced-Dimension fMRI Shared Response Model"
P.-H. Chen, J. Chen, Y. Yeshurun-Dishon, U. Hasson, J. Haxby, P. Ramadge
Advances in Neural Information Processing Systems (NIPS), 2015.
"""

# Authors: Po-Hsuan Chen (Princeton Neuroscience Institute) and Javier Turek (Intel Labs), 2015

import numpy as np
from srm_init import init_w_transforms


def compute_s(data, w, subjects, features, samples):
    # Computes the shared response S as the average of Wi'*Xi
    s = np.zeros((features, samples))
    for m in xrange(subjects):
        s = s + w[m].T.dot(data[m])
    s /= float(subjects)

    return s


def objective_function(data, w, s, subjects):
    # Computes the objective function of the deterministic SRM, which is given by the sum of Frobenius norms of the
    # residuals: || Xi - Wi*S||_F^2
    obj_val_tmp = 0.0
    for subject in range(subjects):
        obj_val_tmp += np.linalg.norm(data[subject] - w[subject].dot(s), 'fro')
    return obj_val_tmp


def srm_det(data, iterations, features, rand_seed, verbose):
    # Main function to run the deterministic SRM algorithm
    subjects = len(data)
    samples = data[0].shape[1]

    # Initialization:
    # Set Wi to a random orthogonal voxels by TRs matrixR_qr
    w, voxels = init_w_transforms(1, data, features, rand_seed)
    # Initialize S = 1/subjects * sum(W'*X)
    s = compute_s(data, w, subjects, features, samples)

    # Main loop:
    for iteration in range(iterations):
        if verbose:
            print ('Iteration %d' % (iteration + 1))

        # Update the mappings Wi
        for m in range(subjects):
            if verbose:
                print ('.'),

            a_subject = data[m].dot(s.T)
            pert = np.zeros(a_subject.shape)
            np.fill_diagonal(pert, 0.001)
            u_subject, s_subject, v_subject = np.linalg.svd(a_subject + pert, full_matrices=False)
            w[m] = u_subject.dot(v_subject)

        # Update the shared response S
        s = compute_s(data, w, subjects, features, samples)

        if verbose:
            # calculate and print the objective function
            obj_val_tmp = objective_function(data, w, s, subjects)
            print('Objective funciton %f' % obj_val_tmp)

    return w, s


def align(data, args):
    """Runs an optimization algorithm for the deterministic Shared Response Model (SRM) method

    The method receives a list of subject's data.
    Each subject is represented with a 2D matrix of size voxels by TRs.
    That is, data[i] is the 2D data matrix of subject i.

    It returns a list of the mappings Wi of each subject and the shared response S.

    The method iteratively updates the shared response and the mappings minimizing the objective function.

    :note:
    The number of voxels may be different between subjects.
    However, the number of TRs must be the same across subjects.

    :param data: a list of 2D arrays with each subject fMRI data
    :param args: parameters for the SRM algorithm, and object with the following fields:
        .iterations = number of iterations
        .features = number of latent features
        .randseed = random seed for initialization
        .verbose = verbose mode on/off
    :return: w - a list of mapping transformations (Wi), s - the shared response
    """
    if args.verbose:
        print('Deterministic SRM')

    # Check if all subjects have same number of TRs
    number_trs = data[0].shape[1]
    number_subjects = len(data)
    for subject in range(number_subjects):
        assert data[subject].shape[1] != number_trs, "Different number of TRs between subjects."

    # Run SRM with varying number of voxels version
    w, s = srm_det(data, args.iterations, args.features, args.randseed, args.verbose)
    return w, s
