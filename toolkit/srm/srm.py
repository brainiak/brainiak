"""EM algorithm for the probabilistic Shared Response Model (SRM)

This implementation is based on the work:
[1] "A Reduced-Dimension fMRI Shared Response Model"
P.-H. Chen, J. Chen, Y. Yeshurun-Dishon, U. Hasson, J. Haxby, P. Ramadge
Advances in Neural Information Processing Systems (NIPS), 2015.
"""

# Authors: Po-Hsuan Chen (Princeton Neuroscience Institute) and Javier Turek (Intel Labs), 2015

import numpy as np
import scipy
from srm_init import init_w_transforms


def init_structures(data, subjects):
    # Initializes the remaining data structures for SRM: compute mean per subject (mu), remove the mean for each subject
    # (x), initialize rho^2 per subject (rho2), and compute the Frobenius norm of the data (trace_xtx)
    x = []
    mu = []
    rho2 = np.zeros(subjects)

    trace_xtx = np.zeros(subjects)
    for subject in range(subjects):
        mu.append(np.mean(data[subject], 1))
        rho2[subject] = 1
        trace_xtx[subject] = np.sum(data[subject] ** 2)
        x.append(data[subject] - mu[subject][:, np.newaxis])

    return x, mu, rho2, trace_xtx


def likelihood(chol_sigma_s_rhos, det_psi, chol_sigma_s, trace_xt_invsigma2_x, inv_sigma_s_rhos, wt_invpsi_x,
               samples):
    # Calculates the likelihood function
    log_det = np.log(np.diag(chol_sigma_s_rhos) ** 2).sum() + det_psi + np.log(np.diag(chol_sigma_s) ** 2).sum()
    sign = -np.sign(log_det)
    loglike = - 0.5 * samples * log_det - 0.5 * trace_xt_invsigma2_x
    loglike += 0.5 * np.trace(wt_invpsi_x.T.dot(inv_sigma_s_rhos).dot(wt_invpsi_x))
    # + const --> -0.5*nTR*nvoxel*subjects*math.log(2*math.pi)

    return sign, loglike


def srm(data, iterations, features, rand_seed, verbose):
    # Main function to run the probabilistic SRM algorithm
    samples = data[0].shape[1]
    subjects = len(data)

    # Initialization
    w, voxels = init_w_transforms(1, data, features, rand_seed)
    x, mu, rho2, trace_xtx = init_structures(data, subjects)
    shared_response = np.zeros((features, samples))
    sigma_s = np.identity(features)

    # Main loop of the algorithm
    for iteration in range(iterations):
        if verbose:
            print ('Iteration %d' % (iteration + 1))

        # E-step
        inv_rho2_sum = (1 / rho2).sum()
        (chol_sigma_s, lower_sigma_s) = scipy.linalg.cho_factor(sigma_s, check_finite=False)
        inv_sigma_s = scipy.linalg.cho_solve((chol_sigma_s, lower_sigma_s), np.identity(features), check_finite=False)
        sigma_s_rhos = inv_sigma_s + np.identity(features) * inv_rho2_sum
        (chol_sigma_s_rhos, lower_sigma_s_rhos) = scipy.linalg.cho_factor(sigma_s_rhos, check_finite=False)
        inv_sigma_s_rhos = scipy.linalg.cho_solve((chol_sigma_s_rhos, lower_sigma_s_rhos), np.identity(features),
                                                  check_finite=False)

        wt_invpsi_x = np.zeros((features, samples))
        trace_xt_invsigma2_x = 0.0

        for subject in range(subjects):
            wt_invpsi_x += (w[subject].T.dot(x[subject])) / rho2[subject]
            trace_xt_invsigma2_x += trace_xtx[subject] / rho2[subject]

        shared_response = sigma_s.dot(np.identity(features) - inv_rho2_sum * inv_sigma_s_rhos).dot(wt_invpsi_x)
        sigma_s = inv_sigma_s_rhos + shared_response.dot(shared_response.T) / float(samples)
        trace_sigma_s = samples * np.trace(sigma_s)

        det_psi = np.sum(np.log(rho2) * voxels)

        # M-step
        for subject in range(subjects):
            if verbose:
                print ('.'),
            a_subject = x[subject].dot(shared_response.T)
            perturbation = np.zeros(a_subject.shape)
            np.fill_diagonal(perturbation, 0.001)
            u_subject, s_subject, v_subject = np.linalg.svd(a_subject + perturbation, full_matrices=False)
            w[subject] = u_subject.dot(v_subject)
            rho2[subject] = trace_xtx[subject]
            rho2[subject] += -2 * np.sum(w[subject] * a_subject).sum()
            rho2[subject] += trace_sigma_s
            rho2[subject] /= float(samples * voxels[subject])

        if verbose:
            # calculate and print the current log-likelihood
            sign, loglike = likelihood(chol_sigma_s_rhos, det_psi, chol_sigma_s, trace_xt_invsigma2_x, inv_sigma_s_rhos,
                                       wt_invpsi_x, samples)
            if sign == -1:
                print('%dth iteration, log sign negative' % (iteration + 1))
            print('Objective function %f' % loglike)

    return sigma_s, w, mu, rho2, shared_response


def align(data, args):
    """Runs the EM algorithm for the probabilistic Shared Response Model (SRM) method

    The method receives a list of subject's data.
    Each subject is represented with a 2D matrix of size voxels by TRs.
    That is, data[i] is the 2D data matrix of subject i.

    It returns a list of the mappings Wi of each subject and the shared response S.

    The method iteratively computes an E-step and an M-step.

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
        print('Probabilistic SRM')

    # Check if all subjects have same number of TRs
    number_trs = data[0].shape[1]
    number_subjects = len(data)
    for subject in range(number_subjects):
        assert data[subject].shape[1] != number_trs, "Different number of TRs between subjects."

    # Run SRM with varying number of voxels version
    sigma_s, w, mu, rho2, s = srm(data, args.iterations, args.features, args.randseed, args.verbose)
    return w, s
