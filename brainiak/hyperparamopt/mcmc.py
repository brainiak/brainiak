"""Metropolis-Hasting Random number generator

This implementation provides random samples from a user-given
probability density function through the Metropolis-Hasting algorithm.

"""

# Authors: Narayanan Sundaram (Intel Labs)

import numpy as np
import scipy.stats as st
import logging


def candidate(x):
    """Generates candidate around point x

    Returns
    -------

    sample from q(x*|x) - Unit Normal distribution around x
    """

    return np.random.standard_normal() + x


def candidate_dist(x, xp):  # return value of q(xp| x)
    return st.norm.pdf(x - xp)


def check_accept(xcurr, xprop, p):
    return min(1.0, p(xprop) / p(xcurr))


def get_next(x, p, n):
    xnext = np.zeros(n)
    xnext[-1] = x
    pxcurr = p(x)
    for i in range(n):
        xp = candidate(xnext[i - 1])
        pxp = p(xp)
        if (np.random.random() < pxp / pxcurr):
            xnext[i] = xp
            pxcurr = pxp
        else:
            xnext[i] = xnext[i - 1]
    return xnext


def get_chain(N, p, burn_in=2000):
    """Get a sequence of numbers sampled from a single chain
    of MCMC (Metropolis-Hastings) sampler

    Arguments
    ---------

    N : int
        Number of samples required

    p : function that returns a pdf value at any real number
        Distribution that needs to be sampled

    burn_in : int, default 2000
        Number of burn-in (discarded) samples

    Returns
    -------

    samples : 1D array, shape [N]
        Samples generated from MCMC sampler (should resemble samples from p(x))
    """

    x = np.ones(N)
    x0 = np.random.standard_normal() * 100
    while(p(x0) <= np.finfo(np.double).eps * 10):
        x0 = np.random.standard_normal() * 100
    if (p(x0) <= 0):
        logging.error('Markov chain failed to initialize properly \
            - Values probably very far from origin')

    # burn in iterations
    x0 = get_next(x0, p, burn_in)[-1]

    # actual iterations
    x = get_next(x0, p, N)
    return x


def get_multichain_samples(N, p, nchains=3, burn_in=2000):
    """Get a sequence of numbers sampled from multiple chains
    of MCMC (Metropolis-Hastings) sampler

    Arguments
    ---------

    N : int
        Number of samples per chain required

    p : function that returns a pdf value at any real number
        Distribution that needs to be sampled

    nchains : int, default 3
        Number of independent MCMC chains to sample

    burn_in : int, default 2000
        Number of burn-in (discarded) samples

    Returns
    -------

    samples : 1D array, shape [nchains*N]
        Samples generated from MCMC sampler (should resemble samples from p(x))
    """

    pts = np.zeros(nchains * N)
    for c in range(nchains):
        xp = get_chain(N * 3, p, burn_in=burn_in)
        pts[c * N: (c + 1) * N] =\
            xp[np.random.choice(N * 3, N)]  # pick N points at random
    return pts
