#  Copyright 2016 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Hyper Parameter Optimization (HPO)

This implementation is based on the work in [Bergstra2011]_ and
[Bergstra2013]_.

.. [Bergstra2011] "Algorithms for Hyper-Parameter Optimization",
   James S. Bergstra and Bardenet, Rémi and Bengio, Yoshua
   and Kégl, Balázs. NIPS 2011

.. [Bergstra2013] "Making a Science of Model Search:
   Hyperparameter Optimization in Hundreds of Dimensions for
   Vision Architectures", James Bergstra, Daniel Yamins, David Cox.
   JMLR W&CP 28 (1) : 115–123, 2013

"""

# Authors: Narayanan Sundaram (Intel Labs)

import logging
import math
import numpy as np
from scipy.special import erf
import scipy.stats as st


logger = logging.getLogger(__name__)


def get_sigma(x, min_limit=-np.inf, max_limit=np.inf):
    """Compute the standard deviations around the points for a 1D GMM.

    We take the distance from the nearest left and right neighbors
    for each point, then use the max as the estimate of standard
    deviation for the gaussian mixture around that point.

    Arguments
    ---------
    x : 1D array
        Set of points to create the GMM

    min_limit : Optional[float], default : -inf
        Minimum limit for the distribution

    max_limit : Optional[float], default : inf
        maximum limit for the distribution

    Returns
    -------
    1D array
        Array of standard deviations
    """

    z = np.append(x, [min_limit, max_limit])
    sigma = np.ones(x.shape)
    for i in range(x.size):
        # Calculate the nearest left neighbor of x[i]
        # Find the minimum of (x[i] - k) for k < x[i]
        xleft = z[np.argmin([(x[i] - k) if k < x[i] else np.inf for k in z])]

        # Calculate the nearest right neighbor of x[i]
        # Find the minimum of (k - x[i]) for k > x[i]
        xright = z[np.argmin([(k - x[i]) if k > x[i] else np.inf for k in z])]

        sigma[i] = max(x[i] - xleft, xright - x[i])
        if sigma[i] == np.inf:
            sigma[i] = min(x[i] - xleft, xright - x[i])
        if (sigma[i] == -np.inf):  # should never happen
            sigma[i] = 1.0
    return sigma


class gmm_1d_distribution:
    """GMM 1D distribution.

    Given a set of points, we create this object so that we
    can calculate likelihoods and generate samples from this
    1D Gaussian mixture model.

    Attributes
    ----------
    points : 1D array
        Set of points to create the GMM

    N : int
        Number of points to create the GMM

    min_limit : Optional[float], default : -inf
        Minimum limit for the distribution

    max_limit : Optional[float], default : inf
        Maximum limit for the distribution

    weights : Optional[1D array], default : array of ones
        Used to weight the points non-uniformly if required
    """

    def __init__(self, x, min_limit=-np.inf, max_limit=np.inf, weights=1.0):
        self.points = x
        self.N = x.size
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.sigma = get_sigma(x, min_limit=min_limit, max_limit=max_limit)
        self.weights = (2
                        / (erf((max_limit - x) / (np.sqrt(2.) * self.sigma))
                           - erf((min_limit - x) / (np.sqrt(2.) * self.sigma)))
                        * weights)
        self.W_sum = np.sum(self.weights)

    def get_gmm_pdf(self, x):
        """Calculate the GMM likelihood for a single point.

        .. math::
            y = \\sum_{i=1}^{N} w_i
            \\times \\text{normpdf}(x, x_i, \\sigma_i)/\\sum_{i=1}^{N} w_i
            :label: gmm-likelihood

        Arguments
        ---------
        x : float
            Point at which likelihood needs to be computed

        Returns
        -------
        float
            Likelihood value at x
        """

        def my_norm_pdf(xt, mu, sigma):
            z = (xt - mu) / sigma
            return (math.exp(-0.5 * z * z)
                    / (math.sqrt(2. * np.pi) * sigma))

        y = 0
        if (x < self.min_limit):
            return 0
        if (x > self.max_limit):
            return 0
        for _x in range(self.points.size):
            y += (my_norm_pdf(x, self.points[_x], self.sigma[_x])
                  * self.weights[_x]) / self.W_sum
        return y

    def __call__(self, x):
        """Return the GMM likelihood for given point(s).

        See :eq:`gmm-likelihood`.

        Arguments
        ---------
        x : scalar (or) 1D array of reals
            Point(s) at which likelihood needs to be computed

        Returns
        -------
        scalar (or) 1D array
            Likelihood values at the given point(s)
        """

        if np.isscalar(x):
            return self.get_gmm_pdf(x)
        else:
            return np.array([self.get_gmm_pdf(t) for t in x])

    def get_samples(self, n):
        """Sample the GMM distribution.

        Arguments
        ---------
        n : int
            Number of samples needed

        Returns
        -------
        1D array
            Samples from the distribution
        """

        normalized_w = self.weights / np.sum(self.weights)
        get_rand_index = st.rv_discrete(values=(range(self.N),
                                        normalized_w)).rvs(size=n)
        samples = np.zeros(n)
        k = 0
        j = 0
        while (k < n):
            i = get_rand_index[j]
            j = j + 1
            if (j == n):
                get_rand_index = st.rv_discrete(values=(range(self.N),
                                                normalized_w)).rvs(size=n)
                j = 0
            v = np.random.normal(loc=self.points[i], scale=self.sigma[i])
            if (v > self.max_limit or v < self.min_limit):
                continue
            else:
                samples[k] = v
                k = k + 1
                if (k == n):
                    break
        return samples


def get_next_sample(x, y, min_limit=-np.inf, max_limit=np.inf):
    """Get the next point to try, given the previous samples.

    We use [Bergstra2013]_ to compute the point that gives the largest
    Expected improvement (EI) in the optimization function. This model fits 2
    different GMMs - one for points that have loss values in the bottom 15%
    and another for the rest. Then we sample from the former distribution
    and estimate EI as the ratio of the likelihoods of the 2 distributions.
    We pick the point with the best EI among the samples that is also not
    very close to a point we have sampled earlier.

    Arguments
    ---------
    x : 1D array
        Samples generated from the distribution so far

    y : 1D array
        Loss values at the corresponding samples

    min_limit : float, default : -inf
        Minimum limit for the distribution

    max_limit : float, default : +inf
        Maximum limit for the distribution

    Returns
    -------
    float
        Next value to use for HPO
    """

    z = np.array(list(zip(x, y)), dtype=np.dtype([('x', float), ('y', float)]))
    z = np.sort(z, order='y')
    n = y.shape[0]
    g = int(np.round(np.ceil(0.15 * n)))
    ldata = z[0:g]
    gdata = z[g:n]
    lymin = ldata['y'].min()
    lymax = ldata['y'].max()
    weights = (lymax - ldata['y']) / (lymax - lymin)
    lx = gmm_1d_distribution(ldata['x'], min_limit=min_limit,
                             max_limit=max_limit, weights=weights)
    gx = gmm_1d_distribution(gdata['x'], min_limit=min_limit,
                             max_limit=max_limit)

    samples = lx.get_samples(n=1000)
    ei = lx(samples) / gx(samples)

    h = (x.max() - x.min()) / (10 * x.size)
    # TODO
    # assumes prior of x is uniform; should ideally change for other priors
    # d = np.abs(x - samples[ei.argmax()]).min()
    # CDF(x+d/2) - CDF(x-d/2) < 1/(10*x.size) then reject else accept
    s = 0
    while (np.abs(x - samples[ei.argmax()]).min() < h):
        ei[ei.argmax()] = 0
        s = s + 1
        if (s == samples.size):
            break
    xnext = samples[ei.argmax()]

    return xnext


def fmin(loss_fn,
         space,
         max_evals,
         trials,
         init_random_evals=30,
         explore_prob=0.2):
    """Find the minimum of function through hyper parameter optimization.

    Arguments
    ---------
    loss_fn : ``function(*args) -> float``
        Function that takes in a dictionary and returns a real value.
        This is the function to be minimized.

    space : dictionary
        Custom dictionary specifying the range and distribution of
        the hyperparamters.
        E.g. ``space = {'x': {'dist':scipy.stats.uniform(0,1),
        'lo':0, 'hi':1}}``
        for a 1-dimensional space with variable x in range [0,1]

    max_evals : int
        Maximum number of evaluations of loss_fn allowed

    trials : list
        Holds the output of the optimization trials.
        Need not be empty to begin with, new trials are appended
        at the end.

    init_random_evals : Optional[int], default 30
        Number of random trials to initialize the
        optimization.

    explore_prob : Optional[float], default 0.2
        Controls the exploration-vs-exploitation ratio. Value should
        be in [0,1]. By default, 20% of trails are random samples.

    Returns
    -------
    trial entry (dictionary of hyperparameters)
           Best hyperparameter setting found.
           E.g. {'x': 5.6, 'loss' : 0.5} where x is the best hyparameter
           value found and loss is the value of the function for the
           best hyperparameter value(s).

    Raises
    ------
    ValueError
        If the distribution specified in space does not support a ``rvs()``
        method to generate random numbers, a ValueError is raised.
    """

    for s in space:
        if not hasattr(space[s]['dist'], 'rvs'):
            raise ValueError('Unknown distribution type for variable')
        if 'lo' not in space[s]:
            space[s]['lo'] = -np.inf
        if 'hi' not in space[s]:
            space[s]['hi'] = np.inf

    if len(trials) > init_random_evals:
        init_random_evals = 0

    for t in range(max_evals):
        sdict = {}

        if t >= init_random_evals and np.random.random() > explore_prob:
            use_random_sampling = False
        else:
            use_random_sampling = True

        yarray = np.array([tr['loss'] for tr in trials])
        for s in space:
            sarray = np.array([tr[s] for tr in trials])
            if use_random_sampling:
                sdict[s] = space[s]['dist'].rvs()
            else:
                sdict[s] = get_next_sample(sarray, yarray,
                                           min_limit=space[s]['lo'],
                                           max_limit=space[s]['hi'])

        logger.debug('Explore' if use_random_sampling else 'Exploit')
        logger.info('Next point ', t, ' = ', sdict)

        y = loss_fn(sdict)
        sdict['loss'] = y
        trials.append(sdict)

    yarray = np.array([tr['loss'] for tr in trials])
    yargmin = yarray.argmin()

    logger.info('Best point so far = ', trials[yargmin])
    return trials[yargmin]
