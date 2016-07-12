"""Hyper Parameter Optimization (HPO)

This implementation is based on the work:

.. [Bergstra2011] "Algorithms for Hyper-Parameter Optimization",
   James S. Bergstra and Bardenet, R\'{e}mi and Bengio, Yoshua
   and Bal\'{a}zs K\'{e}gl. NIPS 2011

.. [Bergstra2013] "Making a Science of Model Search:
   Hyperparameter Optimization in Hundreds of Dimensions for
   Vision Architectures", James Bergstra, Daniel Yamins, David Cox.
   JMLR W&CP 28 (1) : 115–123, 2013

"""

# Authors: Narayanan Sundaram (Intel Labs)

import logging
from .mcmc import get_multichain_samples
from .norm import getgmmpdf
import numpy as np
from scipy.special import erf
from tqdm import tqdm


logger = logging.getLogger(__name__)


def getsigma(x, minlimit=-np.inf, maxlimit=np.inf):
    z = np.append(x, [minlimit, maxlimit])
    sigma = np.ones(x.shape)
    for i in range(x.size):
        xleft = z[np.argmin([(x[i] - k) if k < x[i] else np.inf for k in z])]
        xright = z[np.argmin([(k - x[i]) if k > x[i] else np.inf for k in z])]
        sigma[i] = max(x[i] - xleft, xright - x[i])
        if sigma[i] == np.inf:
            sigma[i] = min(x[i] - xleft, xright - x[i])
        if (sigma[i] == -np.inf):
            sigma[i] = 1.0
    return sigma


class gmm_1d_distribution:
    """GMM 1D distribution.

    Given a set of points, we create this object so that we
    can calculate likelihoods and generate samples from this
    1D Gaussian mixture model.

    Parameters
    ----------

    x : 1D array
      Set of points to create the GMM

    minlimit : double, default : -inf
      Minimum limit for the distribution

    maxlimit : double, default : +inf
      Maximum limit for the distribution

    weights : double scalar or 1D array with same size as x, default 1.0
      Used to weight the points non-uniformly if required
    """

    def __init__(self, x, minlimit=-np.inf, maxlimit=np.inf, weights=1.0):
        self.points = x
        self.N = x.size
        self.minlimit = minlimit
        self.maxlimit = maxlimit
        self.sigma = getsigma(x, minlimit=minlimit, maxlimit=maxlimit)
        self.weights = 2. / (erf((maxlimit - x)
                             / (np.sqrt(2.) * self.sigma))
                             - erf((minlimit - x)
                             / (np.sqrt(2.) * self.sigma))) * weights
        # return self

    def __call__(self, xt):
        if (np.isscalar(xt)):
            return getgmmpdf(xt, self.points, self.sigma, self.weights,
                             self.minlimit, self.maxlimit)
        else:
            return np.array([getgmmpdf(t, self.points, self.sigma,
                                       self.weights, self.minlimit,
                                       self.maxlimit) for t in xt])

    def get_samples(self, chains=1, points_per_chain=1, burn_in=2000):
        pts = get_multichain_samples(N=points_per_chain,
                                     p=self, nchains=chains, burn_in=burn_in)
        return pts


def getNextSample(x, y, minlimit=-np.inf, maxlimit=np.inf, show_plot=False):
    z = np.array(list(zip(x, y)), dtype=np.dtype([('x', float), ('y', float)]))
    z = np.sort(z, order='y')
    n = y.shape[0]
    g = int(np.round(np.ceil(0.15 * n)))
    ldata = z[0:g]
    gdata = z[g:n]
    lymin = ldata['y'].min()
    lymax = ldata['y'].max()
    weights = (lymax - ldata['y']) / (lymax - lymin)
    lx = gmm_1d_distribution(ldata['x'], minlimit=minlimit,
                             maxlimit=maxlimit, weights=weights)
    gx = gmm_1d_distribution(gdata['x'], minlimit=minlimit, maxlimit=maxlimit)

    samples = lx.get_samples(chains=10, points_per_chain=100)
    ei = lx(samples) / gx(samples)

    if show_plot is True:
        import pylab as plt
        plt.scatter(samples, lx(samples), color='r')
        plt.scatter(samples, gx(samples), color='b')
        plt.scatter(samples, ei, color='g')
        plt.show()

    h = (x.max() - x.min()) / (10 * x.size)
    # assumes prior of x is uniform -- should change for different priors
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


def getSample(x, y, dist, minlimit=-np.inf, maxlimit=np.inf):
    if (dist == 'GMM'):
        return getNextSample(x, y, minlimit, maxlimit)
    if (dist == 'uniform'):
        return np.random.random() * (maxlimit - minlimit) + minlimit
    if (dist == 'loguniform'):
        return np.exp(np.random.random()
                      * (np.log(maxlimit) - np.log(minlimit))
                      + np.log(minlimit))
    else:
        logger.error('Unsupported distribution for variable')


def fmin(lossfn,
         space,
         maxevals,
         trials,
         init_random_evals=30,
         explore_prob=0.2,
         verbose=False):
    """Find the minimum of function through hyper paramter optimization

    Arguments
    ---------

    lossfn : function that takes in a dictionary and returns a real value
             Function to be minimized

    space : Dictionary specifying the range and distribution of
            the hyperparamters

    maxevals : int
               Maximum number of evaluations of lossfn allowed

    trials : list
             Holds the output of the optimization trials
             Need not be empty to begin with, new trials are appended
             at the end

    init_random_evals : int, default 30
                        Number of random trials to initialize the
                        optimization

    explore_prob : double in [0, 1], default 0.2
                   Controls the exploration-vs-exploitation ratio
                   Currently 20% of trails are random samples

    verbose : bool, default False
              Get information on current point being processed

    Returns
    -------

    best : trial entry (dictionary of hyperparameters)
           Best hyperparameter setting found
    """

    if (len(trials) > init_random_evals):
        init_random_evals = 0

    for t in tqdm(range(maxevals)):
        sdict = {}

        if (t >= init_random_evals and np.random.random() > explore_prob):
            search_algo = 'Exploit'
        else:
            search_algo = 'Explore'

        yarray = np.array([tr['loss'] for tr in trials])
        for s in space:
            sarray = np.array([tr[s] for tr in trials])
            dist = 'GMM' if (search_algo == 'Exploit') else space[s]['dist']
            sdict[s] = getSample(sarray, yarray, dist,
                                 minlimit=space[s]['lo'],
                                 maxlimit=space[s]['hi'])

        if (verbose):
            logger.info(search_algo)
            logger.info('Next point ', t, ' = ', sdict)

        y = lossfn(sdict)
        sdict['loss'] = y
        trials.append(sdict)

    yarray = np.array([tr['loss'] for tr in trials])
    yargmin = yarray.argmin()

    if (verbose):
        logger.info('Best point so far = ', trials[yargmin])
    return trials[yargmin]
