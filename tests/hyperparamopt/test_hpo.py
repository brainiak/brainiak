import pytest


def test_simple_gmm():
    from brainiak.hyperparamopt.hpo import gmm_1d_distribution
    import numpy as np

    x = np.array([1., 1., 2., 3., 1.])
    d = gmm_1d_distribution(x, minlimit=0., maxlimit=4.)
    assert d(1.1) > d(3.5), "GMM distribution not behaving correctly"
    assert d(2.0) > d(3.0), "GMM distribution not behaving correctly"
    assert d(-1.0) == 0, "GMM distribution out of bounds error"
    assert d(9.0) == 0, "GMM distribution out of bounds error"

    samples = d.get_samples(chains=2, points_per_chain=10, burn_in=50)
    np.testing.assert_array_less(samples, 4.)
    np.testing.assert_array_less(0., samples)

def test_simple_gmm_weights():
    from brainiak.hyperparamopt.hpo import gmm_1d_distribution
    import numpy as np

    x = np.array([1., 1., 2., 3., 1., 3.])
    d = gmm_1d_distribution(x)

    x2 = np.array([1., 2., 3.])
    w = np.array([3., 1., 2.])
    d2 = gmm_1d_distribution(x2, weights=w)
    y2 = d2(np.array([1.1, 2.0]))

    assert(d2(1.1) == y2[0],
           "GMM distribution array & scalar results don't match")
    assert(np.abs(d(1.1) - d2(1.1)) < 1e-5,
           "GMM distribution weights not handled correctly")
    assert(np.abs(d(2.0) - d2(2.0)) < 1e-5,
           "GMM distribution weights not handled correctly")


def test_simple_hpo():
    from brainiak.hyperparamopt.hpo import fmin
    import numpy as np

    def f(args):
      x = args['x']
      return x*x

    s = {'x': {'dist': 'uniform', 'lo': -10., 'hi': 10.}}
    trials = []
    best = fmin(lossfn=f, space=s, maxevals=50, trials=trials, verbose=True)

    yarray = np.array([tr['loss'] for tr in trials])
    np.testing.assert_array_less(yarray, 100.)

    xarray = np.array([tr['x'] for tr in trials])
    np.testing.assert_array_less(np.abs(xarray), 10.)

    assert best['loss'] < 100., "HPO out of range"
    assert np.abs(best['x']) < 10., "HPO out of range"
