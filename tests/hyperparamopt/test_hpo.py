import pytest


def test_simple_gmm():
    from brainiak.hyperparamopt.hpo import gmm_1d_distribution
    import numpy as np

    x = np.array([1., 1., 2., 3., 1.])
    d = gmm_1d_distribution(x, minlimit=0., maxlimit=4.)
    assert d(1.1) > d(3.5), "GMM distribution not behaving correctly"
    assert d(2.0) > d(3.0), "GMM distribution not behaving correctly"

    samples = d.get_samples(chains=3, points_per_chain=10)
    np.testing.assert_array_less(samples, 4.)
    np.testing.assert_array_less(0., samples)

def test_simple_hpo():
    from brainiak.hyperparamopt.hpo import fmin
    import numpy as np

    def f(args):
      x = args['x']
      return x*x

    s = {'x': {'dist': 'uniform', 'lo': -10., 'hi': 10.}}
    trials = []
    best = fmin(lossfn=f, space=s, algo=None, maxevals=100, trials=trials)

    yarray = np.array([tr['loss'] for tr in trials])
    np.testing.assert_array_less(yarray, 100.)

    xarray = np.array([tr['x'] for tr in trials])
    np.testing.assert_array_less(np.abs(xarray), 10.)

    assert best['loss'] < 100., "HPO out of range"
    assert np.abs(best['x']) < 10., "HPO out of range"
    assert np.abs(best['x']) < 1., "HPO not accurate"
