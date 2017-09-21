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
import pytest
import numpy as np
import scipy.stats as st
from brainiak.hyperparamopt.hpo import gmm_1d_distribution, fmin


def test_simple_gmm():
    x = np.array([1., 1., 2., 3., 1.])
    d = gmm_1d_distribution(x, min_limit=0., max_limit=4.)
    assert d(1.1) > d(3.5), "GMM distribution not behaving correctly"
    assert d(2.0) > d(3.0), "GMM distribution not behaving correctly"
    assert d(-1.0) == 0, "GMM distribution out of bounds error"
    assert d(9.0) == 0, "GMM distribution out of bounds error"

    samples = d.get_samples(n=25)
    np.testing.assert_array_less(samples, 4.)
    np.testing.assert_array_less(0., samples)


def test_simple_gmm_weights():
    x = np.array([1., 1., 2., 3., 1., 3.])
    d = gmm_1d_distribution(x)

    x2 = np.array([1., 2., 3.])
    w = np.array([3., 1., 2.])
    d2 = gmm_1d_distribution(x2, weights=w)
    y2 = d2(np.array([1.1, 2.0]))

    assert d2(1.1) == y2[0],\
        "GMM distribution array & scalar results don't match"
    assert np.abs(d(1.1) - d2(1.1)) < 1e-5,\
        "GMM distribution weights not handled correctly"
    assert np.abs(d(2.0) - d2(2.0)) < 1e-5,\
        "GMM distribution weights not handled correctly"


def test_simple_hpo():

    def f(args):
        x = args['x']
        return x*x

    s = {'x': {'dist': st.uniform(loc=-10., scale=20), 'lo': -10., 'hi': 10.}}
    trials = []

    # Test fmin and ability to continue adding to trials
    best = fmin(loss_fn=f, space=s, max_evals=40, trials=trials)
    best = fmin(loss_fn=f, space=s, max_evals=10, trials=trials)

    assert len(trials) == 50, "HPO continuation trials not working"

    # Test verbose flag
    best = fmin(loss_fn=f, space=s, max_evals=10, trials=trials)

    yarray = np.array([tr['loss'] for tr in trials])
    np.testing.assert_array_less(yarray, 100.)

    xarray = np.array([tr['x'] for tr in trials])
    np.testing.assert_array_less(np.abs(xarray), 10.)

    assert best['loss'] < 100., "HPO out of range"
    assert np.abs(best['x']) < 10., "HPO out of range"

    # Test unknown distributions
    s2 = {'x': {'dist': 'normal', 'mu': 0., 'sigma': 1.}}
    trials2 = []
    with pytest.raises(ValueError) as excinfo:
        fmin(loss_fn=f, space=s2, max_evals=40, trials=trials2)
    assert "Unknown distribution type for variable" in str(excinfo.value)

    s3 = {'x': {'dist': st.norm(loc=0., scale=1.)}}
    trials3 = []
    fmin(loss_fn=f, space=s3, max_evals=40, trials=trials3)
