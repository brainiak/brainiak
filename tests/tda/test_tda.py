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

from brainiak import tda
import numpy as np

def test_preprocess():
    n = 10
    input = np.ones(n * n).reshape(n, n)

    output = tda.preprocess(input)
    assert sum(sum(output == 1)) == n * n, "Preprocessing failed"

    output = tda.preprocess(input, t_score=True)
    assert sum(sum(output == 1)) == n * n, "t_score failed"

    output = tda.preprocess(np.random.randn(n * n).reshape(n, n), gauss_size=100)
    assert np.std(output) < 0.1, "Smoothing failed"

    input = np.random.rand(n * n).reshape(n, n)
    output = tda.preprocess(input, normalize=True)
    assert abs(np.mean(output)) < 0.1, "Normalizing failed"


def test_convert_space():
    n = 10
    Input = np.repeat(list(range(0, n)), n, axis=0).reshape(n,n).T

    Output = tda.convert_space(Input)
    assert sum(sum(Output>=0.99999)) == n*n, "Correlation failed"

    Output = tda.convert_space(Input,
                               selectionfunc=tda.SelectionFuncs.variance_score)
    assert sum(sum(Output >= 0.99999)) == n * n, "Variance failed"

    Output = tda.convert_space(Input,
                               distancefunc=tda.DistanceFuncs.compute_euclidean_distance)
    assert sum(sum(Output == 0)) == n * n, "Euclidean distance failed"

    Output = tda.convert_space(Input, run_mds=1, dimensions=3)
    assert Output.shape[1] == 3, "MDS failed"

