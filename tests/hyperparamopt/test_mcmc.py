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

def test_get_chain():
    from brainiak.hyperparamopt.mcmc import get_multichain_samples
    import numpy as np
    import numpy.testing as npt
    import scipy.stats as st

    def normal(mean, std):
        def f(x):
          return st.norm.pdf(x, loc=mean, scale=std)
        return f

    for mean,std in [(0.,1.), (2.,4.), (-5., 3.)]:
        p = normal(mean, std)
        samples = get_multichain_samples(1000, p, nchains=5)
        assert(np.abs(mean - np.mean(samples)) <= 1.)
        assert(np.abs(std - np.std(samples)) <= 1.)
        # assert(st.skewtest(samples).pvalue >= 0.05)
        # assert(st.kurtosistest(samples).pvalue >= 0.05)


    """
    #import pylab as plt
    #plt.hist(samples, 100)
    #plt.show()
    # Anderson-Darling test
    A2, criticalvalues, significancelevel = st.anderson(samples, 'norm')
    print(criticalvalues, significancelevel, A2)

    # critical values at [15, 10, 5, 2.5, 1]
    for i in range(len(significancelevel)):
        if (significancelevel[i] == 5.):  # at 5% significance level 
            assert(A2 <= criticalvalues[i])
    """
