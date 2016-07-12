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
"""Cython file for optimizing GMM likelihood computation

"""

# Authors: Narayanan Sundaram (Intel Labs)

#cython: embedsignature=True
import numpy
cimport numpy as np

cdef extern from "math.h":
    double exp(double x)
    double sqrt(double x)


pi = numpy.pi


cpdef double norm_pdf(double x, double mu, double sigma):
    """Calculate Gaussian pdf

    Given x, returns exp(-0.5*z*z)/(sigma*sqrt(2.*pi)) where
    z = (x-mu)/sigma
    """

    cdef double z
    z = (x-mu)/sigma
    return exp(-0.5*z*z)/sqrt(2.0*pi)/sigma


cpdef double getgmmpdf(double xt,np.ndarray[np.float64_t, ndim=1] x, np.ndarray[np.float64_t, ndim=1] sigma, np.ndarray[np.float64_t, ndim=1] weights, double minlimit, double maxlimit):
    """Calculates the 1D GMM likelihood

    y = \sum_{i=1}^{N} norm_pdf(x, x_i, sigma_i)/(\sum weight_i)
    """

    cdef double y
    cdef double w
    w = sum(weights)
    y = 0
    if (xt < minlimit):
        return 0
    if (xt > maxlimit):
        return 0
    for _x in range(x.size):
        y += norm_pdf(xt, x[_x], sigma[_x])*weights[_x]/w
    return y
