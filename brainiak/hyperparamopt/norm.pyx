#cython: embedsignature=True

"""Cython file for optimizing GMM likelihood computation

"""

# Authors: Narayanan Sundaram (Intel Labs)

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
