import tensorflow as tf
import numpy as np
import abc
from .helpers import define_scope, xx_t


class NoiseCovBase:
    """Base metaclass for noise covariances
    """
    __metaclass__ = abc.ABCMeta

    loc = tf.placeholder(tf.float64, [None, None], name="positions")
    mask = tf.placeholder(tf.float64, [None, None], name="cov_mask")

    @abc.abstractmethod
    def get_optimize_vars(self):
        """ Returns a list of tf variables that need to get optimized to fit
            this covariance
        """
        pass

    @abc.abstractproperty
    def logdet(self):
        """ log|Sigma|
        """
        pass

    @abc.abstractmethod
    def Sigma_inv_x(self, X):
        """Given this Sigma and some X, compute :math:`Sigma^{-1} * x`
        """
        pass

    @define_scope
    def Sigma_inv(self):
        """ Sigma^{-1}. Override me with more efficient implementation in subclasses
        """
        return self.Sigma_inv_x(tf.diag(tf.ones([self.size],
                                dtype=tf.float64)))


class NoiseCovConstant(NoiseCovBase):

    def __init__(self, Sigma):
        self.Sigma = tf.constant(Sigma)
        self.size = Sigma.shape[0]
        self.L = tf.constant(np.linalg.cholesky(Sigma))

    def get_optimize_vars(self):
        """ Returns a list of tf variables that need to get optimized to fit
            this covariance
        """
        return []

    def Sigma_inv_x(self, X):
        """Given this Sigma and some X, compute :math:`Sigma^{-1} * x`
        """
        return tf.cholesky_solve(self.L, X)

    @define_scope
    def logdet(self):
        return 2 * tf.reduce_sum(tf.log(tf.matrix_diag_part(self.L)))


class NoiseCovIdentity(NoiseCovBase):
    """Identity noise covariance.
    """
    def __init__(self, size):
        self.size = size

    def get_optimize_vars(self):
        """ Returns a list of tf variables that need to get optimized to
            fit this covariance
        """
        return []

    @define_scope
    def logdet(self):
        """ log|Sigma|
        """
        return tf.constant(0.0, 'float64')

    def Sigma_inv_x(self, X):
        """Given this Sigma and some X, compute :math:`Sigma^{-1} * x`
        """
        return X

    @define_scope
    def Sigma_inv(self):
        """ Sigma^{-1}.
        """
        return tf.diag(tf.ones([self.size], dtype=tf.float64))


class NoiseCovIsotropic(NoiseCovBase):
    """Scaled identity (isotropic) noise covariance.
    """

    def __init__(self, size):
        self.size = size
        self.log_sigma = tf.Variable(tf.random_normal([1], dtype=tf.float64),
                                     name="sigma")

    @define_scope
    def sigma(self):
        return tf.exp(self.log_sigma)

    def get_optimize_vars(self):
        """ Returns a list of tf variables that need to get optimized to fit
            this covariance
        """
        return [self.log_sigma]

    @define_scope
    def logdet(self):
        """ log|Sigma|
        """
        return self.size * tf.log(self.sigma)

    def Sigma_inv_x(self, X):
        """Given this Sigma and some X, compute :math:`Sigma^{-1} * x`
        """
        return X / self.sigma

    @define_scope
    def Sigma_inv(self):
        """ Sigma^{-1}.
        """
        return tf.diag(tf.ones([self.size], dtype=tf.float64)) / self.sigma


class NoiseCovDiagonal(NoiseCovBase):
    """Uncorrelated (diagonal) noise covariance
    """
    def __init__(self, size):
        self.size = size
        self.logprec = tf.Variable(tf.random_normal([size], dtype=tf.float64),
                                   name="precisions")

    @define_scope
    def prec(self):
        return tf.exp(self.logprec)

    @define_scope
    def prec_dimaugmented(self):
        return tf.expand_dims(self.prec, -1)

    def get_optimize_vars(self):
        """ Returns a list of tf variables that need to get optimized to fit
            this covariance
        """
        return [self.logprec]

    @define_scope
    def logdet(self):
        """ log|Sigma|
        """
        return -tf.reduce_sum(tf.log(self.prec))

    def Sigma_inv_x(self, X):
        """Given this Sigma and some X, compute :math:`Sigma^{-1} * x`
        """
        return tf.mul(self.prec_dimaugmented, X)

    @define_scope
    def Sigma_inv(self):
        """ Sigma^{-1}.
        """
        return tf.diag(tf.ones([self.size], dtype=tf.float64) * self.prec)


class NoiseCovFullRank(NoiseCovBase):
    """Full rank noise covariance parameterized in terms of its cholesky
    """

    def __init__(self, size):
        self.L_full = tf.Variable(tf.random_normal([size, size],
                                  dtype=tf.float64),
                                  name="L_full", dtype="float64")
        self.size = size

    @define_scope
    def L(self):
        """ Zero out triu of L_full to get cholesky L.
            This seems dumb but TF is smart enough to set the gradient to zero
            for those elements, and the alternative (fill_lower_triangular from
            contrib.distributions) is inefficient and recommends not doing the
            packing (for now).
            Also: to make the parameterization unique we log the diagonal so
            it's positive.
        """
        L_indeterminate = tf.matrix_band_part(self.L_full, -1, 0)
        return tf.matrix_set_diag(L_indeterminate,
                                  tf.exp(tf.matrix_diag_part(L_indeterminate)))

    @define_scope
    def Sigma(self):
        """ covariance
        """
        return xx_t(self.L)

    def get_optimize_vars(self):
        """ Returns a list of tf variables that need to get optimized to fit
             this covariance
        """
        return [self.L_full]

    @define_scope
    def logdet(self):
        """ log|Sigma| using a cholesky solve
        """
        return 2 * tf.reduce_sum(tf.log(tf.matrix_diag_part(self.L)))

    def Sigma_inv_x(self, X):
        """
        Given this Sigma and some X, compute :math:`Sigma^{-1} * x` using
        cholesky solve
        """
        return tf.cholesky_solve(self.L, X)


class NoisePrecFullRank(NoiseCovBase):
    """Full rank noise covariance parameterized in terms of its precision cholesky
    """

    def __init__(self, size):
        self.Linv_full = tf.Variable(tf.random_normal([size, size],
                                     dtype=tf.float64), name="L_full")
        self.size = size

    @define_scope
    def Linv(self):
        """ Zero out triu of L_full to get cholesky L.
            This seems dumb but TF is smart enough to set the gradient to zero
            for those elements, and the alternative (fill_lower_triangular from
            contrib.distributions) is inefficient and recommends not doing the
            packing (for now).
            Also: to make the parameterization unique we log the diagonal so
            it's positive.
        """
        L_indeterminate = tf.matrix_band_part(self.Linv_full, -1, 0)
        return tf.matrix_set_diag(L_indeterminate,
                                  tf.exp(tf.matrix_diag_part(L_indeterminate)))

    @define_scope
    def Sigma(self):
        """ cov
        """
        return tf.matrix_inverse(self.Sigma_inv)

    def get_optimize_vars(self):
        """ Returns a list of tf variables that need to get optimized to fit
            this covariance
        """
        return [self.Linv_full]

    @define_scope
    def logdet(self):
        """ log|Sigma| using a cholesky solve
        """
        return -2 * tf.reduce_sum(tf.log(tf.matrix_diag_part(self.Linv)))

    def Sigma_inv_x(self, X):
        """
        Given this Sigma and some X, compute :math:`Sigma^{-1} * x` using
        cholesky solve
        """
        return tf.matmul(xx_t(self.Linv), X)
