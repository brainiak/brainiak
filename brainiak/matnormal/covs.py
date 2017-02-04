import tensorflow as tf
import numpy as np
import abc
from .helpers import define_scope, xx_t
from ..utils.utils import tf_solve_lower_triangular_kron,\
                          tf_solve_upper_triangular_kron


class CovBase:
    """Base metaclass for noise covariances
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, size):
        self.size = size

        # self.mask = tf.placeholder_with_default(tf.ones([self.size],dtype=tf.float64), [size], name="cov_mask")

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


class CovConstant(CovBase):
    """ Constant covariance
    """
    def __init__(self, Sigma):
        self.Sigma = tf.constant(Sigma)

        super(CovConstant, self).__init__(Sigma.shape[0])

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


class CovIdentity(CovBase):
    """Identity noise covariance.
    """
    def __init__(self, size):
        super(CovIdentity, self).__init__(size)

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


class CovIsotropic(CovBase):
    """Scaled identity (isotropic) noise covariance.
    """

    def __init__(self, size, sigma=None):
        super(CovIsotropic, self).__init__(size)
        if sigma is None:
            self.log_sigma = tf.Variable(tf.random_normal([1], dtype=tf.float64),
                                         name="sigma")
        else:
            self.log_sigma = tf.Variable(np.log(sigma), name="sigma")

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
        return self.size * self.log_sigma

    def Sigma_inv_x(self, X):
        """Given this Sigma and some X, compute :math:`Sigma^{-1} * x`
        """
        return X / self.sigma

    @define_scope
    def Sigma_inv(self):
        """ Sigma^{-1}.
        """
        return tf.diag(tf.ones([self.size], dtype=tf.float64)) / self.sigma


class CovDiagonal(CovBase):
    """Uncorrelated (diagonal) noise covariance
    """
    def __init__(self, size, sigma=None):
        super(CovDiagonal, self).__init__(size)
        if sigma is None:
            self.logprec = tf.Variable(tf.random_normal([size], dtype=tf.float64),
                                       name="precisions")
        else:
            self.logprec = tf.Variable(np.log(sigma), name="precisions")

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
        return -tf.reduce_sum(self.logprec)

    def Sigma_inv_x(self, X):
        """Given this Sigma and some X, compute :math:`Sigma^{-1} * x`
        """
        return tf.mul(self.prec_dimaugmented, X)

    @define_scope
    def Sigma_inv(self):
        """ Sigma^{-1}.
        """
        return tf.diag(tf.ones([self.size], dtype=tf.float64) * self.prec)


class CovFullRankCholesky(CovBase):
    """Full rank noise covariance parameterized in terms of its cholesky
    """

    def __init__(self, size, Sigma=None):
        super(CovFullRankCholesky, self).__init__(size)
        if Sigma is None:
            self.L_full = tf.Variable(tf.random_normal([size, size],
                                      dtype=tf.float64),
                                      name="L_full", dtype="float64")
        else:
            self.L_full = tf.Variable(np.linalg.cholesky(Sigma), name="L_full",
                                      dtype="float64")

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


class CovFullRankInvCholesky(CovBase):
    """Full rank noise covariance parameterized in terms of its precision cholesky
    """

    def __init__(self, size, invSigma=None):
        if invSigma is None:
            self.Linv_full = tf.Variable(tf.random_normal([size, size],
                                         dtype=tf.float64), name="Linv_full")
        else:
            self.Linv_full = tf.Variable(np.linalg.cholesky(invSigma), name="Linv_full")

        super(CovFullRankInvCholesky, self).__init__(size)

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


class CovKroneckerFactored(CovBase):
    """ Kronecker product noise covariance parameterized in terms
    of its component cholesky factors
    """

    def __init__(self, sizes):
        """Initialize the kronecker factored covariance object.

        Arguments
        ---------
        sizes : list
            List of dimensions (int) of the factors
            E.g. ``sizes = [2, 3]`` will create two factors of
            sizes 2x2 and 3x3 giving us a 6x6 dimensional covariance

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If sizes is not a list
        """
        if not isinstance(sizes, list):
            raise TypeError('sizes is not a list')

        self.sizes = sizes
        self.nfactors = len(sizes)
        self.size = np.prod(np.array(sizes), dtype=np.int32)
        self.L_full = [tf.Variable(tf.random_normal([sizes[i], sizes[i]],
                       dtype=tf.float64), name="L"+str(i)+"_full")
                       for i in range(self.nfactors)]

    @define_scope
    def L(self):
        """ Zero out triu of all factors in L_full to get cholesky L.
            This seems dumb but TF is smart enough to set the gradient to
            zero for those elements, and the alternative
            (fill_lower_triangular from contrib.distributions)
            is inefficient and recommends not doing the packing (for now).
            Also: to make the parameterization unique we log the diagonal
            so it's positive.
        """
        L_indeterminate = [tf.matrix_band_part(mat, -1, 0)
                           for mat in self.L_full]
        return [tf.matrix_set_diag(mat, tf.exp(tf.matrix_diag_part(mat)))
                for mat in L_indeterminate]

    def get_optimize_vars(self):
        """ Returns a list of tf variables that need to get optimized
            to fit this covariance
        """
        return self.L_full

    @define_scope
    def logdet(self):
        """ log|Sigma| using the diagonals of the cholesky factors.
        """
        n_list = tf.pack([tf.to_double(tf.shape(mat)[0]) for mat in self.L])
        n_prod = tf.reduce_prod(n_list)
        logdet = tf.pack([tf.reduce_sum(tf.log(tf.diag_part(mat)))
                 for mat in self.L])
        logdetfinal = tf.reduce_sum((logdet*n_prod)/n_list)
        return (2.0*logdetfinal)

    def Sigma_inv_x(self, X):
        """ Given this Sigma and some X, compute Sigma^{-1} * x using
        traingular solves with the cholesky factors.
        Do 2 triangular solves - L L^T x = y as L z = y and L^T x = z
        """
        z = tf_solve_lower_triangular_kron(self.L, X)
        x = tf_solve_upper_triangular_kron(self.L, z)
        return x
