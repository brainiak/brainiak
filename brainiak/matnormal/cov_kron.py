import tensorflow as tf
import numpy as np
from .helpers import define_scope
from .covs import CovBase


def tf_solve_lower_triangular_kron(L, y):
    '''
    Calculate L x = y where L = kron(L[0], L[1] .. L[n-1])
    and L[i] are the lower triangular matrices
    '''
    n = len(L)
    if n == 1:
        return tf.matrix_triangular_solve(L[0], y)
    else:
        x = y
        na = L[0].get_shape().as_list()[0]
        n_list = tf.pack([tf.to_double(tf.shape(mat)[0]) for mat in L])
        n_prod = tf.to_int32(tf.reduce_prod(n_list))
        nb = tf.to_int32(n_prod/na)
        col = tf.shape(x)[1]

        for i in range(na):
            xt, xinb, xina = tf.split_v(x, [i*nb, nb, (na-i-1)*nb], 0)
            t = xinb / L[0][i, i]
            xinb = tf_solve_lower_triangular_kron(L[1:], t)
            xina = xina - tf.reshape(tf.tile
                           (tf.slice(L[0], [i+1, i], [na-i-1, 1]),
                           [1, nb*col]), [(na-i-1)*nb, col]) * \
                           tf.reshape(tf.tile(tf.reshape
                           (t, [-1, 1]), [na-i-1, 1]), [(na-i-1)*nb, col])
            x = tf.concat(0, [xt, xinb, xina])

        return x


def tf_solve_upper_triangular_kron(L, y):
    '''
    Calculate L^T x = y where L = kron(L[0], L[1] .. L[n-1])
    and L[i] are the lower triangular matrices
    '''
    n = len(L)
    if n == 1:
        return tf.matrix_triangular_solve(L[0], y, adjoint=True)
    else:
        x = y
        na = L[0].get_shape().as_list()[0]
        n_list = tf.pack([tf.to_double(tf.shape(mat)[0]) for mat in L])
        n_prod = tf.to_int32(tf.reduce_prod(n_list))
        nb = tf.to_int32(n_prod/na)
        col = tf.shape(x)[1]

        for i in range(na-1, -1, -1):
            xt, xinb, xina = tf.split_v(x, [i*nb, nb, (na-i-1)*nb], 0)
            t = xinb / L[0][i, i]
            xinb = tf_solve_upper_triangular_kron(L[1:], t)
            xt = xt - tf.reshape(tf.tile(tf.transpose
                                 (tf.slice(L[0], [i, 0], [1, i])),
                                 [1, nb*col]), [i*nb, col]) * \
                                 tf.reshape(tf.tile(tf.reshape
                                 (t, [-1, 1]), [i, 1]), [i*nb, col])
            x = tf.concat(0, [xt, xinb, xina])

        return x


class CovKroneckerFactored(CovBase):
    """
    Kronecker product noise covariance parameterized in terms
    of its component cholesky factors
    """

    def __init__(self, sizes):
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
        """
        Given this Sigma and some X, compute Sigma^{-1} * x using
        traingular solves with the cholesky factors.
        Do 2 triangular solves - L L^T x = y as L z = y and L^T x = z
        """
        z = tf_solve_lower_triangular_kron(self.L, X)
        x = tf_solve_upper_triangular_kron(self.L, z)
        return x
