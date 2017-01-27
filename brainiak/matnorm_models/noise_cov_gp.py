from .noise_covs import NoiseCovBase
import tensorflow as tf
from .helpers import define_scope

class NoiseCovGP(NoiseCovBase):
    """Wrapper for GPflow kernels that lets us use them. Also exposes methods for constructing
    kernels between arbitrary X's which we need for prediction with trial-by-trial noise
    """

    def __init__(self, kern, input_dim, **kwargs): 
        """ Pass in a gpflow.kernels kernel
        """
        self.kern = kern(input_dim = input_dim, **kwargs)
        
        # self.loc = tf.placeholder(tf.float64, [input_dim, None], name="loc")

        self.kern._tf_mode = True

        self._opt_var = tf.Variable(self.kern.get_free_state(), dtype=tf.float64)

        self.kern.make_tf_array(self._opt_var)


    def get_optimize_vars(self):
        """ Returns a list of tf variables that need to get optimized to fit this covariance
        """
        return [self._opt_var]
    
    @define_scope 
    def L(self):
        return tf.cholesky(self.kern.K(self.loc))

    @define_scope
    def logdet(self, loc = None):
        """ log|Sigma|
        """
        if loc is None: 
            return 2 * tf.reduce_sum(tf.log(tf.matrix_diag_part(self.L)))
        else: 
            return 2 * tf.reduce_sum(tf.log(tf.matrix_diag_part(tf.cholesky(self.kern.K(loc)))))

    def Sigma_inv_x(self, X, loc = None):
        """Given this Sigma and some X, compute Sigma^{-1} * x
        """
        if loc is None: 
            return tf.cholesky_solve(self.L, X)
        else:
            return tf.cholesky_solve(tf.cholesky(self.kern.K(loc)), X)

    def cov_dist(self, X, Y):
        return self.kern.K(X, Y)

