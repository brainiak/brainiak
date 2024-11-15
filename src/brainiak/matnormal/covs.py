import tensorflow as tf
import numpy as np
import abc
import scipy.linalg
import scipy.sparse
import tensorflow_probability as tfp

from brainiak.matnormal.utils import (
    x_tx,
    xx_t,
    unflatten_cholesky_unique,
    flatten_cholesky_unique,
)
from brainiak.utils.kronecker_solvers import (
    tf_solve_lower_triangular_kron,
    tf_solve_upper_triangular_kron,
    tf_solve_lower_triangular_masked_kron,
    tf_solve_upper_triangular_masked_kron,
)

__all__ = [
    "CovBase",
    "CovIdentity",
    "CovAR1",
    "CovIsotropic",
    "CovDiagonal",
    "CovDiagonalGammaPrior",
    "CovUnconstrainedCholesky",
    "CovUnconstrainedCholeskyWishartReg",
    "CovUnconstrainedInvCholesky",
    "CovKroneckerFactored",
]


class CovBase(abc.ABC):
    """Base metaclass for residual covariances.
    For more on abstract classes, see
    https://docs.python.org/3/library/abc.html

    Parameters
    ----------

    size: int
        The size of the covariance matrix.

    """

    def __init__(self, size):
        self.size = size

        # Log-likelihood of this covariance (useful for regularization)
        self.logp = tf.constant(0, dtype=tf.float64)

    @abc.abstractmethod
    def get_optimize_vars(self):
        """ Returns a list of tf variables that need to get optimized to fit
            this covariance
        """
        pass

    @property
    def logdet(self):
        """ log determinant of this covariance
        """
        pass

    @abc.abstractmethod
    def solve(self, X):
        """Given this covariance :math:`\\Sigma` and some input :math:`X`,
            compute :math:`\\Sigma^{-1}x`
        """
        pass

    @property
    def _prec(self):
        """Expose the precision explicitly (mostly for testing /
        visualization, materializing large covariances may be intractable)
        """
        return self.solve(tf.eye(self.size, dtype=tf.float64))

    @property
    def _cov(self):
        """Expose the covariance explicitly (mostly for testing /
        visualization, materializing large covariances may be intractable)
        """
        return tf.linalg.inv(self._prec)


class CovIdentity(CovBase):
    """Identity noise covariance.
    """

    def __init__(self, size):
        super(CovIdentity, self).__init__(size)

    @property
    def logdet(self):
        return tf.constant(0.0, "float64")

    def get_optimize_vars(self):
        """Returns a list of tf variables that need to get optimized to
            fit this covariance
        """
        return []

    def solve(self, X):
        """Given this covariance :math:`\\Sigma` and some input :math:`X`,
            compute :math:`\\Sigma^{-1}x`
        """
        return X

    @property
    def _prec(self):
        """Expose the precision explicitly (mostly for testing /
        visualization, materializing large covariances may be intractable)
        """
        return tf.eye(self.size, dtype=tf.float64)

    @property
    def _cov(self):
        """Expose the covariance explicitly (mostly for testing /
        visualization, materializing large covariances may be intractable)
        """
        return tf.eye(self.size, dtype=tf.float64)


class CovAR1(CovBase):
    """AR(1) covariance parameterized by autoregressive parameter rho
    and new noise sigma.

    Parameters
    ----------
    size: int
        size of covariance matrix
    rho: float or None
        initial value of autoregressive parameter (if None, initialize
        randomly)
    sigma: float or None
        initial value of new noise parameter (if None, initialize randomly)

    """

    def __init__(self, size, rho=None, sigma=None, scan_onsets=None):

        super(CovAR1, self).__init__(size)

        # Similar to BRSA trick I think
        if scan_onsets is None:
            self.run_sizes = [size]
            self.offdiag_template = tf.constant(
                scipy.linalg.toeplitz(np.r_[0, 1, np.zeros(size - 2)]),
                dtype=tf.float64
            )
            self.diag_template = tf.constant(
                np.diag(np.r_[0, np.ones(size - 2), 0]))
        else:
            self.run_sizes = np.ediff1d(np.r_[scan_onsets, size])
            sub_offdiags = [
                scipy.linalg.toeplitz(np.r_[0, 1, np.zeros(r - 2)])
                for r in self.run_sizes
            ]
            self.offdiag_template = tf.constant(
                scipy.sparse.block_diag(sub_offdiags).toarray()
            )
            subdiags = [np.diag(np.r_[0, np.ones(r - 2), 0])
                        for r in self.run_sizes]
            self.diag_template = tf.constant(
                scipy.sparse.block_diag(subdiags).toarray()
            )

        self._identity_mat = tf.constant(np.eye(size))

        if sigma is None:
            self.log_sigma = tf.Variable(
                tf.random.normal([1], dtype=tf.float64), name="log_sigma"
            )
        else:
            self.log_sigma = tf.Variable(np.log(sigma), name="log_sigma")

        if rho is None:
            self.rho_unc = tf.Variable(
                tf.random.normal([1], dtype=tf.float64), name="rho_unc"
            )
        else:
            self.rho_unc = tf.Variable(
                scipy.special.logit(rho / 2 + 0.5), name="rho_unc"
            )

    @property
    def logdet(self):
        """ log-determinant of this covariance
        """
        # first, unconstrain rho and sigma
        rho = 2 * tf.sigmoid(self.rho_unc) - 1
        # now compute logdet
        return tf.reduce_sum(
            input_tensor=2
            * tf.constant(self.run_sizes, dtype=tf.float64)
            * self.log_sigma
            - tf.math.log(1 - tf.square(rho))
        )

    @property
    def _prec(self):
        """Precision matrix corresponding to this AR(1) covariance.
        We assume stationarity within block so no special case
        for first/last element of a block. This makes constructing this
        matrix easier.
        reprsimil.BRSA says (I - rho1 * D + rho1**2 * F) / sigma**2 and we
        use the same trick
        """
        rho = 2 * tf.sigmoid(self.rho_unc) - 1
        sigma = tf.exp(self.log_sigma)

        return (
            self._identity_mat
            - rho * self.offdiag_template
            + rho ** 2 * self.diag_template
        ) / tf.square(sigma)

    def get_optimize_vars(self):
        """ Returns a list of tf variables that need to get optimized to
        fit this covariance
        """
        return [self.rho_unc, self.log_sigma]

    def solve(self, X):
        """Given this covariance :math:`\\Sigma` and some input :math:`X`,
        compute :math:`\\Sigma^{-1}x`
        """
        return tf.matmul(self._prec, X)


class CovIsotropic(CovBase):
    """Scaled identity (isotropic) noise covariance.

    Parameters
    ----------
    size: int
        size of covariance matrix
    var: float or None
        initial value of new variance parameter (if None, initialize randomly)

    """

    def __init__(self, size, var=None):
        super(CovIsotropic, self).__init__(size)
        if var is None:
            self.log_var = tf.Variable(
                tf.random.normal([1], dtype=tf.float64), name="sigma"
            )
        else:
            self.log_var = tf.Variable(np.log(var), name="sigma")
        self.var = tf.exp(self.log_var)

    @property
    def logdet(self):
        return self.size * self.log_var

    def get_optimize_vars(self):
        """ Returns a list of tf variables that need to get optimized to fit
            this covariance
        """
        return [self.log_var]

    def solve(self, X):
        """Given this covariance :math:`\\Sigma` and some input :math:`X`,
        compute :math:`\\Sigma^{-1}x`

        Parameters
        ----------
        X: tf.Tensor
            Tensor to multiply by inverse of this covariance

        """
        return X / self.var


class CovDiagonal(CovBase):
    """Uncorrelated (diagonal) noise covariance

    Parameters
    ----------
    size: int
        size of covariance matrix
    diag_var: float or None
        initial value of (diagonal) variance vector (if None, initialize
        randomly)

    """

    def __init__(self, size, diag_var=None):
        super(CovDiagonal, self).__init__(size)
        if diag_var is None:
            self.logprec = tf.Variable(
                tf.random.normal([size], dtype=tf.float64), name="precisions"
            )
        else:
            self.logprec = tf.Variable(
                np.log(1 / diag_var), name="log-precisions")

    @property
    def logdet(self):
        return -tf.reduce_sum(input_tensor=self.logprec)

    def get_optimize_vars(self):
        """ Returns a list of tf variables that need to get optimized to fit
            this covariance
        """
        return [self.logprec]

    def solve(self, X):
        """Given this covariance :math:`\\Sigma` and some input :math:`X`,
        compute :math:`\\Sigma^{-1}x`

        Parameters
        ----------
        X: tf.Tensor
            Tensor to multiply by inverse of this covariance

        """
        prec = tf.exp(self.logprec)
        prec_dimaugmented = tf.expand_dims(prec, -1)
        return tf.multiply(prec_dimaugmented, X)


class CovDiagonalGammaPrior(CovDiagonal):
    """Uncorrelated (diagonal) noise covariance
    """

    def __init__(self, size, sigma=None, alpha=1.5, beta=1e-10):
        super(CovDiagonalGammaPrior, self).__init__(size, sigma)

        self.ig = tfp.distributions.InverseGamma(
            concentration=tf.constant(alpha, dtype=tf.float64),
            scale=tf.constant(beta, dtype=tf.float64),
        )

        self.logp = tf.reduce_sum(
            input_tensor=self.ig.log_prob(tf.exp(self.logprec)))


class CovUnconstrainedCholesky(CovBase):
    """Unconstrained noise covariance parameterized in terms of its cholesky
    """

    def __init__(self, size=None, Sigma=None):

        if size is None and Sigma is None:
            raise RuntimeError("Must pass either Sigma or size but not both")

        if size is not None and Sigma is not None:
            raise RuntimeError("Must pass either Sigma or size but not both")

        if Sigma is not None:
            size = Sigma.shape[0]

        super(CovUnconstrainedCholesky, self).__init__(size)

        # number of parameters in the triangular mat
        npar = (size * (size + 1)) // 2

        if Sigma is None:
            self.L_flat = tf.Variable(
                tf.random.normal([npar], dtype=tf.float64), name="L_flat"
            )

        else:
            L = np.linalg.cholesky(Sigma)
            self.L_flat = tf.Variable(
                flatten_cholesky_unique(L), name="L_flat")

        self.optimize_vars = [self.L_flat]

    @property
    def L(self):
        """
        Cholesky factor of this covariance
        """
        return unflatten_cholesky_unique(self.L_flat)

    @property
    def logdet(self):
        return 2 * tf.reduce_sum(input_tensor=tf.math.log(
                                 tf.linalg.diag_part(self.L)))

    def get_optimize_vars(self):
        """ Returns a list of tf variables that need to get optimized to fit
             this covariance
        """
        return [self.L_flat]

    def solve(self, X):
        """Given this covariance :math:`\\Sigma` and some input :math:`X`,
        compute :math:`\\Sigma^{-1}x` (using cholesky solve)

        Parameters
        ----------
        X: tf.Tensor
            Tensor to multiply by inverse of this covariance

        """
        return tf.linalg.cholesky_solve(self.L, X)


class CovUnconstrainedCholeskyWishartReg(CovUnconstrainedCholesky):
    """Unconstrained noise covariance parameterized in terms of its
    cholesky factor. Regularized using the trick from
    Chung et al. 2015 such that as the covariance approaches
    singularity, the likelihood goes to 0.

    References
    ----------
    Chung, Y., Gelman, A., Rabe-Hesketh, S., Liu, J., & Dorie, V. (2015).
    Weakly Informative Prior for Point Estimation of Covariance Matrices
    in Hierarchical Models. Journal of Educational and Behavioral Statistics,
    40(2), 136–157. https://doi.org/10.3102/1076998615570945
    """

    def __init__(self, size, Sigma=None):
        super(CovUnconstrainedCholeskyWishartReg, self).__init__(size)
        self.wishartReg = tfp.distributions.WishartTriL(
            df=tf.constant(size + 2, dtype=tf.float64),
            scale_tril=tf.constant(1e5 * np.eye(size), dtype=tf.float64),
        )

        Sigma = xx_t(self.L)
        self.logp = self.wishartReg.log_prob(Sigma)


class CovUnconstrainedInvCholesky(CovBase):
    """Unconstrained noise covariance parameterized
       in terms of its precision cholesky. Use this over the
       regular cholesky unless you have a good reason not to, since
       this saves a cholesky solve on every step of optimization
    """

    def __init__(self, size=None, invSigma=None):

        if size is None and invSigma is None:
            raise RuntimeError(
                "Must pass either invSigma or size but not both")

        if size is not None and invSigma is not None:
            raise RuntimeError(
                "Must pass either invSigma or size but not both")

        if invSigma is not None:
            size = invSigma.shape[0]

        super(CovUnconstrainedInvCholesky, self).__init__(size)

        # number of parameters in the triangular mat
        npar = (size * (size + 1)) // 2

        if invSigma is None:
            self.Linv_flat = tf.Variable(
                tf.random.normal([npar], dtype=tf.float64), name="Linv_flat"
            )

        else:
            Linv = np.linalg.cholesky(invSigma)
            self.Linv_flat = tf.Variable(
                flatten_cholesky_unique(Linv), name="Linv_flat"
            )

    @property
    def Linv(self):
        """
        Inverse of Cholesky factor of this covariance
        """
        return unflatten_cholesky_unique(self.Linv_flat)

    @property
    def logdet(self):
        return -2 * tf.reduce_sum(
            input_tensor=tf.math.log(tf.linalg.diag_part(self.Linv))
        )

    def get_optimize_vars(self):
        """ Returns a list of tf variables that need to get optimized to fit
            this covariance
        """
        return [self.Linv_flat]

    def solve(self, X):
        """Given this covariance :math:`\\Sigma` and some input :math:`X`,
        compute :math:`\\Sigma^{-1}x` (using cholesky solve)

        Parameters
        ----------
        X: tf.Tensor
            Tensor to multiply by inverse of this covariance

        """
        return tf.matmul(x_tx(self.Linv), X)


class CovKroneckerFactored(CovBase):
    """ Kronecker product noise covariance parameterized in terms
    of its component cholesky factors
    """

    def __init__(self, sizes, Sigmas=None, mask=None):
        """Initialize the kronecker factored covariance object.

        Arguments
        ---------
        sizes : list
            List of dimensions (int) of the factors
            E.g. ``sizes = [2, 3]`` will create two factors of
            sizes 2x2 and 3x3 giving us a 6x6 dimensional covariance
        Sigmas : list (default : None)
            Initial guess for the covariances. List of positive definite
            covariance matrices the same sizes as sizes.
        mask : int array (default : None)
            1-D tensor with length equal to product of sizes with 1 for
            valid elements and 0 for don't care

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If sizes is not a list
        """
        if not isinstance(sizes, list):
            raise TypeError("sizes is not a list")

        self.sizes = sizes
        self.nfactors = len(sizes)
        self.size = np.prod(np.array(sizes), dtype=np.int32)

        npar = [(size * (size + 1)) // 2 for size in self.sizes]
        if Sigmas is None:
            self.Lflat = [
                tf.Variable(
                    tf.random.normal([npar[i]], dtype=tf.float64),
                    name="L" + str(i) + "_flat",
                )
                for i in range(self.nfactors)
            ]
        else:
            self.Lflat = [
                tf.Variable(
                    flatten_cholesky_unique(np.linalg.cholesky(Sigmas[i])),
                    name="L" + str(i) + "_flat",
                )
                for i in range(self.nfactors)
            ]
        self.mask = mask

    @property
    def L(self):
        return [unflatten_cholesky_unique(mat) for mat in self.Lflat]

    def get_optimize_vars(self):
        """ Returns a list of tf variables that need to get optimized
            to fit this covariance
        """
        return self.Lflat

    @property
    def logdet(self):
        """ log|Sigma| using the diagonals of the cholesky factors.
        """
        if self.mask is None:
            n_list = tf.stack(
                [tf.cast(tf.shape(input=mat)[0], dtype=tf.float64)
                 for mat in self.L]
            )
            n_prod = tf.reduce_prod(input_tensor=n_list)
            logdet = tf.stack(
                [
                    tf.reduce_sum(
                        input_tensor=tf.math.log(
                            tf.linalg.tensor_diag_part(mat))
                    )
                    for mat in self.L
                ]
            )
            logdetfinal = tf.reduce_sum(
                input_tensor=(logdet * n_prod) / n_list)
        else:
            n_list = [tf.shape(input=mat)[0] for mat in self.L]
            mask_reshaped = tf.reshape(self.mask, n_list)
            logdet = 0.0
            for i in range(self.nfactors):
                indices = list(range(self.nfactors))
                indices.remove(i)
                logdet += (tf.math.log(tf.linalg.tensor_diag_part(self.L[i])) *
                           tf.cast(
                            tf.reduce_sum(
                                input_tensor=mask_reshaped, axis=indices),
                           dtype=tf.float64,
                           ))
            logdetfinal = tf.reduce_sum(input_tensor=logdet)
        return 2.0 * logdetfinal

    def solve(self, X):
        """ Given this covariance :math:`\\Sigma` and some input :math:`X`,
        compute :math:`\\Sigma^{-1}x` using traingular solves with the cholesky
        factors.

        Specifically, we solve :math:`L L^T x = y` by solving
        :math:`L z = y` and :math:`L^T x = z`.

        Parameters
        ----------
        X: tf.Tensor
            Tensor to multiply by inverse of this covariance

        """
        if self.mask is None:
            z = tf_solve_lower_triangular_kron(self.L, X)
            x = tf_solve_upper_triangular_kron(self.L, z)
        else:
            z = tf_solve_lower_triangular_masked_kron(self.L, X, self.mask)
            x = tf_solve_upper_triangular_masked_kron(self.L, z, self.mask)
        return x
