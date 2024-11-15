import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import norm
from numpy.linalg import cholesky
import numpy as np


def rmn(rowcov, colcov):
    """
    Generate random draws from a zero-mean matrix-normal distribution.

    Parameters
    -----------
    rowcov : np.ndarray
        Row covariance (assumed to be positive definite)
    colcov : np.ndarray
        Column covariance (assumed to be positive definite)
    """

    Z = norm.rvs(size=(rowcov.shape[0], colcov.shape[0]))
    return cholesky(rowcov).dot(Z).dot(cholesky(colcov))


def xx_t(x):
    """
    Outer product
    :math:`xx^T`

    Parameters
    -----------
    x : tf.Tensor

    """
    return tf.matmul(x, x, transpose_b=True)


def x_tx(x):
    """Inner product
    :math:`x^T x`

    Parameters
    -----------
    x : tf.Tensor

    """
    return tf.matmul(x, x, transpose_a=True)


def scaled_I(x, size):
    """Scaled identity matrix
    :math:`x I_{size}`

    Parameters
    ------------
    x: float or coercable to float
        Scale to multiply the identity matrix by
    size: int or otherwise coercable to a size
        Dimension of the scaled identity matrix to return
    """
    return tf.linalg.tensor_diag(tf.ones([size], dtype=tf.float64) * x)


def flatten_cholesky_unique(L):
    """
    Flattens nonzero-elements Cholesky (triangular) factor
    into a vector, and logs diagonal to make parameterization
    unique. Inverse of unflatten_cholesky_unique.
    """
    L_tf = tf.linalg.set_diag(L, tf.math.log(tf.linalg.diag_part(L)))
    L_flat = tfp.math.fill_triangular_inverse(L_tf)
    return L_flat


def unflatten_cholesky_unique(L_flat):
    """
    Converts a vector of elements into a triangular matrix
    (Cholesky factor). Exponentiates diagonal to make
    parameterization unique. Inverse of flatten_cholesky_unique.
    """
    L = tfp.math.fill_triangular(L_flat)
    # exp diag for unique parameterization
    L = tf.linalg.set_diag(L, tf.exp(tf.linalg.diag_part(L)))
    return L


def pack_trainable_vars(trainable_vars):
    """
    Pack trainable vars in a model into a single
    vector that can be passed to scipy.optimize
    """
    return tf.concat([tf.reshape(tv, (-1,)) for tv in trainable_vars], axis=0)


def unpack_trainable_vars(x, trainable_vars):
    """
    Unpack trainable vars from a single vector as
    used/returned by scipy.optimize
    """

    sizes = [tv.shape for tv in trainable_vars]
    idxs = [np.prod(sz) for sz in sizes]
    flatvars = tf.split(x, idxs)
    return [tf.reshape(fv, tv.shape) for fv, tv in zip(flatvars,
                                                       trainable_vars)]


def make_val_and_grad(lossfn, train_vars):
    """
    Makes a function that ouptuts the loss and gradient in a format compatible
    with scipy.optimize.minimize
    """

    def val_and_grad(theta):
        with tf.GradientTape() as tape:
            tape.watch(train_vars)
            unpacked_theta = unpack_trainable_vars(theta, train_vars)
            for var, val in zip(train_vars, unpacked_theta):
                var.assign(val)
            loss = lossfn(theta)
        grad = tape.gradient(loss, train_vars)
        packed_grad = pack_trainable_vars(grad)
        return loss.numpy(), packed_grad.numpy()

    return val_and_grad
