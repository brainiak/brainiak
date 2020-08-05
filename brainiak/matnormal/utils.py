import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import norm
from numpy.linalg import cholesky
import numpy as np


def rmn(rowcov, colcov):
    # generate random draws from a zero-mean matrix-normal distribution
    Z = norm.rvs(size=(rowcov.shape[0], colcov.shape[0]))
    return cholesky(rowcov).dot(Z).dot(cholesky(colcov))


def xx_t(x):
    """ x * x' """
    return tf.matmul(x, x, transpose_b=True)


def x_tx(x):
    """ x' * x """
    return tf.matmul(x, x, transpose_a=True)


def quad_form(x, y):
    """ x' * y * x """
    return tf.matmul(x, tf.matmul(y, x), transpose_a=True)


def scaled_I(x, size):
    """ x * I_{size} """
    return tf.linalg.tensor_diag(tf.ones([size], dtype=tf.float64) * x)


def quad_form_trp(x, y):
    """ x * y * x' """
    return tf.matmul(x, tf.matmul(y, x, transpose_b=True))


def flatten_cholesky_unique(L):
    """
    Flattens nonzero-elements Cholesky (triangular) factor
    into a vector, and logs diagonal to make parameterizaation
    unique. Inverse of unflatten_cholesky_unique.
    """
    L[np.diag_indices_from(L)] = np.log(np.diag(L))
    L_flat = tfp.math.fill_triangular_inverse(L)
    return L_flat


def unflatten_cholesky_unique(L_flat):
    """
    Converts a vector of elements into a triangular matrix 
    (Cholesky factor). Exponentiates diagonal to make
    parameterizaation unique. Inverse of flatten_cholesky_unique. 
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
    return [tf.reshape(fv, tv.shape) for fv, tv in zip(flatvars, trainable_vars)]


def make_val_and_grad(model, lossfn=None, extra_args=None, train_vars=None):

    if train_vars is None:
        train_vars = model.train_variables

    if lossfn is None:
        lossfn = lambda theta: -model.logp(*extra_args)

    if extra_args is None:
        extra_args = {}

    def val_and_grad(theta, *extra_args):
        with tf.GradientTape() as tape:
            tape.watch(train_vars)
            unpacked_theta = unpack_trainable_vars(theta, train_vars)
            for var, val in zip(train_vars, unpacked_theta):
                var = val
            loss = lossfn(theta)
        grad = tape.gradient(loss, train_vars)
        packed_grad = pack_trainable_vars(grad)
        return loss.numpy(), packed_grad.numpy()

    return val_and_grad
