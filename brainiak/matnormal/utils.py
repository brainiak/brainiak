import tensorflow as tf
from scipy.stats import norm
from numpy.linalg import cholesky


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
    return tf.diag(tf.ones([size], dtype=tf.float64) * x)


def quad_form_trp(x, y):
    """ x * y * x' """
    return tf.matmul(x, tf.matmul(y, x, transpose_b=True))
