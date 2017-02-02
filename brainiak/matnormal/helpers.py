import functools  # https://danijar.com/structuring-your-tensorflow-models/
import tensorflow as tf
from numpy.linalg import cholesky
from scipy.stats import norm


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


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
    return tf.diag(tf.ones([size], dtype=tf.float64) * x,)


def quad_form_trp(x, y):
    """ x * y * x' """
    return tf.matmul(x, tf.matmul(y, x, transpose_b=True))


def rmn(rowcov, colcov):
    # generate random draws from a zero-mean matrix-normal distribution
    Z = norm.rvs(norm.rvs(size=(rowcov.shape[0], colcov.shape[0])))
    return(cholesky(rowcov).dot(Z).dot(cholesky(colcov)))
