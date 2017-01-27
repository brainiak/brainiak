import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import multivariate_normal
import tensorflow as tf
from brainiak.matnorm_models.helpers import rmn
from brainiak.matnorm_models import MatnormModelBase
from brainiak.matnorm_models.noise_covs import NoiseCovIdentity,NoiseCovFullRank

# X is m x n, so A sould be m x p

m = 5
n = 4
p = 3

rtol = 1e-7


def test_against_scipy_mvn_row():

    with tf.Session() as sess:

        rowcov = NoiseCovFullRank(size=m)
        colcov = NoiseCovIdentity(size=n)
        X = rmn(np.eye(m), np.eye(n))
        X_tf = tf.constant(X, 'float64')

        modelbase = MatnormModelBase()
        sess.run(tf.global_variables_initializer())

        rowcov_np = rowcov.Sigma.eval(session=sess)

        scipy_answer = np.sum(multivariate_normal.logpdf(X.T, np.zeros([m]),
                              rowcov_np))
        tf_answer = modelbase.matnorm_logp(X_tf, rowcov, colcov)
        assert_allclose(scipy_answer, tf_answer.eval(session=sess), rtol=rtol)


def test_against_scipy_mvn_col():

    with tf.Session() as sess:

        rowcov = NoiseCovIdentity(size=n)
        colcov = NoiseCovFullRank(size=m)
        X = rmn(np.eye(n), np.eye(m))
        X_tf = tf.constant(X, 'float64')

        modelbase = MatnormModelBase()
        sess.run(tf.global_variables_initializer())

        colcov_np = colcov.Sigma.eval(session=sess)

        scipy_answer = np.sum(multivariate_normal.logpdf(X, np.zeros([m]),
                              colcov_np))
        tf_answer = modelbase.matnorm_logp(X_tf, rowcov, colcov)
        assert_allclose(scipy_answer, tf_answer.eval(session=sess), rtol=rtol)
