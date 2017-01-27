import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import multivariate_normal
import tensorflow as tf
from brainiak.matnorm_models.helpers import rmn
from brainiak.matnorm_models.base import MatnormModelBase
from brainiak.matnorm_models.noise_covs import NoiseCovIdentity, NoiseCovFullRank

# X is m x n, so A sould be m x p

m = 5
n = 4
p = 3

rtol = 1e-7


def test_against_scipy_mvn_row_marginal():

    rowcov = NoiseCovFullRank(size=m)
    colcov = NoiseCovIdentity(size=n)
    Q = NoiseCovFullRank(size=p)

    modelbase = MatnormModelBase()
    X = rmn(np.eye(m), np.eye(n))
    A = rmn(np.eye(m), np.eye(p))

    A_tf = tf.constant(A, 'float64')
    X_tf = tf.constant(X, 'float64')

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        Q_np = Q.Sigma.eval(session=sess)

        rowcov_np = rowcov.Sigma.eval(session=sess) + A.dot(Q_np).dot(A.T)

        scipy_answer = np.sum(multivariate_normal.logpdf(X.T, np.zeros([m]),
                              rowcov_np))

        tf_answer = modelbase.matnorm_logp_marginal_row(X_tf, rowcov, colcov,
                                                        A_tf, Q)
        assert_allclose(scipy_answer, tf_answer.eval(session=sess), rtol=rtol)


def test_against_scipy_mvn_col_marginal():

    rowcov = NoiseCovIdentity(size=n)
    colcov = NoiseCovFullRank(size=m)
    Q = NoiseCovFullRank(size=p)

    modelbase = MatnormModelBase()
    X = rmn(np.eye(n), np.eye(m))
    A = rmn(np.eye(m), np.eye(p))

    A_tf = tf.constant(A, 'float64')
    X_tf = tf.constant(X, 'float64')

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        Q_np = Q.Sigma.eval(session=sess)

        colcov_np = colcov.Sigma.eval(session=sess) + A.dot(Q_np).dot(A.T)

        scipy_answer = np.sum(multivariate_normal.logpdf(X, np.zeros([m]),
                                                         colcov_np))

        tf_answer = modelbase.matnorm_logp_marginal_col(X_tf, rowcov, colcov,
                                                        A_tf, Q)
        assert_allclose(scipy_answer, tf_answer.eval(session=sess), rtol=rtol)
