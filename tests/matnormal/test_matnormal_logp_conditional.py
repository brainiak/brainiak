import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import wishart, multivariate_normal
import tensorflow as tf
from brainiak.matnormal.utils import rmn
from brainiak.matnormal.matnormal_likelihoods import (
                       matnorm_logp_conditional_col,
                       matnorm_logp_conditional_row)
from brainiak.matnormal.covs import CovIdentity, CovUnconstrainedCholesky
import logging

logging.basicConfig(level=logging.DEBUG)

# X is m x n, so A sould be m x p

m = 5
n = 4
p = 3

rtol = 1e-7


def test_against_scipy_mvn_col_conditional():

    # have to be careful for constructing everything as a submatrix of a big
    # PSD matrix, else no guarantee that anything's invertible.
    cov_np = wishart.rvs(df=m+p+2, scale=np.eye(m+p))

    # rowcov = CovConstant(cov_np[0:m, 0:m])
    rowcov = CovUnconstrainedCholesky(size=m, Sigma=cov_np[0:m, 0:m])
    A = cov_np[0:m, m:]

    colcov = CovIdentity(size=n)

    Q = CovUnconstrainedCholesky(size=p, Sigma=cov_np[m:, m:])

    X = rmn(np.eye(m), np.eye(n))

    A_tf = tf.constant(A, 'float64')
    X_tf = tf.constant(X, 'float64')

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        Q_np = Q.Sigma.eval(session=sess)

        rowcov_np = rowcov.Sigma.eval(session=sess) - \
            A.dot(np.linalg.inv(Q_np)).dot((A.T))

        scipy_answer = np.sum(multivariate_normal.logpdf(X.T, np.zeros([m]),
                              rowcov_np))

        tf_answer = matnorm_logp_conditional_row(X_tf, rowcov, colcov, A_tf, Q)
        assert_allclose(scipy_answer, tf_answer.eval(session=sess), rtol=rtol)


def test_against_scipy_mvn_row_conditional():

    # have to be careful for constructing everything as a submatrix of a big
    # PSD matrix, else no guarantee that anything's invertible.
    cov_np = wishart.rvs(df=m+p+2, scale=np.eye(m+p))

    rowcov = CovIdentity(size=m)
    colcov = CovUnconstrainedCholesky(size=n, Sigma=cov_np[0:n, 0:n])
    A = cov_np[n:, 0:n]

    Q = CovUnconstrainedCholesky(size=p, Sigma=cov_np[n:, n:])

    X = rmn(np.eye(m), np.eye(n))

    A_tf = tf.constant(A, 'float64')
    X_tf = tf.constant(X, 'float64')

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        Q_np = Q.Sigma.eval(session=sess)

        colcov_np = colcov.Sigma.eval(session=sess) - \
            A.T.dot(np.linalg.inv(Q_np)).dot((A))

        scipy_answer = np.sum(multivariate_normal.logpdf(X, np.zeros([n]),
                                                         colcov_np))

        tf_answer = matnorm_logp_conditional_col(X_tf, rowcov, colcov, A_tf, Q)

        assert_allclose(scipy_answer, tf_answer.eval(session=sess), rtol=rtol)