import logging

logging.basicConfig(level=logging.DEBUG)

def test_matnorm_regression_gp():
    try:
        import GPflow
        from GPflow import Matern32
        from brainiak.matnormal.covs import CovIdentity
        from brainiak.matnormal.cov_gp import CovGP
        from brainiak.matnormal import MatnormRegression
        from brainiak.matnormal.helpers import rmn
        from scipy.stats import norm, pearsonr
        import tensorflow as tf
        import numpy as np

        # this is not the official GPflow way of doing it using a ContextManager but
        # since we're not using a lot of the rest of the machinery, we can get away
        # with overriding the float type (which defaults to float32 for some reason).
        GPflow.kernels.float_type = tf.float64

        m = 100
        n = 4
        p = 5

        corrtol = 0.8  # at least this much correlation between true and est to pass

        

        # create the actual kernel
        trueKern = GPflow.kernels.Matern32(input_dim=1)
        trueKern.lengthscales = 10
        trueKern.lengthscales.fixed = True
        trueKern.variance = 1
        trueKern.variance.fixed = True
        trueKern.make_tf_array(trueKern.get_free_state())
        times = np.sort(np.random.choice(1000, p))
        # get our gram matrix
        colcov_true = trueKern.compute_K_symm(times[:, None]) + np.eye(p)*0.0001
        # Y = XB + eps
        # Y is m x n, B is n x p, eps is m x p
        X = norm.rvs(size=(m, n))  # rmn(np.eye(m), np.eye(n))
        B = norm.rvs(size=(n, p))  # rmn(np.eye(n), np.eye(p))
        Y_hat = X.dot(B)
        rowcov_true = np.eye(m)

        Y = Y_hat + rmn(rowcov_true, colcov_true)

        row_cov = CovIdentity(size=m)
        col_cov = CovGP(kern=Matern32, input_dim=3)

        model = MatnormRegression(n_v=p, n_c=n, time_noise_cov=row_cov,
                                  space_noise_cov=col_cov, learnRate=0.01)

        model.fit(X, Y, max_iter=10000, step=10000)

        assert(pearsonr(B.flatten(), model.beta_.flatten())[0] >= corrtol)

    except ImportError:
        
        print("No GPflow found, skipping GPflow tests!")