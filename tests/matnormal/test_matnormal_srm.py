import numpy as np
import tensorflow as tf
# from brainiak.matnormal.srm_em_analytic import MatnormSRM_EM_Analytic
from brainiak.matnormal.srm_margs import MNSRM_OrthoW
from brainiak.matnormal.covs import CovUnconstrainedCholesky, CovDiagonal
from numpy.testing import assert_allclose
from brainiak.matnormal.utils import rmn
from scipy.stats import norm, pearsonr, invwishart
import logging

logging.basicConfig(level=logging.INFO)

n_T = 10
n_V = 15
n_features = 4
n_subj = 3

def rmse(x, xtrue):
    return np.sqrt(np.average((x-xtrue)**2))


def estep(W, sigma_v, sigma_t, sigma_s, b, rho, X):

    sigma_t_prime = sigma_t
    vinv = np.linalg.inv(sigma_v)
    wsw = np.zeros((n_features, n_features))
    wsx = np.zeros((n_features, n_T))
    xmb = X - b

    for j in range(n_subj):
        xmb_j = xmb[j*n_V:(j+1)*n_V]
        w_j = W[j*n_V:(j+1)*n_V]
        wsw = wsw + w_j.T @ np.linalg.solve(sigma_v, w_j) / rho[j]
        wsx = wsx + w_j.T @ np.linalg.solve(sigma_v, xmb_j) / rho[j]

    # additional savings here via cholesky probably
    sigma_s_prime_inv = np.linalg.inv(sigma_s) + wsw
    s_prime = np.linalg.solve(sigma_s_prime_inv, wsx)
    sigma_s_prime = np.linalg.solve(sigma_s_prime_inv, np.eye(sigma_s.shape[0]))
    return s_prime, sigma_s_prime, sigma_t_prime


def ldet(s):
    return np.linalg.slogdet(s)[1]


def gen_srm_data(n_T, n_V, n_subj, n_features, vcov=None, tcov=None, scov=None, ortho_w=False):

    if scov is None:
        sigma_w = sigma_s = np.eye(n_features)
    else:
        sigma_w = scov
        sigma_s = scov

    if vcov is None:
        sigma_v = invwishart.rvs(size=1, df=n_V+2,scale = np.eye(n_V))
    else:
        sigma_v = vcov

    if tcov is None:
        sigma_t = invwishart.rvs(size=1, df=n_T+2,scale = np.eye(n_T))
    else:
        sigma_t = tcov

    rho = np.exp(np.random.normal(size=n_subj))

    W = rmn(np.kron(np.diag(rho), sigma_v), sigma_w)

    wlist = W.reshape(n_subj, n_V, n_features)
    if ortho_w:
        for i in range(n_subj):
            u, s, v = np.linalg.svd(wlist[i].T @ wlist[i])
            wnew = wlist[i] @ u @ np.diag(1/np.sqrt(s)) @ v
            # wnew = u @ np.diag(1/np.sqrt(s)) @ v @ wlist[i]
            assert_allclose(wnew.T @ wnew, np.eye(n_features), rtol=1e-5, atol=1e-5)
            wlist[i] = wnew

    W = wlist.reshape(n_subj*n_V, n_features)
    S = rmn(sigma_s, sigma_t)

    b = np.random.normal(size=(n_subj * n_V, 1))
    ws = W @ S + b
    X = ws + rmn(np.kron(np.diag(rho), sigma_v), sigma_t)
    theta = W, S, b, sigma_v, sigma_t, sigma_s, 1/rho
    true_sufficient_stats = estep(W, sigma_v, sigma_t, sigma_s, b, rho, X)
    return X, theta, ws, true_sufficient_stats


def Q(W, sigma_v, sigma_t, sigma_s, b, rho, X, sigma_s_prime, sigma_t_prime, s_prime):

    v = sigma_v.shape[0]
    n = rho.shape[0]
    t = sigma_t.shape[0]
    k = sigma_s.shape[0]

    kroncov = np.kron(np.diag(rho), sigma_v)

    mean = (X - b - W @ s_prime)

    det_terms = -(v*n)*ldet(sigma_t) - t*n*ldet(sigma_v) - t*ldet(sigma_s) - t*v*np.sum(np.log(rho)) - (k)*ldet(sigma_t)

    x_quad_form = -np.trace(np.linalg.solve(sigma_t, mean.T) @ np.linalg.solve(kroncov, mean))
    s_quad_form = -np.trace(np.linalg.solve(sigma_t, s_prime.T) @ np.linalg.solve(sigma_s, s_prime))

    lik_trace = -np.trace(np.linalg.solve(sigma_t, sigma_t_prime)) * np.trace(sigma_s_prime @ W.T @ np.linalg.solve(kroncov, W))
    prior_trace = -np.trace(np.linalg.solve(sigma_t, sigma_t_prime)) * np.trace(np.linalg.solve(sigma_s, sigma_s_prime))

    return 0.5 * (det_terms + x_quad_form + s_quad_form + lik_trace + prior_trace)#, det_terms, x_quad_form, s_quad_form, lik_trace, prior_trace

def mstep_b(W, sigma_v, sigma_t, sigma_s, b, rho, X, sigma_s_prime, sigma_t_prime, s_prime):

    t = sigma_t.shape[0]
    b_hat = ((X - W @ s_prime) @ np.linalg.solve(sigma_t, np.ones((t,1)))) / np.sum(np.linalg.inv(sigma_t))
    return b_hat

def mstep_rho(W, sigma_v, sigma_t, sigma_s, b, rho, X, sigma_s_prime, sigma_t_prime, s_prime):
    mean = (X - b - W @ s_prime)
    n, v, t = rho.shape[0], sigma_v.shape[0], sigma_t.shape[0]
    rho_grad = np.zeros(n)

    rho_hat = np.zeros(n)
    for j in range(n):
        mean_j = mean[j*v:(j+1)*v]
        w_j = W[j*v:(j+1)*v]
        rho_hat[j] = (np.trace(np.linalg.solve(sigma_t, mean_j.T) @ np.linalg.solve(sigma_v, mean_j)) +
                       np.trace(np.linalg.solve(sigma_t, sigma_t_prime)) * np.trace(sigma_s_prime @ w_j.T @ np.linalg.solve(sigma_v, w_j)))/(t*v)

    return rho_hat

def _load_model_params(model, s_prime, b, rhoprec, Xstack, scov_prime, tcov_prime, w):
    model.s_prime.load(s_prime, session=model.sess)
    model.b.load(b.reshape(n_subj, n_V, 1), session=model.sess)
    model.rhoprec.load(rhoprec, session=model.sess)
    model.w.load(w.reshape(n_subj,n_V,n_features), session=model.sess)
    model.scov_prime.load(scov_prime, session=model.sess)
    model.tcov_prime.load(tcov_prime, session=model.sess)


def _init_all():

    Xstack, theta, ws, true_sufficient_stats = gen_srm_data(n_T, n_V, n_subj, n_features)
    s_prime, sigma_s_prime, sigma_t_prime = true_sufficient_stats

    W, S, b, sigma_v, sigma_t, sigma_s, rhoprec = theta

    q_np = Q(W, sigma_v, sigma_t, sigma_s, b, 1/rhoprec, Xstack, sigma_s_prime, sigma_t_prime, s_prime)

    X = Xstack.reshape(n_subj, n_V, n_T)

    model = MNSRM_OrthoW(n_features=n_features)

    model.n = len(X)

    model.v, model.t = X[0].shape

    model.X = tf.constant(X, name="X")

    xsvd = [np.linalg.svd(x)for x in X]

    # parameters
    model.b = tf.Variable(np.random.normal(size=(model.n, model.v,1)), name="b")
    model.rhoprec = tf.Variable(np.ones(model.n), name="rhoprec")
    model.w = tf.Variable(np.array([sv[0][:,:model.k] for sv in xsvd]), name="w")

    # sufficient statistics
    model.s_prime = tf.Variable(np.average([sv[2][:model.k, :]  for sv in xsvd], 0), dtype=tf.float64, name="s_prime")
    model.tcov_prime = tf.Variable(np.eye(model.t), name="wcov_prime")
    model.scov_prime = tf.Variable(np.eye(model.k), name="vcov_prime")

    model.space_cov = CovFullRankCholesky(size=n_V, Sigma=sigma_v)
    model.time_cov = CovFullRankCholesky(size=n_T, Sigma=sigma_t)
    model.marg_cov = CovFullRankCholesky(size=n_features, Sigma=sigma_s)

    model.sess.run(tf.global_variables_initializer())

    _load_model_params(model, s_prime, b, rhoprec, Xstack, sigma_s_prime, sigma_t_prime, W)

    return Xstack, theta, ws, true_sufficient_stats, model


def test_Q():

    # q_op, det_terms_op, x_quad_form_op, s_quad_form_op, lik_trace_op, prior_trace_op = model._make_Q_op()
    
    Xstack, theta, ws, true_sufficient_stats, model = _init_all()

    q_op = model._make_Q_op()

    q_tf = q_op.eval(session=model.sess)
    # det_terms_tf = det_terms_op.eval(session=model.sess)
    # x_quad_form_tf = x_quad_form_op.eval(session=model.sess)
    # s_quad_form_tf = s_quad_form_op.eval(session=model.sess)
    # lik_trace_tf = lik_trace_op.eval(session=model.sess)
    # prior_trace_tf = prior_trace_op.eval(session=model.sess)

    # q_np, det_terms_np, x_quad_form_np, s_quad_form_np, lik_trace_np, prior_trace_np = Q(W, sigma_v, sigma_t, sigma_s, b, 1/rhoprec, Xstack, sigma_s_prime, sigma_t_prime, s_prime)

    s_prime, sigma_s_prime, sigma_t_prime = true_sufficient_stats

    W, S, b, sigma_v, sigma_t, sigma_s, rhoprec = theta

    q_np = Q(W, sigma_v, sigma_t, sigma_s, b, 1/rhoprec, Xstack, sigma_s_prime, sigma_t_prime, s_prime)

    assert_allclose(q_tf, q_np)


def test_estep():

    Xstack, theta, ws, true_sufficient_stats, model = _init_all()

    s_prime_op, scov_prime_op, tcov_prime_op = model.make_estep_ops()

    s_prime_tf = s_prime_op.eval(session=model.sess)
    scov_prime_tf = scov_prime_op.eval(session=model.sess)
    tcov_prime_tf = tcov_prime_op.eval(session=model.sess)

    W, S, b, sigma_v, sigma_t, sigma_s, rhoprec = theta

    s_prime_np, scov_prime_np, tcov_prime_np = estep(W, sigma_v, sigma_t, sigma_s, b, 1/rhoprec, Xstack)

    assert_allclose(s_prime_np, s_prime_tf)
    assert_allclose(scov_prime_np, scov_prime_tf)
    assert_allclose(tcov_prime_np, tcov_prime_tf)


def test_mstep():
    Xstack, theta, ws, true_sufficient_stats, model = _init_all()
    W, S, b, sigma_v, sigma_t, sigma_s, rhoprec = theta

    b_op = model.make_mstep_b_op()
    # S_op = model.make_mstep_S_op()
    rhoprec_op = model.make_mstep_rhoprec_op()

    s_prime, scov_prime, tcov_prime = estep(W, sigma_v, sigma_t, sigma_s, b, 1/rhoprec, Xstack)

    b_tf = b_op.eval(session=model.sess)

    b_np = mstep_b(W, sigma_v, sigma_t, sigma_s, b, 1/rhoprec, Xstack, scov_prime, tcov_prime, s_prime)

    assert_allclose(b_np.reshape(n_subj, n_V, 1), b_tf)

    rhoprec_tf = rhoprec_op.eval(session=model.sess)

    rhoprec_np = 1/mstep_rho(W, sigma_v, sigma_t, sigma_s, b, 1/rhoprec, Xstack, scov_prime, tcov_prime, s_prime)
    assert_allclose(rhoprec_tf, rhoprec_np)

def test_dpsrm_identity_covs():

    sigma_t = np.eye(n_T)
    sigma_v = np.eye(n_V)
    sigma_w = sigma_s = np.eye(n_features)

    Xstack, theta, ws, true_sufficient_stats = gen_srm_data(n_T, n_V, n_subj, n_features,
                                                            vcov=sigma_v, tcov=sigma_t,
                                                            scov=sigma_s, ortho_w=True)
    X = Xstack.reshape(n_subj, n_V, n_T)
    
    model = MNSRM_OrthoW(n_features=n_features, space_noise_cov=CovIdentity, time_noise_cov=CovIdentity)


    model.fit(X, n_iter=10)

    reconstructed_WS = model.w_.dot(model.s_)
    rmse(ws.reshape(n_subj, n_V, n_T), reconstructed_WS)
    assert(pearsonr(ws.reshape(n_subj, n_V, n_T).flatten(), reconstructed_WS.flatten())[0] > 0.8)



def test_mnsrm_ecm():

    sigma_t = np.diag(np.abs(norm.rvs(size=n_T)))
    sigma_v = np.diag(np.abs(norm.rvs(size=n_V)))

    Xstack, theta, ws, true_sufficient_stats = gen_srm_data(n_T, n_V, n_subj, n_features,
                                                            vcov=sigma_v, tcov=sigma_t,
                                                            scov=None, ortho_w=True)

    X = Xstack.reshape(n_subj, n_V, n_T)
    
    model = MNSRM_OrthoW(n_features=n_features, space_noise_cov=CovDiagonal, time_noise_cov=CovDiagonal)

    model.fit(X, n_iter=10)

    reconstructed_WS = model.w_.dot(model.s_)
    rmse(ws.reshape(n_subj, n_V, n_T), reconstructed_WS)
    assert(pearsonr(ws.reshape(n_subj, n_V, n_T).flatten(), reconstructed_WS.flatten())[0] > 0.8)


def test_dpmnsrm_transform():
    sigma_t = np.diag(np.abs(norm.rvs(size=n_T)))
    sigma_v = np.diag(np.abs(norm.rvs(size=n_V)))

    Xstack, theta, ws, true_sufficient_stats = gen_srm_data(n_T, n_V, n_subj, n_features,
                                                            vcov=sigma_v, tcov=sigma_t,
                                                            wcov=None, ortho_s=False)

    W = theta[0]
    X = Xstack.reshape(n_subj, n_V, n_T)

    model = DPMNSRM(n_features=n_features, space_noise_cov=CovDiagonal, time_noise_cov=CovDiagonal)

    model.fit(X, max_iter=50, convergence_tol=0.001)

    newS = np.random.normal(size=(n_subj, n_features, n_T))

    newX = np.array([w @ s for (w,s) in zip(W.reshape(n_subj, n_V, n_features), newS)]) + np.random.normal(size=(n_subj, n_V, n_T))

    new_WS = np.array([w @ s for (w,s) in zip(W.reshape(n_subj, n_V, n_features), newS)])

    projected_wS = np.array([w @ s for (w,s) in zip(model.w_, model.transform(newX))])
    rmse(new_WS.flatten(), projected_wS.flatten())
    pearsonr(new_WS.flatten(), projected_wS.flatten())
    assert(pearsonr(new_WS.flatten(), projected_wS.flatten())[0] > 0.8)

def test_dpmnsrm_orthos():
    sigma_t = np.diag(np.abs(norm.rvs(size=n_T)))
    sigma_v = np.diag(np.abs(norm.rvs(size=n_V)))
    sigma_w = invwishart.rvs(size=1, df=n_features+2,scale = np.eye(n_features))

    Xstack, theta, ws, true_sufficient_stats = gen_srm_data(n_T, n_V, n_subj, n_features,
                                                            vcov=sigma_v, tcov=sigma_t,
                                                            wcov=sigma_w)

    W = theta[0]
    X = Xstack.reshape(n_subj, n_V, n_T)

    model = DPMNSRM(n_features=n_features, space_noise_cov=CovDiagonal,
                                           time_noise_cov=CovDiagonal,
                                           w_cov=CovFullRankCholesky,
                                           s_constraint='ortho')

    model.fit(X, max_iter=50, convergence_tol=0.001)

    reconstructed_WS = model.w_.dot(model.s_)
    rmse(ws.reshape(n_subj, n_V, n_T), reconstructed_WS)
    assert(pearsonr(ws.reshape(n_subj, n_V, n_T).flatten(), reconstructed_WS.flatten())[0] > 0.8)


def test_dpsrm_identity_covs():
    sigma_t = np.diag(np.abs(norm.rvs(size=n_T)))
    sigma_v = np.diag(np.abs(norm.rvs(size=n_V)))
    sigma_w = invwishart.rvs(size=1, df=n_features+2,scale = np.eye(n_features))

    Xstack, theta, ws, true_sufficient_stats = gen_srm_data(n_T, n_V, n_subj, n_features,
                                                            vcov=sigma_v, tcov=sigma_t,
                                                            wcov=sigma_w)

    W = theta[0]
    X = Xstack.reshape(n_subj, n_V, n_T)

    model = DPMNSRM(n_features=n_features)

    model.fit(X, max_iter=50, convergence_tol=0.001)

    reconstructed_WS = model.w_.dot(model.s_)
    rmse(ws.reshape(n_subj, n_V, n_T), reconstructed_WS)
    assert(pearsonr(ws.reshape(n_subj, n_V, n_T).flatten(), reconstructed_WS.flatten())[0] > 0.8)

def test_dpsrm_identity_covs_orthos():
    sigma_t = np.diag(np.abs(norm.rvs(size=n_T)))
    sigma_v = np.diag(np.abs(norm.rvs(size=n_V)))
    sigma_w = invwishart.rvs(size=1, df=n_features+2,scale = np.eye(n_features))

    Xstack, theta, ws, true_sufficient_stats = gen_srm_data(n_T, n_V, n_subj, n_features,
                                                            vcov=sigma_v, tcov=sigma_t,
                                                            wcov=sigma_w)

    W = theta[0]
    X = Xstack.reshape(n_subj, n_V, n_T)

    model = DPMNSRM(n_features=n_features, s_constraint="ortho", w_cov=CovFullRankCholesky)

    model.fit(X, max_iter=50, convergence_tol=0.001)

    reconstructed_WS = model.w_.dot(model.s_)
    rmse(ws.reshape(n_subj, n_V, n_T), reconstructed_WS)
    assert(pearsonr(ws.reshape(n_subj, n_V, n_T).flatten(), reconstructed_WS.flatten())[0] > 0.8)