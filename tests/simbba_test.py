import numpy as np
from scipy.stats import pearsonr
from simbba.covs import CovIdentity
from simbba.simbba import SiMBBA
from simbba.cogmodels import SignalDetectionTheoryModel
from simbba_gendata import gen_neural_data, gen_sdt_behavior, gen_neural_data_2layer
from scipy.stats import norm

train_obs = 200
test_obs = 100
psi_size = 2
s_size = 4
voxels_x = 3
voxels_y = 3
voxels_z = 3
n_voxels = voxels_x * voxels_y * voxels_z

corrtol = 0.8  # at least this much correlation between true and est to pass

def test_simbba_sdt_iid():

    # Y = XB + eps
    # Y is m x n, B is n x p, eps is m x p
    timecov_true = np.eye(train_obs+test_obs)
    spacecov_true = np.eye(n_voxels)

    behav_train, behav_test = gen_sdt_behavior(train_obs, test_obs, s_size)
    psi = np.r_[behav_train['psi'], behav_test['psi']]
    s = np.r_[behav_train['s'], behav_test['s']]

    tr, te, sz = gen_neural_data(psi, s, train_obs, test_obs, n_voxels,
                                 spacecov_true, timecov_true)

    time_cov = CovIdentity(size=train_obs)
    space_cov = CovIdentity(size=n_voxels)
    cogmodel = SignalDetectionTheoryModel(train_obs)

    model = SiMBBA(n_t=train_obs, n_v=n_voxels, time_noise_cov=time_cov,
                   space_noise_cov=space_cov, cogmodel=cogmodel, learnRate=0.05)

    model.fit(tr['B'], behav_train['choices'][:,None], behav_train['design'], max_iter=10000, restart=True)

    assert(pearsonr(model.cogmodel.psi.eval(session=model.sess).flatten(), behav_train['psi'].flatten())[0] >= corrtol)

def test_simbba_sdt_iid_hard():

    # Y = XB + eps
    # Y is m x n, B is n x p, eps is m x p
    timecov_true = np.eye(train_obs+test_obs)
    spacecov_true = np.eye(n_voxels)

    behav_train, behav_test = gen_sdt_behavior(train_obs, test_obs, s_size, dprime_scale=1, bias_scale=0.2)
    psi = np.r_[behav_train['psi'], behav_test['psi']]
    s = np.r_[behav_train['s'], behav_test['s']]

    tr, te, sz = gen_neural_data(psi, s, train_obs, test_obs, n_voxels,
                                 spacecov_true, timecov_true)

    time_cov = CovIdentity(size=train_obs)
    space_cov = CovIdentity(size=n_voxels)
    cogmodel = SignalDetectionTheoryModel(train_obs)

    model = SiMBBA(n_t=train_obs, n_v=n_voxels, time_noise_cov=time_cov,
                   space_noise_cov=space_cov, cogmodel=cogmodel, learnRate=0.05)

    model.fit(tr['B'], behav_train['choices'][:,None], behav_train['design'], max_iter=10000, restart=True)

    choice_prob_true = behav_train['choice_prob']
    bias_est = model.cogmodel.bias.eval(session=model.sess)
    dprime_est = model.cogmodel.d_prime.eval(session=model.sess)
    choice_prob_est = norm.cdf(np.where(behav_train['design'][:,0] == 1, dprime_est+bias_est, dprime_est-bias_est))


    assert(pearsonr(model.cogmodel.psi.eval(session=model.sess).flatten(), behav_train['psi'].flatten())[0] >= corrtol)

def test_simbba_sdt_iid_2layergendata():

    # Y = XB + eps
    # Y is m x n, B is n x p, eps is m x p
    timecov_true = np.eye(train_obs+test_obs)
    spacecov_true = np.eye(n_voxels)

    behav_train, behav_test = gen_sdt_behavior(train_obs, test_obs, s_size)
    psi = np.r_[behav_train['psi'], behav_test['psi']]
    s = np.r_[behav_train['s'], behav_test['s']]

    tr, te, sz = gen_neural_data_2layer(psi, s, train_obs, test_obs, 5, n_voxels,
                                 spacecov_true, timecov_true)

    time_cov = CovIdentity(size=train_obs)
    space_cov = CovIdentity(size=n_voxels)
    cogmodel = SignalDetectionTheoryModel(train_obs)

    model = SiMBBA(n_t=train_obs, n_v=n_voxels, time_noise_cov=time_cov,
                   space_noise_cov=space_cov, cogmodel=cogmodel, learnRate=0.01)

    model.fit(tr['B'], behav_train['choices'][:,None], behav_train['design'], max_iter=10000, restart=False)

    assert(pearsonr(model.cogmodel.psi.eval(session=model.sess).flatten(), behav_train['psi'].flatten())[0] >= corrtol)

import matplotlib.pyplot as plt
%matplotlib
plt.plot(model.cogmodel.psi.eval(session=model.sess).flatten(), behav_train['psi'].flatten(),'.')
plt.plot(model.cogmodel.d_prime.eval(session=model.sess), behav_train['dprime'],'.')

plt.plot(model.cogmodel.bias.eval(session=model.sess), behav_train['bias'],'.')

pearsonr(model.cogmodel.d_prime.eval(session=model.sess),model.cogmodel.bias.eval(session=model.sess))