import numpy as np
from scipy.stats import norm, bernoulli, invgamma
from scipy.special import expit as inv_logit
from numpy.linalg import cholesky

def gen_choiceRT_behavior(train_obs, test_obs, s_size):
    psi = rmn(np.eye(train_obs+test_obs), np.eye(2))
    
    stims = np.random.choice([-1,1], size=train_obs+test_obs, replace=True)
    choice_prob = norm.cdf(np.where(stims==1, psi[:,0]+psi[:,1], psi[:,0]-psi[:,1]))
    choices = np.array([bernoulli.rvs(cp) for cp in choice_prob])

    stim_representations = rmn(np.eye(2), np.eye(s_size))

    s = stim_representations[stims,:]

    rts = invgamma.rvs(np.exp(psi[:,0]),np.exp(psi[:,1]))

    H = np.c_[choices, rts].T

    train = {"psi":psi[:train_obs,], "H":H[:,:train_obs], "stims":stims[:train_obs], "s":s[:train_obs,:], "choice_prob": choice_prob[:train_obs]}
    test = {"psi":psi[train_obs:,],"H":H[:,train_obs:],"stims":stims[train_obs:,],"s":s[train_obs:,:], "choice_prob":choice_prob[train_obs:]}
    return train, test

def gen_sdt_behavior(train_obs, test_obs, s_size):
    psi = rmn(np.eye(train_obs+test_obs), np.eye(2))

    dprime = np.exp(psi[:,0]) # dprime is positive
    # dprime = psi[:,0]
    bias = psi[:,1]

    design = np.random.choice([-1,1], size=train_obs+test_obs, replace=True)
    choice_prob = norm.cdf(np.where(design==1, dprime+bias, dprime-bias))
    choices = np.array([bernoulli.rvs(cp) for cp in choice_prob])

    stim_representations = rmn(np.eye(2), np.eye(s_size))

    s = stim_representations[design,:]

    train = {"psi":psi[:train_obs,], "choices":choices[:train_obs], "design":design[:train_obs], "s":s[:train_obs,:], "choice_prob": choice_prob[:train_obs]}
    test = {"psi":psi[train_obs:,], "choices":choices[train_obs:],"design":design[train_obs:,],"s":s[train_obs:,:], "choice_prob":choice_prob[train_obs:]}
    return train, test

def gen_ar1_cov(n_obs):
    # 
    x = np.arange(n_obs)[:,None]
    xxt = x.dot(x.T)
    xsq = (x * x)
    D = np.sqrt(-2*xxt + xsq + xsq.T)
    rho = inv_logit(norm.rvs(1))
    sigma_ar = inv_logit(norm.rvs(1)) * 5 # range on 0,5

    return np.exp(-rho * D) + np.eye(n_obs) * sigma_ar, rho, sigma_ar


def gen_neural_data(psi, s, psi_size, s_size, train_obs, test_obs, phi_size, voxels_x, voxels_y, voxels_z, temporal_cov="Identity"):

    n_obs = train_obs + test_obs
    n_voxels = voxels_x * voxels_y * voxels_z
    # the expression in the derivation says
    # phi' ~ MN(psi * beta + s * ell, sigma_phit, sigma_phis)
    # phi is phi_size by n_obs
    # beta is psi_size by phi_size
    # ell is s_size by phi_size 
    # s is n_obs by s_size
    # psi is n_obs by psi_size 
    # note that s_size is the dimensionality of the stimulus, not the number
    # of stimuli. In SDT there are always 2 stimuli

    beta = rmn(np.eye(psi_size),np.eye(phi_size)) 
    # beta = bernoulli.rvs(p=0.5, size=(psi_size, phi_size))
    ell = rmn(np.eye(s_size),np.eye(phi_size))

    Phi_hat = psi @ beta #+ s %*% ell 

    # var_phis = np.abs(norm.rvs()) 
    # var_bs = np.abs(norm.rvs())

    # phi = Phi_hat + rmn(np.eye(n_obs), var_phis * np.eye(phi_size))
    phi = Phi_hat + norm.rvs(size=(n_obs, phi_size))

    spatialCov = np.eye(n_voxels) # * var_bs
    F = rmn(spatialCov, np.eye(phi_size))
    # to generate spatial cov in F for over 10K voxels we need more efficient
    # generation code. For now just force identity cov

    # F = norm.rvs(size=(n_voxels, phi_size))

    # B = F @ phi.T + norm.rvs(size=(n_voxels, n_obs))
    if temporal_cov == "Identity": 
        sigma_t = np.eye(n_obs)
        rho = 0
        sigma_ar = 0
    elif temporal_cov == "AR1":
        sigma_t, rho, sigma_ar = gen_ar1_cov(n_obs)

    B = F @ phi.T + rmn(spatialCov, sigma_t)

    sizes = {"n_test":test_obs,"n_train":train_obs,"n_voxels":n_voxels,"phi_size":phi_size,"psi_size":psi_size,"s_size":s_size,"n_voxels_x":voxels_x,"n_voxels_y":voxels_y,"n_voxels_z":voxels_z}

    train=  {"beta":beta,"phi":phi[:train_obs,:],"B":B[:,:train_obs],"F":F, "rho":rho,"sigma_ar":sigma_ar}

    test = {"beta":beta,"phi":phi[train_obs:,:],"B":B[:,train_obs:],"F":F}

    return train, test, sizes

