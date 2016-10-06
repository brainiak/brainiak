#  Copyright 2016 Mingbo Cai, Princeton Neuroscience Instititute,
#  Princeton University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from sklearn.utils.validation import NotFittedError
import pytest


def test_can_instantiate():
    import brainiak.reprsimil.brsa
    import numpy as np
    s = brainiak.reprsimil.brsa.BRSA()
    assert s, "Invalid BRSA instance!"

    voxels = 100
    samples = 500
    features = 3

    s = brainiak.reprsimil.brsa.BRSA(n_iter=50, rank=5, GP_space=True, GP_inten=True, tol=2e-3,\
                epsilon=0.001,space_smooth_range=10.0,inten_smooth_range=100.0)
    assert s, "Invalid BRSA instance!"

def test_fit():
    from brainiak.reprsimil.brsa import BRSA
    import brainiak.utils.utils as utils
    import scipy.stats
    import numpy as np
    import os.path
    np.random.seed(10)
    file_path = os.path.join(os.path.dirname(__file__), "example_design.1D")
    # Load an example design matrix
    design = utils.ReadDesign(fname=file_path)
    
    
    # concatenate it by 4 times, mimicking 4 runs of itenditcal timing
    design.design_used = np.tile(design.design_used[:,1:17],[4,1])
    design.n_TR = design.n_TR * 4

    # start simulating some data
    n_V = 300
    n_C = np.size(design.design_used,axis=1) 
    n_T = design.n_TR

    noise_bot = 0.5
    noise_top = 1.5
    noise_level = np.random.rand(n_V)*(noise_top-noise_bot)+noise_bot
    # noise level is random.

    # AR(1) coefficient
    rho1_top = 0.8
    rho1_bot = -0.2
    rho1 = np.random.rand(n_V)*(rho1_top-rho1_bot)+rho1_bot

    # generating noise
    noise = np.zeros([n_T,n_V])
    noise[0,:] = np.random.randn(n_V) * noise_level / np.sqrt(1-rho1**2)
    for i_t in range(1,n_T):
        noise[i_t,:] = noise[i_t-1,:] * rho1 +  np.random.randn(n_V) * noise_level

    # ideal covariance matrix
    ideal_cov = np.zeros([n_C,n_C])
    ideal_cov = np.eye(n_C)*0.6
    ideal_cov[5:9,5:9] = 0.6
    for cond in range(5,9):
        ideal_cov[cond,cond] = 1
    idx = np.where(np.sum(np.abs(ideal_cov),axis=0)>0)[0]
    L_full = np.linalg.cholesky(ideal_cov)        

    # generating signal
    snr_level = 5.0 # test with high SNR    
    # snr = np.random.rand(n_V)*(snr_top-snr_bot)+snr_bot
    # Notice that accurately speaking this is not snr. the magnitude of signal depends
    # not only on beta but also on x.
    inten = np.random.randn(n_V) * 20.0

    # parameters of Gaussian process to generate pseuso SNR
    tau = 0.8
    smooth_width = 5.0
    inten_kernel = 1.0
    
    coords = np.arange(0,n_V)[:,None]

    dist2 = np.square(coords-coords.T)

    inten_tile = np.tile(inten,[n_V,1])
    inten_diff2 = (inten_tile-inten_tile.T)**2

    K = np.exp(-dist2/smooth_width**2/2.0 -inten_diff2/inten_kernel**2/2.0) * tau**2 + np.eye(n_V)*tau**2*0.001

    L = np.linalg.cholesky(K)
    snr = np.exp(np.dot(L,np.random.randn(n_V))) * snr_level
    sqrt_v = noise_level*snr
    betas_simulated = np.dot(L_full,np.random.randn(n_C,n_V)) * sqrt_v
    signal = np.dot(design.design_used,betas_simulated)

    # Adding noise to signal as data
    Y = signal + noise


    scan_onsets = np.linspace(0,design.n_TR,num=5)


    # Test fitting with GP prior.
    brsa = BRSA(GP_space=True,GP_inten=True,verbose=False,n_iter = 200)

    # We also test that it can detect baseline regressor included in the design matrix for task conditions
    wrong_design = np.insert(design.design_used, 0, 1, axis=1)
    with pytest.raises(ValueError) as excinfo:
        brsa.fit(X=Y, design=wrong_design, scan_onsets=scan_onsets,
             coords=coords, inten=inten)
    assert 'Your design matrix appears to have included baseline time series.' in str(excinfo.value)
    # Now we fit with the correct design matrix.
    brsa.fit(X=Y, design=design.design_used, scan_onsets=scan_onsets,
             coords=coords, inten=inten)
    
    # Check that result is significantly correlated with the ideal covariance matrix
    u_b = brsa.U_[1:,1:]
    u_i = ideal_cov[1:,1:]
    p = scipy.stats.spearmanr(u_b[np.tril_indices_from(u_b,k=-1)],u_i[np.tril_indices_from(u_i,k=-1)])[1]
    assert p < 0.01, "Fitted covariance matrix does not correlate with ideal covariance matrix!"
    # check that the recovered SNRs makes sense
    p = scipy.stats.pearsonr(brsa.nSNR_,snr)[1]
    assert p < 0.01, "Fitted SNR does not correlate with simualted SNR!"
    assert np.isclose(np.mean(np.log(brsa.nSNR_)),0), "nSNR_ not normalized!"
    p = scipy.stats.pearsonr(brsa.sigma_,noise_level)[1]
    assert p < 0.01, "Fitted noise level does not correlate with simualted noise level!"
    p = scipy.stats.pearsonr(brsa.rho_,rho1)[1]
    assert p < 0.01, "Fitted AR(1) coefficient does not correlate with simualted values!"

    # Test fitting with lower rank
    rank = n_C - 1
    brsa = BRSA(rank=rank)
    brsa.fit(X=Y, design=design.design_used, scan_onsets=scan_onsets,
             coords=coords, inten=inten)
    u_b = brsa.U_[1:,1:]
    u_i = ideal_cov[1:,1:]
    p = scipy.stats.spearmanr(u_b[np.tril_indices_from(u_b,k=-1)],u_i[np.tril_indices_from(u_i,k=-1)])[1]
    assert p < 0.01, "Fitted covariance matrix does not correlate with ideal covariance matrix!"
    # check that the recovered SNRs makes sense
    p = scipy.stats.pearsonr(brsa.nSNR_,snr)[1]
    assert p < 0.01, "Fitted SNR does not correlate with simualted SNR!"
    assert np.isclose(np.mean(np.log(brsa.nSNR_)),0), "nSNR_ not normalized!"
    p = scipy.stats.pearsonr(brsa.sigma_,noise_level)[1]
    assert p < 0.01, "Fitted noise level does not correlate with simualted noise level!"
    p = scipy.stats.pearsonr(brsa.rho_,rho1)[1]
    assert p < 0.01, "Fitted AR(1) coefficient does not correlate with simualted values!"

    # Test fitting without GP prior.
    brsa = BRSA()
    brsa.fit(X=Y, design=design.design_used, scan_onsets=scan_onsets)

    # Check that result is significantly correlated with the ideal covariance matrix
    u_b = brsa.U_[1:,1:]
    u_i = ideal_cov[1:,1:]
    p = scipy.stats.spearmanr(u_b[np.tril_indices_from(u_b,k=-1)],u_i[np.tril_indices_from(u_i,k=-1)])[1]
    assert p < 0.01, "Fitted covariance matrix does not correlate with ideal covariance matrix!"
    # check that the recovered SNRs makes sense
    p = scipy.stats.pearsonr(brsa.nSNR_,snr)[1]
    assert p < 0.01, "Fitted SNR does not correlate with simualted SNR!"
    assert np.isclose(np.mean(np.log(brsa.nSNR_)),0), "nSNR_ not normalized!"
    p = scipy.stats.pearsonr(brsa.sigma_,noise_level)[1]
    assert p < 0.01, "Fitted noise level does not correlate with simualted noise level!"
    p = scipy.stats.pearsonr(brsa.rho_,rho1)[1]
    assert p < 0.01, "Fitted AR(1) coefficient does not correlate with simualted values!"
    assert not hasattr(brsa,'bGP_') and not hasattr(brsa,'lGPspace_') and not hasattr(brsa,'lGPinten_'),\
        'the BRSA object should not have parameters of GP if GP is not requested.'
    # GP parameters are not set if not requested

    # Test fitting with GP over just spatial coordinates.
    brsa = BRSA(GP_space=True)
    brsa.fit(X=Y, design=design.design_used, scan_onsets=scan_onsets, coords=coords)
    # Check that result is significantly correlated with the ideal covariance matrix
    u_b = brsa.U_[1:,1:]
    u_i = ideal_cov[1:,1:]
    p = scipy.stats.spearmanr(u_b[np.tril_indices_from(u_b,k=-1)],u_i[np.tril_indices_from(u_i,k=-1)])[1]
    assert p < 0.01, "Fitted covariance matrix does not correlate with ideal covariance matrix!"
    # check that the recovered SNRs makes sense
    p = scipy.stats.pearsonr(brsa.nSNR_,snr)[1]
    assert p < 0.01, "Fitted SNR does not correlate with simualted SNR!"
    assert np.isclose(np.mean(np.log(brsa.nSNR_)),0), "nSNR_ not normalized!"
    p = scipy.stats.pearsonr(brsa.sigma_,noise_level)[1]
    assert p < 0.01, "Fitted noise level does not correlate with simualted noise level!"
    p = scipy.stats.pearsonr(brsa.rho_,rho1)[1]
    assert p < 0.01, "Fitted AR(1) coefficient does not correlate with simualted values!"
    assert not hasattr(brsa,'lGPinten_'),\
        'the BRSA object should not have parameters of lGPinten_ if only smoothness in space is requested.'
    # GP parameters are not set if not requested


def test_gradient():
    from brainiak.reprsimil.brsa import BRSA
    import brainiak.utils.utils as utils
    import scipy.stats
    import numpy as np
    import os.path
    import numdifftools as nd

    np.random.seed(100)
    file_path = os.path.join(os.path.dirname(__file__), "example_design.1D")
    # Load an example design matrix
    design = utils.ReadDesign(fname=file_path)
    n_run = 4
    # concatenate it by 4 times, mimicking 4 runs of itenditcal timing
    design.design_used = np.tile(design.design_used[:,1:17],[n_run,1])
    design.n_TR = design.n_TR * n_run

    # start simulating some data
    n_V = 200
    n_C = np.size(design.design_used,axis=1)
    n_T = design.n_TR

    noise_bot = 0.5
    noise_top = 1.5
    noise_level = np.random.rand(n_V)*(noise_top-noise_bot)+noise_bot
    # noise level is random.

    # AR(1) coefficient
    rho1_top = 0.8
    rho1_bot = -0.2
    rho1 = np.random.rand(n_V)*(rho1_top-rho1_bot)+rho1_bot

    # generating noise
    noise = np.zeros([n_T,n_V])
    noise[0,:] = np.random.randn(n_V) * noise_level / np.sqrt(1-rho1**2)
    for i_t in range(1,n_T):
        noise[i_t,:] = noise[i_t-1,:] * rho1 +  np.random.randn(n_V) * noise_level

    # ideal covariance matrix
    ideal_cov = np.zeros([n_C,n_C])
    ideal_cov = np.eye(n_C)*0.6
    ideal_cov[0,0] = 0.2
    ideal_cov[5:9,5:9] = 0.6
    for cond in range(5,9):
        ideal_cov[cond,cond] = 1
    idx = np.where(np.sum(np.abs(ideal_cov),axis=0)>0)[0]
    L_full = np.linalg.cholesky(ideal_cov)

    # generating signal
    snr_level = 5.0 # test with high SNR
    # snr = np.random.rand(n_V)*(snr_top-snr_bot)+snr_bot
    # Notice that accurately speaking this is not snr. the magnitude of signal depends
    # not only on beta but also on x.
    inten = np.random.randn(n_V) * 20.0

    # parameters of Gaussian process to generate pseuso SNR
    tau = 0.8
    smooth_width = 5.0
    inten_kernel = 1.0

    coords = np.arange(0,n_V)[:,None]

    dist2 = np.square(coords-coords.T)

    inten_tile = np.tile(inten,[n_V,1])
    inten_diff2 = (inten_tile-inten_tile.T)**2

    K = np.exp(-dist2/smooth_width**2/2.0 -inten_diff2/inten_kernel**2/2.0) * tau**2 + np.eye(n_V)*tau**2*0.001

    L = np.linalg.cholesky(K)
    snr = np.exp(np.dot(L,np.random.randn(n_V))) * snr_level
    sqrt_v = noise_level*snr
    betas_simulated = np.dot(L_full,np.random.randn(n_C,n_V)) * sqrt_v
    signal = np.dot(design.design_used,betas_simulated)

    # Adding noise to signal as data
    Y = signal + noise


    scan_onsets = np.linspace(0,design.n_TR,num=n_run+1)


    # Test fitting with GP prior.
    brsa = BRSA(GP_space=True,GP_inten=True,verbose=False,n_iter = 200,rank=n_C)

    # test if the gradients are correct
    XTY, XTDY, XTFY, YTY_diag, YTDY_diag, YTFY_diag, XTX, \
        XTDX, XTFX, X0TX0, X0TDX0, X0TFX0, XTX0, XTDX0, XTFX0, \
        X0TY, X0TDY, X0TFY, X0, n_run_returned, n_base =\
            brsa._prepare_data(design.design_used,Y,n_T,scan_onsets)
    assert n_run_returned == n_run, 'There is mistake in counting number of runs'
    assert np.ndim(XTY) == 2 and np.ndim(XTDY) == 2 and np.ndim(XTFY) == 2,\
        'Dimension of XTY etc. returned from _prepare_data is wrong'
    assert np.ndim(YTY_diag) == 1 and np.ndim(YTDY_diag) == 1 and np.ndim(YTFY_diag) == 1,\
        'Dimension of YTY_diag etc. returned from _prepare_data is wrong'
    assert np.ndim(XTX) == 2 and np.ndim(XTDX) == 2 and np.ndim(XTFX) == 2,\
        'Dimension of XTX etc. returned from _prepare_data is wrong'
    assert np.ndim(X0TX0) == 2 and np.ndim(X0TDX0) == 2 and np.ndim(X0TFX0) == 2,\
        'Dimension of X0TX0 etc. returned from _prepare_data is wrong'
    assert np.ndim(XTX0) == 2 and np.ndim(XTDX0) == 2 and np.ndim(XTFX0) == 2,\
        'Dimension of XTX0 etc. returned from _prepare_data is wrong'
    assert np.ndim(X0TY) == 2 and np.ndim(X0TDY) == 2 and np.ndim(X0TFY) == 2,\
        'Dimension of X0TY etc. returned from _prepare_data is wrong'
    X0 = np.ones(n_T)
    l_idx = np.tril_indices(n_C)
    # rank = n_C - 1
    # idx_rank = np.where(l_idx[1] < rank)
    # l_idx = (l_idx[0][idx_rank], l_idx[1][idx_rank])
    n_l = np.size(l_idx[0])

    # Make sure all the fields are in the indices.
    idx_param_sing, idx_param_fitU, idx_param_fitV = brsa._build_index_param(n_l, n_V, 2)
    assert 'Cholesky' in idx_param_sing and 'a1' in idx_param_sing, \
        'The dictionary for parameter indexing misses some keys'
    assert 'Cholesky' in idx_param_fitU and 'a1' in idx_param_fitU, \
        'The dictionary for parameter indexing misses some keys'
    assert 'log_SNR2' in idx_param_fitV and 'c_space' in idx_param_fitV \
        and 'c_inten' in idx_param_fitV and 'c_both' in idx_param_fitV, \
        'The dictionary for parameter indexing misses some keys'
    
    # Initial parameters are correct parameters with some perturbation
    param0_fitU = np.random.randn(n_l+n_V) * 0.1
    param0_fitV = np.random.randn(n_V+1) * 0.1
    param0_sing = np.random.randn(n_l+1) * 0.1
    # param0_sing[idx_param_sing['log_sigma2']] += np.mean(np.log(noise_level)) * 2
    param0_sing[idx_param_sing['a1']] += np.mean(np.tan(rho1 * np.pi / 2))
    param0_fitV[idx_param_fitV['log_SNR2']] += np.log(snr[:n_V-1])*2
    param0_fitV[idx_param_fitV['c_space']] += np.log(smooth_width)*2
    param0_fitV[idx_param_fitV['c_inten']] += np.log(inten_kernel)*2
    
    # log likelihood and derivative of the _singpara function
    ll0, deriv0 = brsa._loglike_AR1_singpara(param0_sing, XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag,
                                             XTY, XTDY, XTFY, X0TX0, X0TDX0, X0TFX0,
                                             XTX0, XTDX0, XTFX0, X0TY, X0TDY, X0TFY, 
                                             l_idx, n_C, n_T, n_V, n_run, n_base,
                                             idx_param_sing)
    # We test the gradient to the Cholesky factor
    vec = np.zeros(np.size(param0_sing))
    vec[idx_param_sing['Cholesky'][0]] = 1
    dd = nd.directionaldiff(lambda x: brsa._loglike_AR1_singpara(x, XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag,
                                                                 XTY, XTDY, XTFY, X0TX0, X0TDX0, X0TFX0,
                                                                 XTX0, XTDX0, XTFX0, X0TY, X0TDY, X0TFY,
                                                                 l_idx, n_C, n_T, n_V, n_run, n_base,
                                                                 idx_param_sing)[0],
                            param0_sing, vec)
    assert np.isclose(dd, np.dot(deriv0, vec), rtol=0.01), 'gradient of singpara wrt Cholesky is incorrect'

    # We test the gradient to log(sigma^2)
    # vec = np.zeros(np.size(param0_sing))
    # vec[idx_param_sing['log_sigma2']] = 1
    # dd = nd.directionaldiff(lambda x: brsa._loglike_AR1_singpara(x, XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag,
    #                                                              XTY, XTDY, XTFY, X0TX0, X0TDX0, X0TFX0,
    #                                                              XTX0, XTDX0, XTFX0, X0TY, X0TDY, X0TFY,
    #                                                              l_idx, n_C, n_T, n_V, n_run, n_base,
    #                                                              idx_param_sing)[0],
    #                         param0_sing, vec)
    # assert np.isclose(dd, np.dot(deriv0, vec), rtol=0.01), 'gradient of singpara wrt log(sigma2) is incorrect'

    # We test the gradient to a1
    vec = np.zeros(np.size(param0_sing))
    vec[idx_param_sing['a1']] = 1
    dd = nd.directionaldiff(lambda x: brsa._loglike_AR1_singpara(x, XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag,
                                                                 XTY, XTDY, XTFY, X0TX0, X0TDX0, X0TFX0,
                                                                 XTX0, XTDX0, XTFX0, X0TY, X0TDY, X0TFY,
                                                                 l_idx, n_C, n_T, n_V, n_run, n_base,
                                                                 idx_param_sing)[0],
                            param0_sing, vec)
    assert np.isclose(dd, np.dot(deriv0, vec), rtol=0.01), 'gradient of singpara wrt a1 is incorrect'

    
    # log likelihood and derivative of the fitU function.
    ll0, deriv0 = brsa._loglike_AR1_diagV_fitU(param0_fitU, XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag,
                                               XTY, XTDY, XTFY, X0TX0, X0TDX0, X0TFX0,
                                               XTX0, XTDX0, XTFX0, X0TY, X0TDY, X0TFY,
                                               np.log(snr)*2, l_idx,n_C,n_T,n_V,n_run,n_base,idx_param_fitU,n_C)

    
    # We test the gradient wrt the reparametrization of AR(1) coefficient of noise.
    vec = np.zeros(np.size(param0_fitU))
    vec[idx_param_fitU['a1'][0]] = 1
    dd = nd.directionaldiff(lambda x: brsa._loglike_AR1_diagV_fitU(x, XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag,
                                                                   XTY, XTDY, XTFY, X0TX0, X0TDX0, X0TFX0,
                                                                   XTX0, XTDX0, XTFX0, X0TY, X0TDY, X0TFY,
                                                                   np.log(snr)*2, l_idx, n_C, n_T, n_V, n_run, n_base,
                                                                   idx_param_fitU, n_C)[0], param0_fitU, vec)
    assert np.isclose(dd, np.dot(deriv0,vec), rtol=0.01), 'gradient of fitU wrt to AR(1) coefficient incorrect'

    # We test if the numerical and analytical gradient wrt to the first element of Cholesky factor is correct
    vec = np.zeros(np.size(param0_fitU))
    vec[idx_param_fitU['Cholesky'][0]] = 1
    dd = nd.directionaldiff(lambda x: brsa._loglike_AR1_diagV_fitU(x, XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag,
                                                                   XTY, XTDY, XTFY, X0TX0, X0TDX0, X0TFX0,
                                                                   XTX0, XTDX0, XTFX0, X0TY, X0TDY, X0TFY,
                                                                   np.log(snr)*2, l_idx, n_C, n_T, n_V, n_run,n_base,
                                                                   idx_param_fitU, n_C)[0], param0_fitU, vec)
    assert np.isclose(dd, np.dot(deriv0,vec), rtol=0.01), 'gradient of fitU wrt Cholesky factor incorrect'

    # Test on a random direction
    vec = np.random.randn(np.size(param0_fitU))
    vec = vec / np.linalg.norm(vec)
    dd = nd.directionaldiff(lambda x: brsa._loglike_AR1_diagV_fitU(x, XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag,
                                                                   XTY, XTDY, XTFY, X0TX0, X0TDX0, X0TFX0,
                                                                   XTX0, XTDX0, XTFX0, X0TY, X0TDY, X0TFY, 
                                                                   np.log(snr)*2, l_idx, n_C, n_T, n_V, n_run, n_base,
                                                                   idx_param_fitU, n_C)[0], param0_fitU, vec)
    assert np.isclose(dd, np.dot(deriv0,vec), rtol=0.01), 'gradient of fitU incorrect'

    # We test the gradient of _fitV wrt to log(SNR^2) assuming no GP prior.
    XTAX, XTAY, YTAY, X0TAX0, XTAX0, X0TAY, X0TAX0_i, \
        XTAcorrX, XTAcorrY, YTAcorrY, LTXTAcorrY, XTAcorrXL, LTXTAcorrXL = \
        brsa._calc_sandwidge(XTY, XTDY, XTFY, 
                             YTY_diag, YTDY_diag, YTFY_diag,
                             XTX, XTDX, XTFX,
                             X0TX0, X0TDX0, X0TFX0,
                             XTX0, XTDX0, XTFX0,
                             X0TY, X0TDY, X0TFY,
                             L_full, rho1, n_V, n_base)
    assert np.ndim(XTAX) == 3, 'Dimension of XTAX is wrong by _calc_sandwidge()'
    assert XTAY.shape == XTY.shape, 'Shape of XTAY is wrong by _calc_sandwidge()'
    assert YTAY.shape == YTY_diag.shape, 'Shape of YTAY is wrong by _calc_sandwidge()'
    assert np.ndim(X0TAX0) == 3, 'Dimension of X0TAX0 is wrong by _calc_sandwidge()'
    assert np.ndim(XTAX0) == 3, 'Dimension of XTAX0 is wrong by _calc_sandwidge()'
    assert X0TAY.shape == X0TY.shape, 'Shape of X0TAX0 is wrong by _calc_sandwidge()'
    assert np.all(np.isfinite(X0TAX0_i)), 'Inverse of X0TAX0 includes NaN or Inf'
    ll0, deriv0 = brsa._loglike_AR1_diagV_fitV(param0_fitV[idx_param_fitV['log_SNR2']],
                                               XTAX, XTAY, YTAY, X0TAX0, XTAX0, X0TAY,
                                               X0TAX0_i, XTAcorrX, XTAcorrY, YTAcorrY, 
                                               LTXTAcorrY, XTAcorrXL, LTXTAcorrXL,
                                               L_full[l_idx], np.tan(rho1*np.pi/2),
                                               l_idx,n_C,n_T,n_V,n_run,n_base,
                                               idx_param_fitV,n_C,False,False)
    vec = np.zeros(np.size(param0_fitV[idx_param_fitV['log_SNR2']]))
    vec[idx_param_fitV['log_SNR2'][0]] = 1
    dd = nd.directionaldiff(lambda x: brsa._loglike_AR1_diagV_fitV(x, XTAX, XTAY, YTAY, X0TAX0, XTAX0, X0TAY,
                                                                   X0TAX0_i, XTAcorrX, XTAcorrY, YTAcorrY, 
                                                                   LTXTAcorrY, XTAcorrXL, LTXTAcorrXL,
                                                                   L_full[l_idx], np.tan(rho1*np.pi/2),
                                                                   l_idx, n_C, n_T, n_V, n_run, n_base,
                                                                   idx_param_fitV, n_C, False, False)[0],
                            param0_fitV[idx_param_fitV['log_SNR2']], vec)
    assert np.isclose(dd, np.dot(deriv0,vec), rtol=0.01), 'gradient of fitV wrt log(SNR2) incorrect for model without GP'

    # We test the gradient of _fitV wrt to log(SNR^2) assuming GP prior.
    ll0, deriv0 = brsa._loglike_AR1_diagV_fitV(param0_fitV, XTAX, XTAY, YTAY, X0TAX0, XTAX0, X0TAY,
                                               X0TAX0_i, XTAcorrX, XTAcorrY, YTAcorrY, 
                                               LTXTAcorrY, XTAcorrXL, LTXTAcorrXL,
                                               L_full[l_idx], np.tan(rho1*np.pi/2),
                                               l_idx,n_C,n_T,n_V,n_run,n_base,
                                               idx_param_fitV,n_C,True,True,
                                               dist2,inten_diff2,100,100)
    vec = np.zeros(np.size(param0_fitV))
    vec[idx_param_fitV['log_SNR2'][0]] = 1
    dd = nd.directionaldiff(lambda x: brsa._loglike_AR1_diagV_fitV(x, XTAX, XTAY, YTAY, X0TAX0, XTAX0, X0TAY,
                                                                   X0TAX0_i, XTAcorrX, XTAcorrY, YTAcorrY, 
                                                                   LTXTAcorrY, XTAcorrXL, LTXTAcorrXL,
                                                                   L_full[l_idx], np.tan(rho1*np.pi/2),
                                                                   l_idx, n_C, n_T, n_V, n_run, n_base,
                                                                   idx_param_fitV, n_C, True, True,
                                                                   dist2, inten_diff2,
                                                                   100, 100)[0], param0_fitV, vec)
    assert np.isclose(dd, np.dot(deriv0,vec), rtol=0.01), 'gradient of fitV srt log(SNR2) incorrect for model with GP'

    # We test the graident wrt spatial length scale parameter of GP prior
    vec = np.zeros(np.size(param0_fitV))
    vec[idx_param_fitV['c_space']] = 1
    dd = nd.directionaldiff(lambda x: brsa._loglike_AR1_diagV_fitV(x, XTAX, XTAY, YTAY, X0TAX0, XTAX0, X0TAY,
                                                                   X0TAX0_i, XTAcorrX, XTAcorrY, YTAcorrY, 
                                                                   LTXTAcorrY, XTAcorrXL, LTXTAcorrXL,
                                                                   L_full[l_idx], np.tan(rho1*np.pi/2),
                                                                   l_idx, n_C, n_T, n_V, n_run, n_base,
                                                                   idx_param_fitV, n_C, True, True,
                                                                   dist2, inten_diff2,
                                                                   100, 100)[0], param0_fitV, vec)
    assert np.isclose(dd, np.dot(deriv0,vec), rtol=0.01), 'gradient of fitV wrt spatial length scale of GP incorrect'

    # We test the graident wrt intensity length scale parameter of GP prior
    vec = np.zeros(np.size(param0_fitV))
    vec[idx_param_fitV['c_inten']] = 1
    dd = nd.directionaldiff(lambda x: brsa._loglike_AR1_diagV_fitV(x, XTAX, XTAY, YTAY, X0TAX0, XTAX0, X0TAY,
                                                                   X0TAX0_i, XTAcorrX, XTAcorrY, YTAcorrY, 
                                                                   LTXTAcorrY, XTAcorrXL, LTXTAcorrXL,
                                                                   L_full[l_idx], np.tan(rho1*np.pi/2),
                                                                   l_idx, n_C, n_T, n_V, n_run, n_base,
                                                                   idx_param_fitV, n_C, True, True,
                                                                   dist2, inten_diff2,
                                                                   100, 100)[0], param0_fitV, vec)
    assert np.isclose(dd, np.dot(deriv0,vec), rtol=0.01), 'gradient of fitV wrt intensity length scale of GP incorrect'

    # We test the graident on a random direction
    vec = np.random.randn(np.size(param0_fitV))
    vec = vec / np.linalg.norm(vec)
    dd = nd.directionaldiff(lambda x: brsa._loglike_AR1_diagV_fitV(x, XTAX, XTAY, YTAY, X0TAX0, XTAX0, X0TAY,
                                                                   X0TAX0_i, XTAcorrX, XTAcorrY, YTAcorrY, 
                                                                   LTXTAcorrY, XTAcorrXL, LTXTAcorrXL,
                                                                   L_full[l_idx], np.tan(rho1*np.pi/2),
                                                                   l_idx, n_C, n_T, n_V, n_run, n_base,
                                                                   idx_param_fitV, n_C, True, True,
                                                                   dist2, inten_diff2,
                                                                   100, 100)[0], param0_fitV, vec)
    assert np.isclose(dd, np.dot(deriv0,vec), rtol=0.01), 'gradient of fitV incorrect'
