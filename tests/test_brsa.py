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
    import brainiak.brsa
    import numpy as np
    s = brainiak.bayesianrsa.brsa.BRSA()
    assert s, "Invalid BRSA instance!"

    

    voxels = 100
    samples = 500
    features = 3

    s = brainiak.representationsimilarity.brsa.BRSA(n_iter=50, rank=5, GP_space=True, GP_inten=True, tolx=2e-3,verbose=True,\
                pad_DC=False,epsilon=0.001,space_smooth_range=10.0,inten_smooth_range=100.0)
    assert s, "Invalid BRSA instance!"

def test_fit():
    from brainiak.bayesianrsa.brsa import BRSA
    import brainiak.utils.read_design as read_design
    import scipy.stats
    import numpy as np
    # Load an example design matrix
    design = read_design.design_matrix('example_design.1D')
    # concatenate it by 4 times, mimicking 4 runs of itenditcal timing
    design.design_used = np.tile(design.design_used[:,0:17],[4,1])
    design.n_TR = design.n_TR * 4
    
    
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
    ideal_cov[0,0] = 0.2
    ideal_cov[5:9,5:9] = 0.6
    for cond in range(5,9):
        ideal_cov[cond,cond] = 1
    idx = np.where(np.sum(np.abs(ideal_cov),axis=0)>0)[0]
    L_full = np.linalg.cholesky(ideal_cov)        

    # generating signal
    snr_level = 5.0 # test with high SNR    
    snr = np.random.rand(n_V)*(snr_top-snr_bot)+snr_bot
    # Notice that accurately speaking this is not snr. the magnitude of signal depends
    # not only on beta but also on x.
    inten = np.random.randn(n_V) * 5.0

    # parameters of Gaussian process to generate pseuso SNR
    tau = 0.5
    smooth_width = 5.0
    inten_kernel = 1.0
    
    coords = np.arange(0,n_V)
    coords_tile = np.tile(coords,[n_V,1])
    dist2 = (coords_tile-coords_tile.T)**2
    
    inten_tile = np.tile(inten,[n_V,1])
    inten_diff2 = (inten_tile-inten_tile.T)**2

    K = np.exp(-dist2/smooth_width**2/2.0 * 0.9 -inten_diff2/inten_kernel**2/2.0 * 0.1) * tau**2 + np.eye(n_V)*tau**2*0.001

    L = np.linalg.cholesky(K)
    snr = np.exp(np.dot(L,np.random.randn(n_V))) * snr_level
    sqrt_v = noise_level*snr
    betas_simulated = np.dot(L_full,np.random.randn(n_C,n_V)) * sqrt_v
    signal = np.dot(design.design_used,betas_simulated)
    
    
    Y = signal + noise


    scan_onsets = np.linspace(0,design.n_TR,num=5)

    
    brsa = BRSA(GP_space=True,GP_inten=True,verbose=True,n_iter = 20,rank=rank)

    brsa.fit(X=Y,design=design.design_used,scan_onsets=scan_onsets,coords=coords,inten=inten,inten_weight=0.1)
    
    # Check that result is significantly correlated with the ideal covariance matrix
    u_b = brsa.U_[1:,1:]
    u_i = ideal_cov[1:,1:]
    p = scipy.stats.spearmanr(u_b[np.tril_indices_from(u_b,k=-1)],u_i[np.tril_indices_from(u_i,k=-1)]).pvalue 
    assert p < 0.01, "Fitted covariance matrix does not correlate with ideal covariance matrix!"
    # check that the recovered SNR makes sense
    p = scipy.stats.pearsonr(brsa.nSNR_,snr).pvalue
    assert p < 0.05, "Fitted SNR does not correlate with simualted SNR!"
    assert np.isclose(np.mean(np.log(brsa.nSNR_))), "nSNR_ not normalized!"
    
    
    
    # test if gradient is correct
    XTY,XTDY,XTFY,YTY_diag, YTDY_diag, YTFY_diag, XTX, XTDX, XTFX = brsa._prepare_data(design.design_used,Y,n_T,n_V,scan_onsets)
    n_l = n_C*(n_C+1)/2
    param0_fitU = np.random.randn(n_l+n_V) * 0.1
    param0_fitV = np.random.randn(n_V+2) * 0.1
    l_idx = np.tril_indices(n_C)
    perturb = 1e-6
    ll0, deriv0 = brsa._loglike_y_AR1_diagV_fitU(param0_fitU, XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag, \
                XTY, XTDY, XTFY, np.log(snr)*2,  l_idx,n_C,n_T,n_V)
    param1_fitU = param0_fitU.copy()
    param1_fitU[0] = param1_fitU[0]+perturb
    ll1, deriv1 = brsa._loglike_y_AR1_diagV_fitU(param1_fitU, XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag, \
                XTY, XTDY, XTFY, np.log(snr)*2, l_idx,n_C,n_T,n_V)
    num_diff = ll1-ll0
    ana_diff = deriv0[0]*perturb
    assert np.abs(num_diff-ana_diff)/np.abs(ana_diff) < 0.1, "Gradient of L is not right"
    
    param1_fitU = param0_fitU.copy()
    param1_fitU[n_l] = param1_fitU[n_l]+perturb
    ll1, deriv1 = brsa._loglike_y_AR1_diagV_fitU(param1_fitU, XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag, \
                XTY, XTDY, XTFY, np.log(snr)*2, l_idx,n_C,n_T,n_V)
    num_diff = ll1-ll0
    ana_diff = deriv0[n_l]*perturb
    assert np.abs(num_diff-ana_diff)/np.abs(ana_diff) < 0.1, "Gradient of AR(1) coefficient reparametrization is not right"
    
    ll0, deriv0 = brsa._loglike_y_AR1_diagV_fitV(param0_fitV, XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag, \
                XTY, XTDY, XTFY, L_full[l_idx], np.tan(rho1*np.pi/2), l_idx,n_C,n_T,n_V,n_C,True,True,\
                dist2,inten_diff2,100,100,0.1)
    param1_fitV = param0_fitV.copy()
    param1_fitV[0] = param1_fitV[0]+perturb
    ll1, deriv1 = brsa._loglike_y_AR1_diagV_fitV(param1_fitV, XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag, \
                XTY, XTDY, XTFY, L_full[l_idx], np.tan(rho1*np.pi/2), l_idx,n_C,n_T,n_V,n_C,True,True,\
                dist2,inten_diff2,100,100,0.1)
    num_diff = ll1-ll0
    ana_diff = deriv0[0]*perturb
    assert np.abs(num_diff-ana_diff)/np.abs(ana_diff) < 0.1, "Gradient of log_SNR2 parameter is not right"
    
    param1_fitV = param0_fitV.copy()
    param1_fitV[n_V-1] = param1_fitV[n_V-1]+perturb
    ll1, deriv1 = brsa._loglike_y_AR1_diagV_fitV(param1_fitV, XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag, \
                XTY, XTDY, XTFY, L_full[l_idx], np.tan(rho1*np.pi/2), l_idx,n_C,n_T,n_V,n_C,True,True,\
                dist2,inten_diff2,100,100,0.1)
    num_diff = ll1-ll0
    ana_diff = deriv0[n_V-1]*perturb
    assert np.abs(num_diff-ana_diff)/np.abs(ana_diff) < 0.1, "Gradient of GP spatial lengths scale parameter is not right"
    
    param1_fitV = param0_fitV.copy()
    param1_fitV[n_V] = param1_fitV[n_V]+perturb
    ll1, deriv1 = brsa._loglike_y_AR1_diagV_fitV(param1_fitV, XTX, XTDX, XTFX, YTY_diag, YTDY_diag, YTFY_diag, \
                XTY, XTDY, XTFY, L_full[l_idx], np.tan(rho1*np.pi/2), l_idx,n_C,n_T,n_V,n_C,True,True,\
                dist2,inten_diff2,100,100,0.1)
    num_diff = ll1-ll0
    ana_diff = deriv0[n_V]*perturb
    assert np.abs(num_diff-ana_diff)/np.abs(ana_diff) < 0.1, "Gradient of GP intensity lengths scale parameter is not right"
    
    
    

    
    