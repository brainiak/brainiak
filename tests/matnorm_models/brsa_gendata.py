import numpy as np
from scipy.stats import norm
from numpy.linalg import cholesky
import scipy.spatial.distance as spdist
from brainiak.matnorm_base.helpers import rmn


def gen_U_nips2016_example():

    n_C = 16
    U = np.zeros([n_C, n_C])
    U = np.eye(n_C) * 0.6
    U[8:12, 8:12] = 0.8
    for cond in range(8, 12):
        U[cond,cond] = 1

    return U

def gen_distMat(n_T): 

    x = np.reshape(np.arange(n_T), (n_T, 1))

    r = np.reshape(x*x, [-1, 1])
    D = np.sqrt(-2* x @ x.T + r + r.T)
    return D

def gen_brsa_data_from_model(U, n_T, n_V, space_cov, time_cov, n_nureg):

    n_C = U.shape[0]
    # the expression in the derivation says
    # Y ~ MN(X * beta + X_0 * beta_0, temmporal_cov, spatial_cov)
    # X is n_T by n_C
    # beta is n_C by n_V
    # ell is s_size by phi_size 
    # s is n_obs by s_size
    # psi is n_obs by psi_size 
    # note that s_size is the dimensionality of the stimulus, not the number
    # of stimuli. In SDT there are always 2 stimuli

    beta = rmn(U,np.eye(n_V))
    X = rmn(np.eye(n_T),np.eye(n_C))

    beta_0 = rmn(np.eye(n_nureg),np.eye(n_V))
    X_0 = rmn(np.eye(n_T),np.eye(n_nureg))

    Y_hat = X @ beta  + X_0 @ beta_0

    # rho = np.abs(norm.rvs())

    # sigma_ar = np.abs(norm.rvs())

    # sigma_s = np.abs(norm.rvs(size=n_V))

    # spatial_cov = np.diag(np.ones(n_V) * sigma_s)

    # temporal_cov = np.exp(-rho * gen_distMat(n_T)) + np.diag(np.ones(n_T))* sigma_ar
    
    Y = Y_hat + rmn(time_cov, space_cov)

    sizes = {"n_C":n_C, "n_T":n_T, "n_V":n_V}

    # train=  {"beta":beta,"X" : X, "Y":Y, "U":U, 'sigma_ar':sigma_ar, 'sigma_s':sigma_s, 'rho':rho, 'X_0':X_0}
    train=  {"beta":beta,"X" : X, "Y":Y, "U":U,'X_0':X_0}

    return train, sizes

def gen_brsa_data_brainiak_example(design, n_run):

    design.n_TR = design.n_TR * n_run
    design.design_task = np.tile(design.design_task[:,:-1],
                                 [n_run, 1])

    n_C = np.size(design.design_task, axis=1)
    # The total number of conditions.
    ROI_edge = 20
    # We simulate "ROI" of a square shape
    n_V = ROI_edge**2
    # The total number of simulated voxels
    n_T = design.n_TR
    # The total number of time points,
    # after concatenating all fMRI runs

    noise_bot = 0.5
    noise_top = 1.5
    noise_level = np.random.rand(n_V) * \
        (noise_top - noise_bot) + noise_bot
    # The standard deviation of the noise is in the range of [noise_bot, noise_top]
    # In fact, we simulate autocorrelated noise with AR(1) model. So the noise_level reflects
    # the independent additive noise at each time point (the "fresh" noise)

    # AR(1) coefficient
    rho1_top = 0.8
    rho1_bot = -0.2
    rho1 = np.random.rand(n_V) \
        * (rho1_top - rho1_bot) + rho1_bot

    noise_smooth_width = 10.0
    coords = np.mgrid[0:ROI_edge, 0:ROI_edge, 0:1]
    coords_flat = np.reshape(coords,[3, n_V]).T
    dist2 = spdist.squareform(spdist.pdist(coords_flat, 'sqeuclidean'))

    # generating noise
    K_noise = noise_level[:, np.newaxis] \
        * (np.exp(-dist2 / noise_smooth_width**2 / 2.0) \
           + np.eye(n_V) * 0.1) * noise_level

    L_noise = np.linalg.cholesky(K_noise)
    noise = np.zeros([n_T, n_V])
    noise[0, :] = np.dot(L_noise, np.random.randn(n_V))\
        / np.sqrt(1 - rho1**2)
    for i_t in range(1, n_T):
        noise[i_t, :] = noise[i_t - 1, :] * rho1 \
            + np.dot(L_noise,np.random.randn(n_V))
    # For each voxel, the noise follows AR(1) process:
    # fresh noise plus a dampened version of noise at
    # the previous time point.
    noise = noise + np.random.randn(n_V)
    # ideal covariance matrix

    ideal_cov = np.zeros([n_C, n_C])
    ideal_cov = np.eye(n_C) * 0.6
    ideal_cov[8:12, 8:12] = 0.8
    for cond in range(8, 12):
        ideal_cov[cond,cond] = 1

    std_diag = np.diag(ideal_cov)**0.5
    ideal_corr = ideal_cov / std_diag / std_diag[:, None]

    L_full = np.linalg.cholesky(ideal_cov)        

    # generating signal
    snr_level = 0.6
    # Notice that accurately speaking this is not SNR.
    # The magnitude of signal depends not only on beta but also on x.
    # (noise_level*snr_level)**2 is the factor multiplied
    # with ideal_cov to form the covariance matrix from which
    # the response amplitudes (beta) of a voxel are drawn from.

    tau = 0.8
    # magnitude of Gaussian Process from which the log(SNR) is drawn
    smooth_width = 3.0
    # spatial length scale of the Gaussian Process, unit: voxel
    inten_kernel = 4.0
    # intensity length scale of the Gaussian Process
    # Slightly counter-intuitively, if this parameter is very large,
    # say, much larger than the range of intensities of the voxels,
    # then the smoothness has much small dependency on the intensity.


    inten = np.random.rand(n_V) * 20.0
    # For simplicity, we just assume that the intensity
    # of all voxels are uniform distributed between 0 and 20
    # parameters of Gaussian process to generate pseuso SNR
    # For curious user, you can also try the following commond
    # to see what an example snr map might look like if the intensity
    # grows linearly in one spatial direction

    # inten = coords_flat[:,0] * 2


    inten_tile = np.tile(inten, [n_V, 1])
    inten_diff2 = (inten_tile - inten_tile.T)**2

    K = np.exp(-dist2 / smooth_width**2 / 2.0 
               - inten_diff2 / inten_kernel**2 / 2.0) * tau**2 \
        + np.eye(n_V) * tau**2 * 0.001
    # A tiny amount is added to the diagonal of
    # the GP covariance matrix to make sure it can be inverted
    L = np.linalg.cholesky(K)
    snr = np.exp(np.dot(L, np.random.randn(n_V))) * snr_level
    sqrt_v = noise_level * snr
    betas_simulated = np.dot(L_full, np.random.randn(n_C, n_V)) * sqrt_v
    signal = np.dot(design.design_task, betas_simulated)


    Y = signal + noise 

    scan_onsets = np.int32(np.linspace(0, design.n_TR,num=n_run + 1)[: -1])

    sizes = {"n_C":n_C, "n_T":n_T, "n_V":n_V}

    train=  {"beta":betas_simulated,"X" : design.design_task, "Y":Y, "U":ideal_cov, "coords":coords_flat,\
     "scan_onsets":scan_onsets, "inten":inten}

    return train, sizes
    