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


def test_can_instantiate():
    import brainiak.reprsimil.brsa
    s = brainiak.reprsimil.brsa.GBRSA()
    assert s, "Invalid GBRSA instance!"

    s = brainiak.reprsimil.brsa.GBRSA(n_iter=40, rank=4, auto_nuisance=False,
                                      nureg_method='PCA',
                                      baseline_single=False, logS_range=1.0,
                                      SNR_bins=11, rho_bins=40, tol=2e-3,
                                      optimizer='CG', random_state=0,
                                      anneal_speed=20, SNR_prior='unif')
    assert s, "Invalid GBRSA instance!"


def test_fit():
    from brainiak.reprsimil.brsa import GBRSA
    import brainiak.utils.utils as utils
    import scipy.stats
    import numpy as np
    import os.path
    np.random.seed(10)
    file_path = os.path.join(os.path.dirname(__file__), "example_design.1D")
    # Load an example design matrix
    design = utils.ReadDesign(fname=file_path)

    # concatenate it by 1, 2, and 3 times, mimicking different length
    # of experiments for different participants
    n_run = [2, 1, 1]
    design_mat = [None] * 3
    n_T = [None] * 3
    n_V = [40, 60, 60]
    for i in range(3):
        design_mat[i] = np.tile(design.design_task[:, :-1], [n_run[i], 1])

    # start simulating some data
    n_C = np.size(design_mat[0], axis=1)

    noise_bot = 0.5
    noise_top = 1.5
    noise_level = [None] * 3

    # AR(1) coefficient
    rho1_top = 0.8
    rho1_bot = -0.2
    rho1 = [None] * 3

    # generating noise
    noise = [None] * 3

    # baseline
    inten = [None] * 3
    for i in range(3):
        design_mat[i] = np.tile(design.design_task[:, :-1], [n_run[i], 1])
        n_T[i] = n_run[i] * design.n_TR
        noise_level[i] = np.random.rand(
            n_V[i]) * (noise_top - noise_bot) + noise_bot
        # noise level is random.
        rho1[i] = np.random.rand(n_V[i]) * (rho1_top - rho1_bot) + rho1_bot

        noise[i] = np.zeros([n_T[i], n_V[i]])
        noise[i][0, :] = np.random.randn(
            n_V[i]) * noise_level[i] / np.sqrt(1 - rho1[i]**2)
        for i_t in range(1, n_T[i]):
            noise[i][i_t, :] = noise[i][i_t - 1, :] * rho1[i] + \
                np.random.randn(n_V[i]) * noise_level[i]
        noise[i] = noise[i] + \
            np.dot(np.random.randn(n_T[i], 2), np.random.randn(2, n_V[i]))
        inten[i] = np.random.rand(n_V[i]) * 20.0

    # ideal covariance matrix
    ideal_cov = np.zeros([n_C, n_C])
    ideal_cov = np.eye(n_C) * 0.6
    ideal_cov[0:4, 0:4] = 0.2
    for cond in range(0, 4):
        ideal_cov[cond, cond] = 2
    ideal_cov[5:9, 5:9] = 0.9
    for cond in range(5, 9):
        ideal_cov[cond, cond] = 1
    L_full = np.linalg.cholesky(ideal_cov)

    # generating signal
    snr_top = 5.0  # test with high SNR
    snr_bot = 1.0
    # snr = np.random.rand(n_V)*(snr_top-snr_bot)+snr_bot
    # Notice that accurately speaking this is not snr. the magnitude of signal
    # depends not only on beta but also on x.

    snr = [None] * 3
    signal = [None] * 3
    betas_simulated = [None] * 3
    scan_onsets = [None] * 3
    Y = [None] * 3
    for i in range(3):
        snr[i] = np.random.rand(n_V[i]) * (snr_top - snr_bot) + snr_bot
        sqrt_v = noise_level[i] * snr[i]
        betas_simulated[i] = np.dot(
            L_full, np.random.randn(n_C, n_V[i])) * sqrt_v
        signal[i] = np.dot(design_mat[i], betas_simulated[i])

        # Adding noise to signal as data
        Y[i] = signal[i] + noise[i] + inten[i]

        scan_onsets[i] = np.linspace(0, n_T[i], num=n_run[i] + 1)

    # Test fitting.
    n_nureg = 2
    gbrsa = GBRSA(n_iter=15, auto_nuisance=True, logS_range=0.5, SNR_bins=11,
                  rho_bins=16, n_nureg=n_nureg, optimizer='L-BFGS-B')

    gbrsa.fit(X=Y, design=design_mat, scan_onsets=scan_onsets)

    # Check that result is significantly correlated with the ideal covariance
    # matrix
    u_b = gbrsa.U_
    u_i = ideal_cov
    p = scipy.stats.spearmanr(u_b[np.tril_indices_from(u_b)],
                              u_i[np.tril_indices_from(u_i)])[1]
    assert p < 0.01, (
        "Fitted covariance matrix does not correlate with ideal covariance "
        "matrix!")
    # check that the recovered SNRs makes sense
    p = scipy.stats.pearsonr(gbrsa.nSNR_[0], snr[0])[1]
    assert p < 0.01, "Fitted SNR does not correlate with simulated SNR!"
    p = scipy.stats.pearsonr(gbrsa.sigma_[1], noise_level[1])[1]
    assert p < 0.01, (
        "Fitted noise level does not correlate with simulated noise level!")
    p = scipy.stats.pearsonr(gbrsa.rho_[2], rho1[2])[1]
    assert p < 0.01, (
        "Fitted AR(1) coefficient does not correlate with simulated values!")
    assert np.shape(gbrsa.X0_[1]) == (n_T[1], n_nureg + 1), "Wrong size of X0"

    Y_new = [None] * 3
    noise_new = [None] * 3
    for i in range(3):
        noise_new[i] = np.zeros([n_T[i], n_V[i]])
        noise_new[i][0, :] = np.random.randn(
            n_V[i]) * noise_level[i] / np.sqrt(1 - rho1[i]**2)
        for i_t in range(1, n_T[i]):
            noise_new[i][i_t, :] = noise_new[i][i_t - 1, :] * \
                rho1[i] + np.random.randn(n_V[i]) * noise_level[i]

        Y_new[i] = signal[i] + noise_new[i] + inten[i]
    ts, ts0 = gbrsa.transform(Y_new, scan_onsets=scan_onsets)
    [score, score_null] = gbrsa.score(
        X=Y_new, design=design_mat, scan_onsets=scan_onsets)
    [score_noise, score_null_noise] = gbrsa.score(
        X=noise_new, design=design_mat, scan_onsets=scan_onsets)
    for i in range(3):
        assert np.shape(ts[i]) == (n_T[i], n_C) and np.shape(
            ts0[i]) == (n_T[i], n_nureg + 1)
        p = scipy.stats.pearsonr(ts[i][:, 0], design_mat[i][:, 0])[1]
        assert p < 0.01, (
            "Recovered time series does not correlate with true time series!")

        assert score[i] > score_null[i], (
            "Full model does not win over null model on data containing "
            "signal")

        assert score_noise[i] < score_null_noise[i], (
            "Null model does not win over full model on data without signal")

    [score, score_null] = gbrsa.score(
        X=[None] * 3, design=design_mat, scan_onsets=scan_onsets)
    assert score == [None] * 3 and score_null == [None] * \
        3, "score did not return list of None when data is None"
    ts, ts0 = gbrsa.transform(X=[None] * 3, scan_onsets=scan_onsets)
    assert ts == [None] * 3 and ts0 == [None] * \
        3, "transform did not return list of None when data is None"


def test_gradient():
    from brainiak.reprsimil.brsa import GBRSA
    import brainiak.utils.utils as utils
    import numpy as np
    import os.path
    import numdifftools as nd

    np.random.seed(100)
    file_path = os.path.join(os.path.dirname(__file__), "example_design.1D")
    # Load an example design matrix
    design = utils.ReadDesign(fname=file_path)

    # concatenate it by 1, 2, and 3 times, mimicking different length
    # of experiments for different participants
    n_run = [1, 2, 1]
    design_mat = [None] * 3
    n_T = [None] * 3
    n_V = [30, 30, 20]
    for i in range(3):
        design_mat[i] = np.tile(design.design_task[:, :-1], [n_run[i], 1])
        n_T[i] = n_run[i] * design.n_TR

    # start simulating some data
    n_C = np.size(design_mat[0], axis=1)

    noise_bot = 0.5
    noise_top = 1.5
    noise_level = [None] * 3
    for i in range(3):
        noise_level[i] = np.random.rand(
            n_V[i]) * (noise_top - noise_bot) + noise_bot
    # noise level is random.

    # AR(1) coefficient
    rho1_top = 0.8
    rho1_bot = -0.2
    rho1 = [None] * 3

    # generating noise
    noise = [None] * 3

    # baseline
    inten = [None] * 3
    for i in range(3):
        rho1[i] = np.random.rand(n_V[i]) * (rho1_top - rho1_bot) + rho1_bot
        noise[i] = np.zeros([n_T[i], n_V[i]])
        noise[i][0, :] = np.random.randn(
            n_V[i]) * noise_level[i] / np.sqrt(1 - rho1[i]**2)
        for i_t in range(1, n_T[i]):
            noise[i][i_t, :] = noise[i][i_t - 1, :] * rho1[i] + \
                np.random.randn(n_V[i]) * noise_level[i]
        noise[i] = noise[i] + \
            np.dot(np.random.randn(n_T[i], 2), np.random.randn(2, n_V[i]))
        inten[i] = np.random.rand(n_V[i]) * 20.0

    # ideal covariance matrix
    ideal_cov = np.zeros([n_C, n_C])
    ideal_cov = np.eye(n_C) * 0.6
    ideal_cov[0:4, 0:4] = 0.2
    for cond in range(0, 4):
        ideal_cov[cond, cond] = 2
    ideal_cov[5:9, 5:9] = 0.9
    for cond in range(5, 9):
        ideal_cov[cond, cond] = 1
    L_full = np.linalg.cholesky(ideal_cov)

    # generating signal
    snr_top = 5.0  # test with high SNR
    snr_bot = 1.0
    # snr = np.random.rand(n_V)*(snr_top-snr_bot)+snr_bot
    # Notice that accurately speaking this is not snr. the magnitude of signal
    # depends not only on beta but also on x.

    snr = [None] * 3
    signal = [None] * 3
    betas_simulated = [None] * 3
    scan_onsets = [None] * 3
    Y = [None] * 3
    for i in range(3):
        snr[i] = np.random.rand(n_V[i]) * (snr_top - snr_bot) + snr_bot
        sqrt_v = noise_level[i] * snr[i]
        betas_simulated[i] = np.dot(
            L_full, np.random.randn(n_C, n_V[i])) * sqrt_v
        signal[i] = np.dot(design_mat[i], betas_simulated[i])

        # Adding noise to signal as data
        Y[i] = signal[i] + noise[i] + inten[i]

        scan_onsets[i] = np.linspace(0, n_T[i], num=n_run[i] + 1)

    # Get some initial fitting.
    SNR_bins = 11
    rho_bins = 20
    gbrsa = GBRSA(n_iter=3, rank=n_C, SNR_bins=SNR_bins,
                  rho_bins=rho_bins, logS_range=0.5)

    n_grid = SNR_bins * rho_bins
    half_log_det_X0TAX0 = [np.random.randn(n_grid) for i in range(3)]
    log_weights = np.random.randn(n_grid)
    log_fixed_terms = [np.random.randn(n_grid) for i in range(3)]
    l_idx = np.tril_indices(n_C)
    L_vec = np.random.randn(int(n_C * (n_C + 1) / 2))
    n_X0 = [2, 2, 2]
    s = np.linspace(1, SNR_bins, n_grid)
    a = np.linspace(0.5, 1, n_grid)
    s2XTAcorrX = [None] * 3
    YTAcorrY_diag = [None] * 3
    sXTAcorrY = [None] * 3
    # The calculations below are quite arbitrary and do not conform
    # to the model. They simply conform to the symmetry property and shape of
    # the matrix indicated by the model
    for i in range(3):
        YTAcorrY_diag[i] = np.sum(Y[i] * Y[i], axis=0) * a[:, None]
        s2XTAcorrX[i] = np.dot(design_mat[i].T, design_mat[
                               i]) * s[:, None, None]**2 * a[:, None, None]
        sXTAcorrY[i] = np.dot(design_mat[i].T, Y[i]) * \
            s[:, None, None] * a[:, None, None]

    # test if the gradients are correct
    print(log_fixed_terms)
    ll0, deriv0 = gbrsa._sum_loglike_marginalized(L_vec, s2XTAcorrX,
                                                  YTAcorrY_diag, sXTAcorrY,
                                                  half_log_det_X0TAX0,
                                                  log_weights, log_fixed_terms,
                                                  l_idx, n_C, n_T, n_V, n_X0,
                                                  n_grid, rank=None)
    # We test the gradient to the Cholesky factor
    vec = np.random.randn(np.size(L_vec))
    vec = vec / np.linalg.norm(vec)
    dd = nd.directionaldiff(
        lambda x: gbrsa._sum_loglike_marginalized(x, s2XTAcorrX, YTAcorrY_diag,
                                                  sXTAcorrY,
                                                  half_log_det_X0TAX0,
                                                  log_weights, log_fixed_terms,
                                                  l_idx, n_C, n_T, n_V, n_X0,
                                                  n_grid, rank=None)[0],
        L_vec,
        vec)
    assert np.isclose(dd, np.dot(deriv0, vec), rtol=1e-5), 'gradient incorrect'


def test_SNR_grids():
    import brainiak.reprsimil.brsa
    import numpy as np

    s = brainiak.reprsimil.brsa.GBRSA(SNR_prior='unif', SNR_bins=10)
    SNR_grids, SNR_weights = s._set_SNR_grids()
    assert (np.isclose(np.sum(SNR_weights), 1)
            and np.isclose(np.std(SNR_weights[1:-1]), 0)
            and np.all(SNR_weights > 0)
            and np.isclose(np.min(SNR_grids), 0)
            and np.all(SNR_grids >= 0)
            and np.isclose(np.max(SNR_grids), 1)
            ), 'SNR_weights or SNR_grids are incorrect for uniform prior'
    assert np.isclose(np.ptp(np.diff(SNR_grids[1:-1])), 0), \
        'SNR grids are not equally spaced for uniform prior'
    assert (np.size(SNR_grids) == np.size(SNR_weights)
            and np.size(SNR_grids) == 10
            ), ("size of SNR_grids or SNR_weights is not correct for uniform "
                "prior")

    s = brainiak.reprsimil.brsa.GBRSA(SNR_prior='lognorm', SNR_bins=35)
    SNR_grids, SNR_weights = s._set_SNR_grids()
    assert (np.all(SNR_grids >= 0)
            and np.isclose(np.sum(SNR_weights), 1)
            and np.all(SNR_weights > 0)
            and np.all(np.diff(SNR_grids) > 0)
            ), 'SNR_grids or SNR_weights not correct for log normal prior'

    s = brainiak.reprsimil.brsa.GBRSA(SNR_prior='exp')
    SNR_grids, SNR_weights = s._set_SNR_grids()
    assert (np.all(SNR_grids >= 0)
            and np.isclose(np.sum(SNR_weights), 1)
            and np.all(SNR_weights > 0)
            and np.all(np.diff(SNR_grids) > 0)
            ), 'SNR_grids or SNR_weights not correct for exponential prior'


def test_n_nureg():
    import brainiak.reprsimil.brsa
    import numpy as np
    noise = np.dot(np.random.randn(100, 8), np.random.randn(
        8, 30)) + np.random.randn(100, 30) * 0.001
    design = np.random.randn(100, 2)
    s = brainiak.reprsimil.brsa.GBRSA(n_iter=2)
    s.fit(X=noise, design=design)
    assert s.n_nureg_[0] == 8, 'n_nureg_ estimation is wrong in GBRSA'


def test_grid_flatten_num_int():
    # Check for numeric integration of SNR, and correctly flattening 2-D grids
    # to 1-D grid.
    import brainiak.reprsimil.brsa
    import brainiak.utils.utils as utils
    import numpy as np
    import scipy.special
    n_V = 30
    n_T = 50
    n_C = 3
    design = np.random.randn(n_T, n_C)
    U_simu = np.asarray([[1.0, 0.1, 0.0], [0.1, 1.0, 0.2], [0.0, 0.2, 1.0]])
    L_simu = np.linalg.cholesky(U_simu)
    SNR = np.random.exponential(size=n_V)
    beta = np.dot(L_simu, np.random.randn(n_C, n_V)) * SNR
    noise = np.random.randn(n_T, n_V)
    Y = np.dot(design, beta) + noise
    X = design
    X_base = None
    scan_onsets = [0]

    s = brainiak.reprsimil.brsa.GBRSA(n_iter=1, auto_nuisance=False,
                                      SNR_prior='exp')
    s.fit(X=[Y], design=[design])
    rank = n_C
    l_idx, rank = s._chol_idx(n_C, rank)
    L = np.zeros((n_C, rank))
    n_l = np.size(l_idx[0])
    current_vec_U_chlsk_l = s.random_state_.randn(n_l) * 10
    L[l_idx] = current_vec_U_chlsk_l

    # Now we change the grids for SNR and rho for testing.
    s.SNR_bins = 2
    s.rho_bins = 2
    SNR_grids, SNR_weights = s._set_SNR_grids()
    # rho_grids, rho_weights = s._set_rho_grids()
    rho_grids = np.ones(2) * 0.1
    rho_weights = np.ones(2) / 2
    # We purposefully set all rhos to be equal to test flattening of
    # grids.
    n_grid = s.SNR_bins * s.rho_bins

    D, F, run_TRs, n_run = s._prepare_DF(
        n_T, scan_onsets=scan_onsets)
    XTY, XTDY, XTFY, YTY_diag, YTDY_diag, YTFY_diag, XTX, XTDX, XTFX \
        = s._prepare_data_XY(X, Y, D, F)
    X0TX0, X0TDX0, X0TFX0, XTX0, XTDX0, XTFX0, X0TY, X0TDY, X0TFY, X0, \
        X_base, n_X0, idx_DC = s._prepare_data_XYX0(
            X, Y, X_base, None, D, F, run_TRs, no_DC=False)

    X0TAX0, X0TAX0_i, XTAcorrX, XTAcorrY, YTAcorrY_diag, X0TAY, XTAX0 \
        = s._precompute_ar1_quad_forms_marginalized(
            XTY, XTDY, XTFY, YTY_diag, YTDY_diag, YTFY_diag, XTX,
            XTDX, XTFX, X0TX0, X0TDX0, X0TFX0, XTX0, XTDX0, XTFX0,
            X0TY, X0TDY, X0TFY, rho_grids, n_V, n_X0)

    half_log_det_X0TAX0, X0TAX0, X0TAX0_i, s2XTAcorrX, YTAcorrY_diag, \
        sXTAcorrY, X0TAY, XTAX0 = s._matrix_flattened_grid(
            X0TAX0, X0TAX0_i, SNR_grids, XTAcorrX, YTAcorrY_diag, XTAcorrY,
            X0TAY, XTAX0, n_C, n_V, n_X0, n_grid)

    assert (half_log_det_X0TAX0[0] == half_log_det_X0TAX0[1]
            and half_log_det_X0TAX0[2] == half_log_det_X0TAX0[3]
            and half_log_det_X0TAX0[0] == half_log_det_X0TAX0[2]
            ), '_matrix_flattened_grid has mistake with half_log_det_X0TAX0'
    assert (np.array_equal(X0TAX0[0, :, :], X0TAX0[1, :, :])
            and np.array_equal(X0TAX0[2, :, :], X0TAX0[3, :, :])
            and np.array_equal(X0TAX0[0, :, :], X0TAX0[2, :, :])
            ), '_matrix_flattened_grid has mistake X0TAX0'
    assert (np.array_equal(X0TAX0_i[0, :, :], X0TAX0_i[1, :, :])
            and np.array_equal(X0TAX0_i[2, :, :], X0TAX0_i[3, :, :])
            and np.array_equal(X0TAX0_i[0, :, :], X0TAX0_i[2, :, :])
            ), '_matrix_flattened_grid has mistake X0TAX0_i'
    assert np.allclose(
        np.dot(X0TAX0[0, :, :], X0TAX0_i[0, :, :]),
        np.eye(n_X0)
        ), 'X0TAX0_i is not inverse of X0TAX0'
    assert (np.array_equal(YTAcorrY_diag[0, :], YTAcorrY_diag[1, :])
            and np.array_equal(YTAcorrY_diag[2, :], YTAcorrY_diag[3, :])
            and np.array_equal(YTAcorrY_diag[0, :], YTAcorrY_diag[2, :])
            ), '_matrix_flattened_grid has mistake YTAcorrY_diag'
    assert (np.array_equal(sXTAcorrY[0, :, :], sXTAcorrY[1, :, :])
            and np.array_equal(sXTAcorrY[2, :, :], sXTAcorrY[3, :, :])
            and not np.array_equal(sXTAcorrY[0, :, :], sXTAcorrY[2, :, :])
            ), '_matrix_flattened_grid has mistake sXTAcorrY'
    assert (np.array_equal(X0TAY[0, :, :], X0TAY[1, :, :])
            and np.array_equal(X0TAY[2, :, :], X0TAY[3, :, :])
            and np.array_equal(X0TAY[0, :, :], X0TAY[2, :, :])
            ), '_matrix_flattened_grid has mistake X0TAY'
    assert (np.array_equal(XTAX0[0, :, :], XTAX0[1, :, :])
            and np.array_equal(XTAX0[2, :, :], XTAX0[3, :, :])
            and np.array_equal(XTAX0[0, :, :], XTAX0[2, :, :])
            ), '_matrix_flattened_grid has mistake XTAX0'

    # Now we test the other way
    rho_grids, rho_weights = s._set_rho_grids()
    # rho_grids, rho_weights = s._set_rho_grids()
    SNR_grids = np.ones(2) * 0.1
    SNR_weights = np.ones(2) / 2
    # We purposefully set all SNR to be equal to test flattening of
    # grids.
    X0TAX0, X0TAX0_i, XTAcorrX, XTAcorrY, YTAcorrY_diag, X0TAY, XTAX0 \
        = s._precompute_ar1_quad_forms_marginalized(
            XTY, XTDY, XTFY, YTY_diag, YTDY_diag, YTFY_diag, XTX,
            XTDX, XTFX, X0TX0, X0TDX0, X0TFX0, XTX0, XTDX0, XTFX0,
            X0TY, X0TDY, X0TFY, rho_grids, n_V, n_X0)

    half_log_det_X0TAX0, X0TAX0, X0TAX0_i, s2XTAcorrX, YTAcorrY_diag, \
        sXTAcorrY, X0TAY, XTAX0 = s._matrix_flattened_grid(
            X0TAX0, X0TAX0_i, SNR_grids, XTAcorrX, YTAcorrY_diag, XTAcorrY,
            X0TAY, XTAX0, n_C, n_V, n_X0, n_grid)

    assert (half_log_det_X0TAX0[0] == half_log_det_X0TAX0[2]
            and half_log_det_X0TAX0[1] == half_log_det_X0TAX0[3]
            and not half_log_det_X0TAX0[0] == half_log_det_X0TAX0[1]
            ), '_matrix_flattened_grid has mistake with half_log_det_X0TAX0'
    assert (np.array_equal(X0TAX0[0, :, :], X0TAX0[2, :, :])
            and np.array_equal(X0TAX0[1, :, :], X0TAX0[3, :, :])
            and not np.array_equal(X0TAX0[0, :, :], X0TAX0[1, :, :])
            ), '_matrix_flattened_grid has mistake X0TAX0'
    assert (np.array_equal(X0TAX0_i[0, :, :], X0TAX0_i[2, :, :])
            and np.array_equal(X0TAX0_i[1, :, :], X0TAX0_i[3, :, :])
            and not np.array_equal(X0TAX0_i[0, :, :], X0TAX0_i[1, :, :])
            ), '_matrix_flattened_grid has mistake X0TAX0_i'
    assert np.allclose(
        np.dot(X0TAX0[0, :, :], X0TAX0_i[0, :, :]),
        np.eye(n_X0)
        ), 'X0TAX0_i is not inverse of X0TAX0'
    assert (np.array_equal(YTAcorrY_diag[0, :], YTAcorrY_diag[2, :])
            and np.array_equal(YTAcorrY_diag[1, :], YTAcorrY_diag[3, :])
            and not np.array_equal(YTAcorrY_diag[0, :],
                                   YTAcorrY_diag[1, :])
            ), '_matrix_flattened_grid has mistake YTAcorrY_diag'
    assert (np.array_equal(sXTAcorrY[0, :, :], sXTAcorrY[2, :, :])
            and np.array_equal(sXTAcorrY[1, :, :], sXTAcorrY[3, :, :])
            and not np.array_equal(sXTAcorrY[0, :, :], sXTAcorrY[1, :, :])
            ), '_matrix_flattened_grid has mistake sXTAcorrY'
    assert (np.array_equal(X0TAY[0, :, :], X0TAY[2, :, :])
            and np.array_equal(X0TAY[1, :, :], X0TAY[3, :, :])
            and not np.array_equal(X0TAY[0, :, :], X0TAY[1, :, :])
            ), '_matrix_flattened_grid has mistake X0TAY'
    assert (np.array_equal(XTAX0[0, :, :], XTAX0[2, :, :])
            and np.array_equal(XTAX0[1, :, :], XTAX0[3, :, :])
            and not np.array_equal(XTAX0[0, :, :], XTAX0[1, :, :])
            ), '_matrix_flattened_grid has mistake XTAX0'

    # Now test the integration over SNR
    s.SNR_bins = 50
    s.rho_bins = 1
    SNR_grids, SNR_weights = s._set_SNR_grids()
    rho_grids, rho_weights = s._set_rho_grids()
    n_grid = s.SNR_bins * s.rho_bins

    def setup_for_test():
        # This function will be re-used to set up the variables necessary for
        # testing.

        X0TAX0, X0TAX0_i, XTAcorrX, XTAcorrY, YTAcorrY_diag, X0TAY, XTAX0 \
            = s._precompute_ar1_quad_forms_marginalized(
                XTY, XTDY, XTFY, YTY_diag, YTDY_diag, YTFY_diag, XTX,
                XTDX, XTFX, X0TX0, X0TDX0, X0TFX0, XTX0, XTDX0, XTFX0,
                X0TY, X0TDY, X0TFY, rho_grids, n_V, n_X0)

        half_log_det_X0TAX0, X0TAX0, X0TAX0_i, s2XTAcorrX, YTAcorrY_diag, \
            sXTAcorrY, X0TAY, XTAX0 = s._matrix_flattened_grid(
                X0TAX0, X0TAX0_i, SNR_grids, XTAcorrX, YTAcorrY_diag, XTAcorrY,
                X0TAY, XTAX0, n_C, n_V, n_X0, n_grid)

        log_weights = np.reshape(
            np.log(SNR_weights[:, None]) + np.log(rho_weights), n_grid)
        all_rho_grids = np.reshape(np.repeat(
            rho_grids[None, :], s.SNR_bins, axis=0), n_grid)
        log_fixed_terms = - (n_T - n_X0) / 2 * np.log(2 * np.pi) + n_run \
            / 2 * np.log(1 - all_rho_grids**2) + scipy.special.gammaln(
                (n_T - n_X0 - 2) / 2) + (n_T - n_X0 - 2) / 2 * np.log(2)
        return s2XTAcorrX, YTAcorrY_diag, sXTAcorrY, half_log_det_X0TAX0, \
            log_weights, log_fixed_terms

    (s2XTAcorrX, YTAcorrY_diag, sXTAcorrY, half_log_det_X0TAX0, log_weights,
     log_fixed_terms) = setup_for_test()
    LL_total, _ = s._loglike_marginalized(current_vec_U_chlsk_l, s2XTAcorrX,
                                          YTAcorrY_diag, sXTAcorrY,
                                          half_log_det_X0TAX0, log_weights,
                                          log_fixed_terms, l_idx, n_C, n_T,
                                          n_V, n_X0, n_grid, rank=rank)
    LL_total = - LL_total
    # Now we re-calculate using scipy.integrate
    s.SNR_bins = 100
    SNR_grids = np.linspace(0, 12, s.SNR_bins)
    SNR_weights = np.exp(- SNR_grids)
    SNR_weights = SNR_weights / np.sum(SNR_weights)
    n_grid = s.SNR_bins * s.rho_bins

    (s2XTAcorrX, YTAcorrY_diag, sXTAcorrY, half_log_det_X0TAX0, log_weights,
     log_fixed_terms) = setup_for_test()
    LL_raw, _, _, _ = s._raw_loglike_grids(L, s2XTAcorrX, YTAcorrY_diag,
                                           sXTAcorrY, half_log_det_X0TAX0,
                                           log_weights, log_fixed_terms,
                                           n_C, n_T, n_V, n_X0,
                                           n_grid, rank)
    result_sum, max_value, result_exp = utils.sumexp_stable(LL_raw)
    scipy_sum = scipy.integrate.simps(y=result_exp, axis=0)
    LL_total_scipy = np.sum(np.log(scipy_sum) + max_value)

    tol = 1e-3
    assert(np.isclose(LL_total_scipy, LL_total, rtol=tol)), \
        'Error of log likelihood calculation exceeds the tolerance'

    # Now test the log normal prior
    s = brainiak.reprsimil.brsa.GBRSA(n_iter=1, auto_nuisance=False,
                                      SNR_prior='lognorm')
    s.SNR_bins = 50
    s.rho_bins = 1
    SNR_grids, SNR_weights = s._set_SNR_grids()
    rho_grids, rho_weights = s._set_rho_grids()
    n_grid = s.SNR_bins * s.rho_bins

    (s2XTAcorrX, YTAcorrY_diag, sXTAcorrY, half_log_det_X0TAX0, log_weights,
     log_fixed_terms) = setup_for_test()
    LL_total, _ = s._loglike_marginalized(current_vec_U_chlsk_l, s2XTAcorrX,
                                          YTAcorrY_diag, sXTAcorrY,
                                          half_log_det_X0TAX0, log_weights,
                                          log_fixed_terms, l_idx, n_C, n_T,
                                          n_V, n_X0, n_grid, rank=rank)
    LL_total = - LL_total
    # Now we re-calculate using scipy.integrate
    s.SNR_bins = 400
    SNR_grids = np.linspace(1e-8, 20, s.SNR_bins)
    log_SNR_weights = scipy.stats.lognorm.logpdf(SNR_grids, s=s.logS_range)
    result_sum, max_value, result_exp = utils.sumexp_stable(
        log_SNR_weights[:, None])
    SNR_weights = np.squeeze(result_exp / result_sum)
    n_grid = s.SNR_bins * s.rho_bins

    (s2XTAcorrX, YTAcorrY_diag, sXTAcorrY, half_log_det_X0TAX0, log_weights,
     log_fixed_terms) = setup_for_test()
    LL_raw, _, _, _ = s._raw_loglike_grids(L, s2XTAcorrX, YTAcorrY_diag,
                                           sXTAcorrY, half_log_det_X0TAX0,
                                           log_weights, log_fixed_terms,
                                           n_C, n_T, n_V, n_X0,
                                           n_grid, rank)
    result_sum, max_value, result_exp = utils.sumexp_stable(LL_raw)
    scipy_sum = scipy.integrate.simps(y=result_exp, axis=0)
    LL_total_scipy = np.sum(np.log(scipy_sum) + max_value)

    tol = 1e-3
    assert(np.isclose(LL_total_scipy, LL_total, rtol=tol)), \
        'Error of log likelihood calculation exceeds the tolerance'
