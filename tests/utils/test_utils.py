#  Copyright 2016 Intel Corporation
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

import pytest


def test_tri_sym_convert():
    from brainiak.utils.utils import from_tri_2_sym, from_sym_2_tri
    import numpy as np

    sym = np.random.rand(3, 3)
    tri = from_sym_2_tri(sym)
    assert tri.shape[0] == 6,\
        "from_sym_2_tri returned wrong result!"
    sym1 = from_tri_2_sym(tri, 3)
    assert sym1.shape[0] == sym1.shape[1],\
        "from_tri_2_sym returned wrong shape!"
    tri1 = from_sym_2_tri(sym1)
    assert np.array_equiv(tri, tri1),\
        "from_sym_2_tri returned wrong result!"


def test_sumexp():
    from brainiak.utils.utils import sumexp_stable
    import numpy as np

    data = np.array([[1, 1], [0, 1]])
    sums, maxs, exps = sumexp_stable(data)
    assert sums.size == data.shape[1], (
        "Invalid sum(exp(v)) computation (wrong # samples in sums)")
    assert exps.shape[0] == data.shape[0], (
        "Invalid exp(v) computation (wrong # features)")
    assert exps.shape[1] == data.shape[1], (
        "Invalid exp(v) computation (wrong # samples)")
    assert maxs.size == data.shape[1], (
        "Invalid max computation (wrong # samples in maxs)")


def test_concatenate_not_none():
    from brainiak.utils.utils import concatenate_not_none
    import numpy as np
    arrays = [None] * 5

    arrays[1] = np.array([0, 1, 2])
    arrays[3] = np.array([3, 4])

    r = concatenate_not_none(arrays, axis=0)

    assert np.all(np.arange(5) == r), (
        "Invalid concatenation of a list of arrays")


def test_cov2corr():
    from brainiak.utils.utils import cov2corr
    import numpy as np
    cov = np.array([[4, 3, 0], [3, 9, 0], [0, 0, 1]])
    corr = cov2corr(cov)
    assert np.allclose(corr,
                       np.array([[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]])), (
        "Converting from covariance matrix to correlation incorrect")


def test_ReadDesign():
    from brainiak.utils.utils import ReadDesign
    import numpy as np
    import os.path
    file_path = os.path.join(os.path.dirname(__file__), "example_design.1D")
    design = ReadDesign(fname=file_path, include_orth=False,
                        include_pols=False)
    assert design, 'Failed to read design matrix'
    assert design.reg_nuisance is None, \
        'Nuiance regressor is not None when include_orth and include_pols are'\
        ' both set to False'
    read = ReadDesign()
    assert read, 'Failed to initialize an instance of the class'
    design = ReadDesign(fname=file_path, include_orth=True, include_pols=True)
    assert np.size(design.cols_nuisance) == 10, \
        'Mistake in counting the number of nuiance regressors'
    assert np.size(design.cols_task) == 17, \
        'Mistake in counting the number of task conditions'
    assert (np.shape(design.reg_nuisance)[0]
            == np.shape(design.design_task)[0]
            ), 'The number of time points in nuiance regressor does not match'\
               ' that of task response'


def test_gen_design():
    from brainiak.utils.utils import gen_design
    import numpy as np
    import os.path
    files = {'FSL1': 'example_stimtime_1_FSL.txt',
             'FSL2': 'example_stimtime_2_FSL.txt',
             'AFNI1': 'example_stimtime_1_AFNI.txt',
             'AFNI2': 'example_stimtime_2_AFNI.txt'}
    for key in files.keys():
        files[key] = os.path.join(os.path.dirname(__file__), files[key])
    design1 = gen_design(stimtime_files=files['FSL1'], scan_duration=[48, 20],
                         TR=2, style='FSL')
    assert design1.shape == (34, 1), 'Returned design matrix has wrong shape'
    assert design1[24] == 0, (
        "gen_design should generated design matrix for each run separately "
        "and concatenate them.")
    design2 = gen_design(stimtime_files=[files['FSL1'], files['FSL2']],
                         scan_duration=[48, 20], TR=2, style='FSL')
    assert design2.shape == (34, 2), 'Returned design matrix has wrong shape'
    design3 = gen_design(stimtime_files=files['FSL1'], scan_duration=68, TR=2,
                         style='FSL')
    assert design3[24] != 0, (
        'design matrix should be non-zero 8 seconds after an event onset.')
    design4 = gen_design(stimtime_files=[files['FSL2']],
                         scan_duration=[48, 20], TR=2, style='FSL')
    assert np.all(np.isclose(design1 * 0.5, design4)), (
        'gen_design does not treat missing values correctly')
    design5 = gen_design(stimtime_files=[files['FSL2']],
                         scan_duration=[48, 20], TR=1)
    assert (np.abs(design4 - design5[::2])).mean() < 0.1, (
        'design matrices sampled at different frequency do not match'
        ' at corresponding time points')
    design6 = gen_design(stimtime_files=[files['AFNI1']],
                         scan_duration=[48, 20], TR=2, style='AFNI')
    assert np.all(np.isclose(design1, design6)), (
        'design matrices generated from AFNI style and FSL style do not match')
    design7 = gen_design(stimtime_files=[files['AFNI2']],
                         scan_duration=[48], TR=2, style='AFNI')
    assert np.all(design7 == 0.0), (
        'A negative stimulus onset of AFNI style should result in an all-zero'
        + ' design matrix')


def test_center_mass_exp():
    from brainiak.utils.utils import center_mass_exp
    import numpy as np

    with pytest.raises(AssertionError) as excinfo:
        result = center_mass_exp([1, 2])
    assert ('interval must be a tuple'
            in str(excinfo.value))

    with pytest.raises(AssertionError) as excinfo:
        result = center_mass_exp((1, 2, 3))
    assert ('interval must be length two'
            in str(excinfo.value))

    with pytest.raises(AssertionError) as excinfo:
        result = center_mass_exp((-2, -1))
    assert ('interval_left must be non-negative'
            in str(excinfo.value))

    with pytest.raises(AssertionError) as excinfo:
        result = center_mass_exp((-2, 3))
    assert ('interval_left must be non-negative'
            in str(excinfo.value))

    with pytest.raises(AssertionError) as excinfo:
        result = center_mass_exp((3, 3))
    assert ('interval_right must be bigger than interval_left'
            in str(excinfo.value))

    with pytest.raises(AssertionError) as excinfo:
        result = center_mass_exp((1, 2), -1)
    assert ('scale must be positive'
            in str(excinfo.value))

    result = center_mass_exp((0, np.inf), 2.0)
    assert np.isclose(result, 2.0), 'center of mass '\
        'incorrect for the whole distribution'
    result = center_mass_exp((1.0, 1.0+2e-10))
    assert np.isclose(result, 1.0+1e-10), 'for a small '\
        'enough interval, the center of mass should be '\
        'close to its mid-point'


def test_p_from_null():
    import numpy as np
    from brainiak.utils.utils import p_from_null

    # Create random null and observed value in tail
    null = np.random.randn(10000)
    observed = np.ceil(np.percentile(null, 97.5) * 1000) / 1000

    # Check that we catch improper side
    with pytest.raises(ValueError):
        _ = p_from_null(observed, null, side='wrong')

    # Check two-tailed p-value for observed
    p_ts = p_from_null(observed, null)
    assert np.isclose(p_ts, 0.05, atol=1e-02)

    # Check two-tailed p-value for observed
    p_right = p_from_null(observed, null, side='right')
    assert np.isclose(p_right, 0.025, atol=1e-02)
    assert np.isclose(p_right, p_ts / 2, atol=1e-02)

    # Check two-tailed p-value for observed
    p_left = p_from_null(observed, null, side='left')
    assert np.isclose(p_left, 0.975, atol=1e-02)
    assert np.isclose(1 - p_left, p_right, atol=1e-02)
    assert np.isclose(1 - p_left, p_ts / 2, atol=1e-02)

    # Check 2-dimensional input (i.e., samples by voxels)
    null = np.random.randn(10000, 3)
    observed = np.ceil(np.percentile(null, 97.5, axis=0) * 1000) / 1000

    # Check two-tailed p-value for observed
    p_ts = p_from_null(observed, null, axis=0)
    assert np.allclose(p_ts, 0.05, atol=1e-02)

    # Check two-tailed p-value for observed
    p_right = p_from_null(observed, null, side='right', axis=0)
    assert np.allclose(p_right, 0.025, atol=1e-02)
    assert np.allclose(p_right, p_ts / 2, atol=1e-02)

    # Check two-tailed p-value for observed
    p_left = p_from_null(observed, null, side='left', axis=0)
    assert np.allclose(p_left, 0.975, atol=1e-02)
    assert np.allclose(1 - p_left, p_right, atol=1e-02)
    assert np.allclose(1 - p_left, p_ts / 2, atol=1e-02)

    # Check for exact test
    p_ts = p_from_null(observed, null, exact=True, axis=0)
    assert np.allclose(p_ts, 0.05, atol=1e-02)

    # Check two-tailed p-value for exact
    p_right = p_from_null(observed, null, side='right',
                          exact=True, axis=0)
    assert np.allclose(p_right, 0.025, atol=1e-02)
    assert np.allclose(p_right, p_ts / 2, atol=1e-02)

    # Check two-tailed p-value for exact
    p_left = p_from_null(observed, null, side='left',
                         exact=True, axis=0)
    assert np.allclose(p_left, 0.975, atol=1e-02)
    assert np.allclose(1 - p_left, p_right, atol=1e-02)
    assert np.allclose(1 - p_left, p_ts / 2, atol=1e-02)


def test_phase_randomize():
    import numpy as np
    from scipy.fftpack import fft
    from scipy.stats import pearsonr
    from brainiak.utils.utils import phase_randomize

    data = np.repeat(np.repeat(np.random.randn(60)[:, np.newaxis, np.newaxis],
                               30, axis=1),
                     20, axis=2)
    assert np.array_equal(data[..., 0], data[..., 1])

    # Phase-randomize data across subjects (same across voxels)
    shifted_data = phase_randomize(data, voxelwise=False, random_state=1)
    assert shifted_data.shape == data.shape
    assert not np.array_equal(shifted_data[..., 0], shifted_data[..., 1])
    assert not np.array_equal(shifted_data[..., 0], data[..., 0])

    # Check that uneven n_TRs doesn't explode
    _ = phase_randomize(data[:-1, ...])

    # Check that random_state returns same shifts
    shifted_data_ = phase_randomize(data, voxelwise=False, random_state=1)
    assert np.array_equal(shifted_data, shifted_data_)

    shifted_data_ = phase_randomize(data, voxelwise=False, random_state=2)
    assert not np.array_equal(shifted_data, shifted_data_)

    # Phase-randomize subjects and voxels
    shifted_data = phase_randomize(data, voxelwise=True, random_state=1)
    assert shifted_data.shape == data.shape
    assert not np.array_equal(shifted_data[..., 0], shifted_data[..., 1])
    assert not np.array_equal(shifted_data[..., 0], data[..., 0])
    assert not np.array_equal(shifted_data[:, 0, 0], shifted_data[:, 1, 0])

    # Try with 2-dimensional input
    shifted_data = phase_randomize(data[..., 0],
                                   voxelwise=True,
                                   random_state=1)
    assert shifted_data.ndim == 2
    assert not np.array_equal(shifted_data[:, 0], shifted_data[:, 1])

    # Create correlated noisy data
    corr_data = np.repeat(np.random.randn(60)[:, np.newaxis, np.newaxis],
                          2, axis=2) + np.random.randn(60, 1, 2)

    # Get correlation and frequency domain for data
    corr_r = pearsonr(corr_data[:, 0, 0],
                      corr_data[:, 0, 1])[0]
    corr_freq = fft(corr_data, axis=0)

    # Phase-randomize time series and get correlation/frequency
    shifted_data = phase_randomize(corr_data)
    shifted_r = pearsonr(shifted_data[:, 0, 0],
                         shifted_data[:, 0, 1])[0]
    shifted_freq = fft(shifted_data, axis=0)

    # Check that phase-randomization reduces correlation
    assert np.abs(shifted_r) < np.abs(corr_r)

    # Check that amplitude spectrum is preserved
    assert np.allclose(np.abs(shifted_freq), np.abs(corr_freq))


def test_check_timeseries_input():
    import numpy as np
    from itertools import combinations
    from brainiak.utils.utils import _check_timeseries_input

    # Set a fixed vector for comparison
    vector = np.random.randn(60)

    # List of subjects with one voxel/ROI
    list_1d = [vector for _ in np.arange(10)]
    (data_list_1d, n_TRs,
     n_voxels, n_subjects) = _check_timeseries_input(list_1d)
    assert n_TRs == 60
    assert n_voxels == 1
    assert n_subjects == 10

    # Array of subjects with one voxel/ROI
    array_2d = np.hstack([vector[:, np.newaxis]
                          for _ in np.arange(10)])
    (data_array_2d, n_TRs,
     n_voxels, n_subjects) = _check_timeseries_input(array_2d)
    assert n_TRs == 60
    assert n_voxels == 1
    assert n_subjects == 10

    # List of 2-dimensional arrays
    list_2d = [vector[:, np.newaxis] for _ in np.arange(10)]
    (data_list_2d, n_TRs,
     n_voxels, n_subjects) = _check_timeseries_input(list_2d)
    assert n_TRs == 60
    assert n_voxels == 1
    assert n_subjects == 10

    # Check if lists have mismatching size
    list_bad = [list_2d[0][:-1, :]] + list_2d[1:]
    with pytest.raises(ValueError):
        (data_list_bad, _, _, _) = _check_timeseries_input(list_bad)

    # List of 3-dimensional arrays
    list_3d = [vector[:, np.newaxis, np.newaxis]
               for _ in np.arange(10)]
    (data_list_3d, n_TRs,
     n_voxels, n_subjects) = _check_timeseries_input(list_3d)
    assert n_TRs == 60
    assert n_voxels == 1
    assert n_subjects == 10

    # 3-dimensional array
    array_3d = np.dstack([vector[:, np.newaxis]
                          for _ in np.arange(10)])
    (data_array_3d, n_TRs,
     n_voxels, n_subjects) = _check_timeseries_input(array_3d)
    assert n_TRs == 60
    assert n_voxels == 1
    assert n_subjects == 10

    # Check that 4-dimensional input array throws error
    array_4d = array_3d[..., np.newaxis]
    with pytest.raises(ValueError):
        (data_array_4d, _, _, _) = _check_timeseries_input(array_4d)

    # Check they're the same
    for pair in combinations([data_list_1d, data_array_2d,
                              data_list_2d, data_list_3d,
                              data_array_3d], 2):
        assert np.array_equal(pair[0], pair[1])

    # List of multivoxel arrays
    matrix = np.random.randn(60, 30)
    list_mv = [matrix
               for _ in np.arange(10)]
    (data_list_mv, n_TRs,
     n_voxels, n_subjects) = _check_timeseries_input(list_mv)
    assert n_TRs == 60
    assert n_voxels == 30
    assert n_subjects == 10

    # 3-dimensional array with multiple voxels
    array_mv = np.dstack([matrix for _ in np.arange(10)])
    (data_array_mv, n_TRs,
     n_voxels, n_subjects) = _check_timeseries_input(array_mv)
    assert n_TRs == 60
    assert n_voxels == 30
    assert n_subjects == 10

    assert np.array_equal(data_list_mv, data_array_mv)


def test_array_correlation():
    import numpy as np
    from brainiak.utils.utils import array_correlation
    from scipy.stats import pearsonr

    # Minimal array datasets
    n_TRs = 30
    n_voxels = 2
    x, y = (np.random.randn(n_TRs, n_voxels),
            np.random.randn(n_TRs, n_voxels))

    # Perform the correlation
    r = array_correlation(x, y)

    # Check there are the right number of voxels in the output
    assert r.shape == (n_voxels,)

    # Check that this (roughly) matches corrcoef
    assert np.allclose(r, np.corrcoef(x.T, y.T)[[0, 1], [2, 3]])

    # Check that this (roughly) matches pearsonr
    assert np.allclose(r, np.array([pearsonr(x[:, 0], y[:, 0])[0],
                                    pearsonr(x[:, 1], y[:, 1])[0]]))

    # Try axis argument
    assert np.allclose(array_correlation(x, y, axis=0),
                       array_correlation(x.T, y.T, axis=1))

    # Trigger shape mismatch error
    with pytest.raises(ValueError):
        array_correlation(x, y[:, 0])

    with pytest.raises(ValueError):
        array_correlation(x, y[:-1])

    # Feed in lists
    _ = array_correlation(x.tolist(), y)
    _ = array_correlation(x, y.tolist())
    _ = array_correlation(x.tolist(), y.tolist())

    # Check 1D array input
    x, y = (np.random.randn(n_TRs),
            np.random.randn(n_TRs))

    assert type(array_correlation(x, y)) == np.float64
    assert np.isclose(array_correlation(x, y),
                      pearsonr(x, y)[0])

    # 1D list inputs
    _ = array_correlation(x.tolist(), y)
    _ = array_correlation(x, y.tolist())
    _ = array_correlation(x.tolist(), y.tolist())

    # Check integer inputs
    x, y = (np.random.randint(0, 9, (n_TRs, n_voxels)),
            np.random.randint(0, 9, (n_TRs, n_voxels)))
    _ = array_correlation(x, y)
