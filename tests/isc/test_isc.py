import numpy as np
import logging
from brainiak.isc import (isc, isfc, bootstrap_isc, permutation_isc,
                          timeshift_isc, phaseshift_isc)
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


# Create simple simulated data with high intersubject correlation
def simulated_timeseries(n_subjects, n_TRs, n_voxels=30,
                         noise=1, data_type='array',
                         random_state=None):
    prng = np.random.RandomState(random_state)
    if n_voxels:
        signal = prng.randn(n_TRs, n_voxels)
        prng = np.random.RandomState(prng.randint(0, 2**32 - 1))
        data = [signal + prng.randn(n_TRs, n_voxels) * noise
                for subject in np.arange(n_subjects)]
    elif not n_voxels:
        signal = prng.randn(n_TRs)
        prng = np.random.RandomState(prng.randint(0, 2**32 - 1))
        data = [signal + prng.randn(n_TRs) * noise
                for subject in np.arange(n_subjects)]
    if data_type == 'array':
        if n_voxels:
            data = np.dstack(data)
        elif not n_voxels:
            data = np.column_stack(data)
    return data


# Create 3 voxel simulated data with correlated time series
def correlated_timeseries(n_subjects, n_TRs, noise=0,
                          random_state=None):
    prng = np.random.RandomState(random_state)
    signal = prng.randn(n_TRs)
    correlated = True
    while correlated:
        uncorrelated = np.random.randn(n_TRs,
                                       n_subjects)[:, np.newaxis, :]
        unc_max = np.amax(squareform(np.corrcoef(
            uncorrelated[:, 0, :].T), checks=False))
        unc_mean = np.mean(squareform(np.corrcoef(
            uncorrelated[:, 0, :].T), checks=False))
        if unc_max < .3 and np.abs(unc_mean) < .001:
            correlated = False
    data = np.repeat(np.column_stack((signal, signal))[..., np.newaxis],
                     20, axis=2)
    data = np.concatenate((data, uncorrelated), axis=1)
    data = data + np.random.randn(n_TRs, 3, n_subjects) * noise
    return data


# Compute ISCs using different input types
# List of subjects with one voxel/ROI
def test_isc_input():

    # Set parameters for toy time series data
    n_subjects = 20
    n_TRs = 60
    n_voxels = 30
    random_state = 42

    logger.info("Testing ISC inputs")

    data = simulated_timeseries(n_subjects, n_TRs,
                                n_voxels=None, data_type='list',
                                random_state=random_state)
    iscs_list = isc(data, pairwise=False, summary_statistic=None)

    # Array of subjects with one voxel/ROI
    data = simulated_timeseries(n_subjects, n_TRs,
                                n_voxels=None, data_type='array',
                                random_state=random_state)
    iscs_array = isc(data, pairwise=False, summary_statistic=None)

    # Check they're the same
    assert np.array_equal(iscs_list, iscs_array)

    # List of subjects with multiple voxels/ROIs
    data = simulated_timeseries(n_subjects, n_TRs,
                                n_voxels=n_voxels, data_type='list',
                                random_state=random_state)
    iscs_list = isc(data, pairwise=False, summary_statistic=None)

    # Array of subjects with multiple voxels/ROIs
    data = simulated_timeseries(n_subjects, n_TRs,
                                n_voxels=n_voxels, data_type='array',
                                random_state=random_state)
    iscs_array = isc(data, pairwise=False, summary_statistic=None)

    # Check they're the same
    assert np.array_equal(iscs_list, iscs_array)

    logger.info("Finished testing ISC inputs")


# Check pairwise and leave-one-out, and summary statistics for ISC
def test_isc_options():

    # Set parameters for toy time series data
    n_subjects = 20
    n_TRs = 60
    n_voxels = 30
    random_state = 42

    logger.info("Testing ISC options")

    data = simulated_timeseries(n_subjects, n_TRs,
                                n_voxels=n_voxels, data_type='array',
                                random_state=random_state)

    iscs_loo = isc(data, pairwise=False, summary_statistic=None)
    assert iscs_loo.shape == (n_subjects, n_voxels)

    iscs_pw = isc(data, pairwise=True, summary_statistic=None)
    assert iscs_pw.shape == (n_subjects*(n_subjects-1)/2, n_voxels)

    # Check summary statistics
    isc_mean = isc(data, pairwise=False, summary_statistic='mean')
    assert isc_mean.shape == (1, n_voxels)

    isc_median = isc(data, pairwise=False, summary_statistic='median')
    assert isc_median.shape == (1, n_voxels)

    try:
        isc(data, pairwise=False, summary_statistic='min')
    except ValueError:
        logger.info("Correctly caught unexpected summary statistic")

    logger.info("Finished testing ISC options")


# Make sure ISC recovers correlations of 1 and less than 1
def test_isc_output():

    logger.info("Testing ISC outputs")

    data = correlated_timeseries(20, 60, noise=0,
                                 random_state=42)
    iscs = isc(data, pairwise=False)
    assert np.allclose(iscs[:, :2], 1., rtol=1e-05)
    assert np.all(iscs[:, -1] < 1.)

    iscs = isc(data, pairwise=True)
    assert np.allclose(iscs[:, :2], 1., rtol=1e-05)
    assert np.all(iscs[:, -1] < 1.)

    logger.info("Finished testing ISC outputs")


# Check for proper handling of NaNs in ISC
def test_isc_nans():

    # Set parameters for toy time series data
    n_subjects = 20
    n_TRs = 60
    n_voxels = 30
    random_state = 42

    logger.info("Testing ISC options")

    data = simulated_timeseries(n_subjects, n_TRs,
                                n_voxels=n_voxels, data_type='array',
                                random_state=random_state)

    # Inject NaNs into data
    data[0, 0, 0] = np.nan

    # Don't tolerate NaNs, should lose zeroeth voxel
    iscs_loo = isc(data, pairwise=False, tolerate_nans=False)
    assert np.sum(np.isnan(iscs_loo)) == n_subjects

    # Tolerate all NaNs, only subject with NaNs yields NaN
    iscs_loo = isc(data, pairwise=False, tolerate_nans=True)
    assert np.sum(np.isnan(iscs_loo)) == 1

    # Pairwise approach shouldn't care
    iscs_pw_T = isc(data, pairwise=True, tolerate_nans=True)
    iscs_pw_F = isc(data, pairwise=True, tolerate_nans=False)
    assert np.allclose(iscs_pw_T, iscs_pw_F, equal_nan=True)

    assert (np.sum(np.isnan(iscs_pw_T)) ==
            np.sum(np.isnan(iscs_pw_F)) ==
            n_subjects - 1)

    # Set proportion of nans to reject (70% and 90% non-NaN)
    data[0, 0, :] = np.nan
    data[0, 1, :n_subjects - int(n_subjects * .7)] = np.nan
    data[0, 2, :n_subjects - int(n_subjects * .9)] = np.nan

    iscs_loo_T = isc(data, pairwise=False, tolerate_nans=True)
    iscs_loo_F = isc(data, pairwise=False, tolerate_nans=False)
    iscs_loo_95 = isc(data, pairwise=False, tolerate_nans=.95)
    iscs_loo_90 = isc(data, pairwise=False, tolerate_nans=.90)
    iscs_loo_80 = isc(data, pairwise=False, tolerate_nans=.8)
    iscs_loo_70 = isc(data, pairwise=False, tolerate_nans=.7)
    iscs_loo_60 = isc(data, pairwise=False, tolerate_nans=.6)

    assert (np.sum(np.isnan(iscs_loo_F)) ==
            np.sum(np.isnan(iscs_loo_95)) == 60)
    assert (np.sum(np.isnan(iscs_loo_80)) ==
            np.sum(np.isnan(iscs_loo_90)) == 42)
    assert (np.sum(np.isnan(iscs_loo_T)) ==
            np.sum(np.isnan(iscs_loo_60)) ==
            np.sum(np.isnan(iscs_loo_70)) == 28)
    assert np.array_equal(np.sum(np.isnan(iscs_loo_F), axis=0),
                          np.sum(np.isnan(iscs_loo_95), axis=0))
    assert np.array_equal(np.sum(np.isnan(iscs_loo_80), axis=0),
                          np.sum(np.isnan(iscs_loo_90), axis=0))
    assert np.all((np.array_equal(
                        np.sum(np.isnan(iscs_loo_T), axis=0),
                        np.sum(np.isnan(iscs_loo_60), axis=0)),
                   np.array_equal(
                        np.sum(np.isnan(iscs_loo_T), axis=0),
                        np.sum(np.isnan(iscs_loo_70), axis=0)),
                   np.array_equal(
                        np.sum(np.isnan(iscs_loo_60), axis=0),
                        np.sum(np.isnan(iscs_loo_70), axis=0))))

    data = simulated_timeseries(n_subjects, n_TRs,
                                n_voxels=n_voxels, data_type='array',
                                random_state=random_state)

    # Inject NaNs into data
    data[0, 0, 0] = np.nan

    # Make sure voxel with NaNs across all subjects is always removed
    data[0, 0, :] = np.nan
    iscs_loo_T = isc(data, pairwise=False, tolerate_nans=True)
    iscs_loo_F = isc(data, pairwise=False, tolerate_nans=False)
    assert np.allclose(iscs_loo_T, iscs_loo_F, equal_nan=True)
    assert (np.sum(np.isnan(iscs_loo_T)) ==
            np.sum(np.isnan(iscs_loo_T)) ==
            n_subjects)

    iscs_pw_T = isc(data, pairwise=True, tolerate_nans=True)
    iscs_pw_F = isc(data, pairwise=True, tolerate_nans=False)
    assert np.allclose(iscs_pw_T, iscs_pw_F, equal_nan=True)

    assert (np.sum(np.isnan(iscs_pw_T)) ==
            np.sum(np.isnan(iscs_pw_T)) ==
            n_subjects * (n_subjects - 1) / 2)


# Test one-sample bootstrap test
def test_bootstrap_isc():

    # Set parameters for toy time series data
    n_subjects = 20
    n_TRs = 60
    n_voxels = 30
    random_state = 42
    n_bootstraps = 10

    logger.info("Testing bootstrap hypothesis test")

    data = simulated_timeseries(n_subjects, n_TRs,
                                n_voxels=n_voxels, data_type='array',
                                random_state=random_state)

    iscs = isc(data, pairwise=False, summary_statistic=None)
    observed, ci, p, distribution = bootstrap_isc(iscs, pairwise=False,
                                                  summary_statistic='median',
                                                  n_bootstraps=n_bootstraps,
                                                  ci_percentile=95)
    assert distribution.shape == (n_bootstraps, n_voxels)

    # Test one-sample bootstrap test with pairwise approach
    n_bootstraps = 10

    iscs = isc(data, pairwise=True, summary_statistic=None)
    observed, ci, p, distribution = bootstrap_isc(iscs, pairwise=True,
                                                  summary_statistic='median',
                                                  n_bootstraps=n_bootstraps,
                                                  ci_percentile=95)
    assert distribution.shape == (n_bootstraps, n_voxels)

    # Check random seeds
    iscs = isc(data, pairwise=False, summary_statistic=None)
    distributions = []
    for random_state in [42, 42, None]:
        observed, ci, p, distribution = bootstrap_isc(
                                                iscs, pairwise=False,
                                                summary_statistic='median',
                                                n_bootstraps=n_bootstraps,
                                                ci_percentile=95,
                                                random_state=random_state)
        distributions.append(distribution)
    assert np.array_equal(distributions[0], distributions[1])
    assert not np.array_equal(distributions[1], distributions[2])

    # Check output p-values
    data = correlated_timeseries(20, 60, noise=.5,
                                 random_state=42)
    iscs = isc(data, pairwise=False)
    observed, ci, p, distribution = bootstrap_isc(iscs, pairwise=False)
    assert np.all(iscs[:, :2] > .5)
    assert np.all(iscs[:, -1] < .5)
    assert p[0, 0] < .05 and p[0, 1] < .05
    assert p[0, 2] > .01

    iscs = isc(data, pairwise=True)
    observed, ci, p, distribution = bootstrap_isc(iscs, pairwise=True)
    assert np.all(iscs[:, :2] > .5)
    assert np.all(iscs[:, -1] < .5)
    assert p[0, 0] < .05 and p[0, 1] < .05
    assert p[0, 2] > .01

    # Check that ISC computation and bootstrap observed are same
    iscs = isc(data, pairwise=False)
    observed, ci, p, distribution = bootstrap_isc(iscs, pairwise=False,
                                                  summary_statistic='median')
    assert np.array_equal(observed, isc(data, pairwise=False,
                                        summary_statistic='median'))

    # Check that ISC computation and bootstrap observed are same
    iscs = isc(data, pairwise=True)
    observed, ci, p, distribution = bootstrap_isc(iscs, pairwise=True,
                                                  summary_statistic='median')
    assert np.array_equal(observed, isc(data, pairwise=True,
                                        summary_statistic='median'))

    logger.info("Finished testing bootstrap hypothesis test")


# Test permutation test with group assignments
def test_permutation_isc():

    # Set parameters for toy time series data
    n_subjects = 20
    n_TRs = 60
    n_voxels = 30
    random_state = 42
    group_assignment = [1] * 10 + [2] * 10

    logger.info("Testing permutation test")

    # Create dataset with two groups in pairwise approach
    data = np.dstack((simulated_timeseries(10, n_TRs, n_voxels=n_voxels,
                                           noise=1, data_type='array',
                                           random_state=3),
                      simulated_timeseries(10, n_TRs, n_voxels=n_voxels,
                                           noise=5, data_type='array',
                                           random_state=4)))
    iscs = isc(data, pairwise=True, summary_statistic=None)

    observed, p, distribution = permutation_isc(
                                            iscs,
                                            group_assignment=group_assignment,
                                            pairwise=True,
                                            summary_statistic='mean',
                                            n_permutations=200)

    # Create data with two groups in leave-one-out approach
    data_1 = simulated_timeseries(10, n_TRs, n_voxels=n_voxels,
                                  noise=1, data_type='array',
                                  random_state=3)
    data_2 = simulated_timeseries(10, n_TRs, n_voxels=n_voxels,
                                  noise=10, data_type='array',
                                  random_state=4)
    iscs = np.vstack((isc(data_1, pairwise=False, summary_statistic=None),
                      isc(data_2, pairwise=False, summary_statistic=None)))

    observed, p, distribution = permutation_isc(
                                            iscs,
                                            group_assignment=group_assignment,
                                            pairwise=False,
                                            summary_statistic='mean',
                                            n_permutations=200)

    # One-sample leave-one-out permutation test
    data = simulated_timeseries(n_subjects, n_TRs,
                                n_voxels=n_voxels, data_type='array',
                                random_state=random_state)
    iscs = isc(data, pairwise=False, summary_statistic=None)

    observed, p, distribution = permutation_isc(iscs,
                                                pairwise=False,
                                                summary_statistic='median',
                                                n_permutations=200)

    # One-sample pairwise permutation test
    iscs = isc(data, pairwise=True, summary_statistic=None)

    observed, p, distribution = permutation_isc(iscs,
                                                pairwise=True,
                                                summary_statistic='median',
                                                n_permutations=200)

    # Small one-sample pairwise exact test
    data = simulated_timeseries(12, n_TRs,
                                n_voxels=n_voxels, data_type='array',
                                random_state=random_state)
    iscs = isc(data, pairwise=False, summary_statistic=None)

    observed, p, distribution = permutation_isc(iscs, pairwise=False,
                                                summary_statistic='median',
                                                n_permutations=10000)

    # Small two-sample pairwise exact test (and unequal groups)
    data = np.dstack((simulated_timeseries(3, n_TRs, n_voxels=n_voxels,
                                           noise=1, data_type='array',
                                           random_state=3),
                      simulated_timeseries(4, n_TRs, n_voxels=n_voxels,
                                           noise=50, data_type='array',
                                           random_state=4)))
    iscs = isc(data, pairwise=True, summary_statistic=None)
    group_assignment = [1, 1, 1, 2, 2, 2, 2]

    observed, p, distribution = permutation_isc(
                                            iscs,
                                            group_assignment=group_assignment,
                                            pairwise=True,
                                            summary_statistic='mean',
                                            n_permutations=10000)

    # Small two-sample leave-one-out exact test (and unequal groups)
    data_1 = simulated_timeseries(3, n_TRs, n_voxels=n_voxels,
                                  noise=1, data_type='array',
                                  random_state=3)
    data_2 = simulated_timeseries(4, n_TRs, n_voxels=n_voxels,
                                  noise=50, data_type='array',
                                  random_state=4)
    iscs = np.vstack((isc(data_1, pairwise=False, summary_statistic=None),
                      isc(data_2, pairwise=False, summary_statistic=None)))
    group_assignment = [1, 1, 1, 2, 2, 2, 2]

    observed, p, distribution = permutation_isc(
                                            iscs,
                                            group_assignment=group_assignment,
                                            pairwise=False,
                                            summary_statistic='mean',
                                            n_permutations=10000)

    # Check output p-values
    data = correlated_timeseries(20, 60, noise=.5,
                                 random_state=42)
    iscs = isc(data, pairwise=False)
    observed, p, distribution = permutation_isc(iscs, pairwise=False)
    assert np.all(iscs[:, :2] > .5)
    assert np.all(iscs[:, -1] < .5)
    assert p[0, 0] < .05 and p[0, 1] < .05
    assert p[0, 2] > .01

    iscs = isc(data, pairwise=True)
    observed, p, distribution = permutation_isc(iscs, pairwise=True)
    assert np.all(iscs[:, :2] > .5)
    assert np.all(iscs[:, -1] < .5)
    assert p[0, 0] < .05 and p[0, 1] < .05
    assert p[0, 2] > .01

    # Check that ISC computation and permutation observed are same
    iscs = isc(data, pairwise=False)
    observed, p, distribution = permutation_isc(iscs, pairwise=False,
                                                summary_statistic='median')
    assert np.array_equal(observed, isc(data, pairwise=False,
                                        summary_statistic='median'))

    # Check that ISC computation and permuation observed are same
    iscs = isc(data, pairwise=True)
    observed, p, distribution = permutation_isc(iscs, pairwise=True,
                                                summary_statistic='mean')
    assert np.array_equal(observed, isc(data, pairwise=True,
                                        summary_statistic='mean'))

    logger.info("Finished testing permutaton test")


def test_timeshift_isc():

    # Set parameters for toy time series data
    n_subjects = 20
    n_TRs = 60
    n_voxels = 30

    logger.info("Testing circular time-shift")

    # Circular time-shift on one sample, leave-one-out
    data = simulated_timeseries(n_subjects, n_TRs,
                                n_voxels=n_voxels, data_type='array')
    observed, p, distribution = timeshift_isc(data, pairwise=False,
                                              summary_statistic='median',
                                              n_shifts=200)

    # Circular time-shift on one sample, pairwise
    data = simulated_timeseries(n_subjects, n_TRs,
                                n_voxels=n_voxels, data_type='array')
    observed, p, distribution = timeshift_isc(data, pairwise=True,
                                              summary_statistic='median',
                                              n_shifts=200)

    # Circular time-shift on one sample, leave-one-out
    data = simulated_timeseries(n_subjects, n_TRs,
                                n_voxels=n_voxels, data_type='array')
    observed, p, distribution = timeshift_isc(data, pairwise=False,
                                              summary_statistic='mean',
                                              n_shifts=200)
    # Check output p-values
    data = correlated_timeseries(20, 60, noise=.5,
                                 random_state=42)
    iscs = isc(data, pairwise=False)
    observed, p, distribution = timeshift_isc(data, pairwise=False)
    assert np.all(iscs[:, :2] > .5)
    assert np.all(iscs[:, -1] < .5)
    assert p[0, 0] < .05 and p[0, 1] < .05
    assert p[0, 2] > .01

    iscs = isc(data, pairwise=True)
    observed, p, distribution = timeshift_isc(data, pairwise=True)
    assert np.all(iscs[:, :2] > .5)
    assert np.all(iscs[:, -1] < .5)
    assert p[0, 0] < .05 and p[0, 1] < .05
    assert p[0, 2] > .01

    # Check that ISC computation and permutation observed are same
    iscs = isc(data, pairwise=False)
    observed, p, distribution = timeshift_isc(data, pairwise=False,
                                              summary_statistic='median')
    assert np.array_equal(observed, isc(data, pairwise=False,
                                        summary_statistic='median'))

    # Check that ISC computation and permuation observed are same
    iscs = isc(data, pairwise=True)
    observed, p, distribution = timeshift_isc(data, pairwise=True,
                                              summary_statistic='mean')
    assert np.array_equal(observed, isc(data, pairwise=True,
                                        summary_statistic='mean'))

    logger.info("Finished testing circular time-shift")


# Phase randomization test
def test_phaseshift_isc():

    # Set parameters for toy time series data
    n_subjects = 20
    n_TRs = 60
    n_voxels = 30

    logger.info("Testing phase randomization")

    data = simulated_timeseries(n_subjects, n_TRs,
                                n_voxels=n_voxels, data_type='array')
    observed, p, distribution = phaseshift_isc(data, pairwise=True,
                                               summary_statistic='median',
                                               n_shifts=200)

    # Phase randomization one-sample test, leave-one-out
    data = simulated_timeseries(n_subjects, n_TRs,
                                n_voxels=n_voxels, data_type='array')
    observed, p, distribution = phaseshift_isc(data, pairwise=False,
                                               summary_statistic='mean',
                                               n_shifts=200)

    # Check output p-values
    data = correlated_timeseries(20, 60, noise=.5,
                                 random_state=42)
    iscs = isc(data, pairwise=False)
    observed, p, distribution = phaseshift_isc(data, pairwise=False)
    assert np.all(iscs[:, :2] > .5)
    assert np.all(iscs[:, -1] < .5)
    assert p[0, 0] < .05 and p[0, 1] < .05
    assert p[0, 2] > .01

    iscs = isc(data, pairwise=True)
    observed, p, distribution = phaseshift_isc(data, pairwise=True)
    assert np.all(iscs[:, :2] > .5)
    assert np.all(iscs[:, -1] < .5)
    assert p[0, 0] < .05 and p[0, 1] < .05
    assert p[0, 2] > .01

    # Check that ISC computation and permutation observed are same
    iscs = isc(data, pairwise=False)
    observed, p, distribution = phaseshift_isc(data, pairwise=False,
                                               summary_statistic='median')
    assert np.array_equal(observed, isc(data, pairwise=False,
                                        summary_statistic='median'))

    # Check that ISC computation and permuation observed are same
    iscs = isc(data, pairwise=True)
    observed, p, distribution = phaseshift_isc(data, pairwise=True,
                                               summary_statistic='mean')
    assert np.array_equal(observed, isc(data, pairwise=True,
                                        summary_statistic='mean'))

    logger.info("Finished testing phase randomization")


# Test ISFC
def test_isfc_options():

    # Set parameters for toy time series data
    n_subjects = 20
    n_TRs = 60
    n_voxels = 30

    logger.info("Testing ISFC options")

    data = simulated_timeseries(n_subjects, n_TRs,
                                n_voxels=n_voxels, data_type='array')
    isfcs = isfc(data, pairwise=False, summary_statistic=None)

    # Just two subjects
    isfcs = isfc(data[..., :2], pairwise=False, summary_statistic=None)

    # ISFC with pairwise approach
    isfcs = isfc(data, pairwise=True, summary_statistic=None)

    # ISFC with summary statistics
    isfcs = isfc(data, pairwise=True, summary_statistic='mean')
    isfcs = isfc(data, pairwise=True, summary_statistic='median')

    # Check output p-values
    data = correlated_timeseries(20, 60, noise=.5,
                                 random_state=42)
    isfcs = isfc(data, pairwise=False)
    assert np.all(isfcs[0, 1, :] > .5) and np.all(isfcs[1, 0, :] > .5)
    assert np.all(isfcs[:2, 2, :] < .5) and np.all(isfcs[2, :2, :] < .5)

    isfcs = isfc(data, pairwise=True)
    assert np.all(isfcs[0, 1, :] > .5) and np.all(isfcs[1, 0, :] > .5)
    assert np.all(isfcs[:2, 2, :] < .5) and np.all(isfcs[2, :2, :] < .5)

    # Check that ISC and ISFC diagonal are identical
    iscs = isc(data, pairwise=False)
    isfcs = isfc(data, pairwise=False)
    for s in np.arange(len(iscs)):
        assert np.allclose(isfcs[..., s].diagonal(), iscs[s, :], rtol=1e-03)

    # Check that ISC and ISFC diagonal are identical
    iscs = isc(data, pairwise=True)
    isfcs = isfc(data, pairwise=True)
    for s in np.arange(len(iscs)):
        assert np.allclose(isfcs[..., s].diagonal(), iscs[s, :], rtol=1e-03)

    logger.info("Finished testing ISFC options")


# Check for proper handling of NaNs in ISFC
def test_isfc_nans():

    # Set parameters for toy time series data
    n_subjects = 20
    n_TRs = 60
    n_voxels = 30
    random_state = 42

    logger.info("Testing ISC options")

    data = simulated_timeseries(n_subjects, n_TRs,
                                n_voxels=n_voxels, data_type='array',
                                random_state=random_state)

    # Inject NaNs into data
    data[0, 0, 0] = np.nan

    # Don't tolerate NaNs, should lose zeroeth voxel
    isfcs_loo = isfc(data, pairwise=False, tolerate_nans=False)
    assert np.sum(isfcs_loo == 0) == n_subjects

    # Tolerate all NaNs, only subject with NaNs yields NaN
    isfcs_loo = isfc(data, pairwise=False, tolerate_nans=True)
    assert np.sum(np.isnan(isfcs_loo)) == 0

    # Pairwise approach shouldn't care
    isfcs_pw_T = isfc(data, pairwise=True, tolerate_nans=True)
    isfcs_pw_F = isfc(data, pairwise=True, tolerate_nans=False)
    assert np.allclose(isfcs_pw_T, isfcs_pw_F, equal_nan=True)

    assert (np.sum(np.isnan(isfcs_pw_T)) ==
            np.sum(np.isnan(isfcs_pw_F)) == 0)

    # Set proportion of nans to reject (70% and 90% non-NaN)
    data[0, 0, :] = np.nan
    data[0, 1, :n_subjects - int(n_subjects * .7)] = np.nan
    data[0, 2, :n_subjects - int(n_subjects * .9)] = np.nan

    nans = np.sum(np.any(np.isnan(data), axis=0), axis=1)

    isfcs_loo_T = isfc(data, pairwise=False, tolerate_nans=True)
    isfcs_loo_F = isfc(data, pairwise=False, tolerate_nans=False)
    isfcs_loo_95 = isfc(data, pairwise=False, tolerate_nans=.95)
    isfcs_loo_90 = isfc(data, pairwise=False, tolerate_nans=.90)
    isfcs_loo_80 = isfc(data, pairwise=False, tolerate_nans=.8)
    isfcs_loo_70 = isfc(data, pairwise=False, tolerate_nans=.7)
    isfcs_loo_60 = isfc(data, pairwise=False, tolerate_nans=.6)

    assert (np.sum(np.isnan(iscs_loo_F)) ==
            np.sum(np.isnan(iscs_loo_95)) == 60)
    assert (np.sum(np.isnan(iscs_loo_80)) ==
            np.sum(np.isnan(iscs_loo_90)) == 42)
    assert (np.sum(np.isnan(iscs_loo_T)) ==
            np.sum(np.isnan(iscs_loo_60)) ==
            np.sum(np.isnan(iscs_loo_70)) == 28)
    assert np.array_equal(np.sum(np.isnan(iscs_loo_F), axis=0),
                          np.sum(np.isnan(iscs_loo_95), axis=0))
    assert np.array_equal(np.sum(np.isnan(iscs_loo_80), axis=0),
                          np.sum(np.isnan(iscs_loo_90), axis=0))
    assert np.all((np.array_equal(
                        np.sum(np.isnan(iscs_loo_T), axis=0),
                        np.sum(np.isnan(iscs_loo_60), axis=0)),
                   np.array_equal(
                        np.sum(np.isnan(iscs_loo_T), axis=0),
                        np.sum(np.isnan(iscs_loo_70), axis=0)),
                   np.array_equal(
                        np.sum(np.isnan(iscs_loo_60), axis=0),
                        np.sum(np.isnan(iscs_loo_70), axis=0))))

    # Make sure voxel with NaNs across all subjects is always removed
    data[0, 0, :] = np.nan
    iscs_loo_T = isc(data, pairwise=False, tolerate_nans=True)
    iscs_loo_F = isc(data, pairwise=False, tolerate_nans=False)
    assert np.allclose(iscs_loo_T, iscs_loo_F, equal_nan=True)
    assert (np.sum(np.isnan(iscs_loo_T)) ==
            np.sum(np.isnan(iscs_loo_T)) ==
            n_subjects)

    iscs_pw_T = isc(data, pairwise=True, tolerate_nans=True)
    iscs_pw_F = isc(data, pairwise=True, tolerate_nans=False)
    assert np.allclose(iscs_pw_T, iscs_pw_F, equal_nan=True)

    assert (np.sum(np.isnan(iscs_pw_T)) ==
            np.sum(np.isnan(iscs_pw_T)) ==
            n_subjects * (n_subjects - 1) / 2)


if __name__ == '__main__':
    test_isc_input()
    test_isc_options()
    test_isc_output()
    test_isc_nans()
    test_bootstrap_isc()
    test_permutation_isc()
    test_timeshift_isc()
    test_phaseshift_isc()
    test_isfc_options()
    logger.info("Finished all ISC tests")
