import brainiak.isfc
from brainiak import image, io
import numpy as np
<<<<<<< HEAD
from brainiak.isfc import (isc, isfc, bootstrap_isc, permutation_ics,
                           timeshift_isc, phaseshift_isc)
=======
import os
>>>>>>> parent of 94d10cc... Fixed location of test_isfc.py and added module imports


def test_ISC():
    # Create dataset in which one voxel is highly correlated across subjects
    # and the other is not
    D = np.zeros((2, 5, 3))
    D[:, :, 0] = \
        [[-0.36225433, -0.43482456,  0.26723158,  0.16461712, -0.37991465],
         [-0.62305959, -0.46660116, -0.50037994,  1.81083754,  0.23499509]]
    D[:, :, 1] = \
        [[-0.30484153, -0.49486988,  0.10966625, -0.19568572, -0.20535156],
         [1.68267639, -0.78433298, -0.35875085, -0.6121344,  0.28603493]]
    D[:, :, 2] = \
        [[-0.36593192, -0.50914734,  0.21397317,  0.30276589, -0.42637472],
         [0.04127293, -0.67598379, -0.51549055, -0.64196342,  1.60686666]]

    (ISC, p) = brainiak.isfc.isc(D, return_p=True, num_perm=100,
                                 two_sided=True, random_state=0)

    assert np.isclose(ISC, [0.8909243, 0.0267954]).all(), \
        "Calculated ISC does not match ground truth"

    assert np.isclose(p, [0.02, 1]).all(), \
        "Calculated p values do not match ground truth"

    (ISC, p) = brainiak.isfc.isc(D, return_p=True, num_perm=100,
                                 two_sided=True, collapse_subj=False,
                                 random_state=0)
    true_ISC = [[0.98221543, 0.76747914, 0.92307833],
                [-0.26377767, 0.01490501, 0.32925896]]
    true_p = [[0, 0.6, 0.08], [1, 1, 1]]

    assert np.isclose(ISC, true_ISC).all(), \
        "Calculated ISC (non collapse) does not match ground truth"

    assert np.isclose(p, true_p).all(), \
        "Calculated p values (non collapse) do not match ground truth"


def test_ISFC():
    curr_dir = os.path.dirname(__file__)

    mask_fname = os.path.join(curr_dir, 'mask.nii.gz')
    mask = io.load_boolean_mask(mask_fname)
    fnames = [os.path.join(curr_dir, 'subj1.nii.gz'),
              os.path.join(curr_dir, 'subj2.nii.gz')]
    masked_images = image.mask_images(io.load_images(fnames), mask)

    D = image.MaskedMultiSubjectData.from_masked_images(masked_images,
                                                        len(fnames))

    assert D.shape == (4, 5, 2), "Loaded data has incorrect shape"

    (ISFC, p) = brainiak.isfc.isfc(D, return_p=True, num_perm=100,
                                   two_sided=True, random_state=0)

    ground_truth = \
        [[1, 1, 0, -1],
         [1, 1, 0, -1],
         [0, 0, 1,  0],
         [-1, -1, 0, 1]]

    ground_truth_p = 1 - np.abs(ground_truth)

    assert np.isclose(ISFC, ground_truth).all(), \
        "Calculated ISFC does not match ground truth"

    assert np.isclose(p, ground_truth_p).all(), \
        "Calculated p values do not match ground truth"

    (ISFC, p) = brainiak.isfc.isfc(D, return_p=True, num_perm=100,
                                   two_sided=True, collapse_subj=False,
                                   random_state=0)
    array1 = np.array([[1, 1], [1, 1], [0, 0], [-1, -1]])
    array2 = -array1
    array3 = np.absolute(array1)
    array4 = 1 - array3

    true_ISFC = np.array([array1, array1, array4, array2])
    true_p = np.array([array4, array4, array3, array4])

    assert np.isclose(ISFC, true_ISFC).all(), \
        "Calculated ISFC (non collapse) does not match ground truth"

<<<<<<< HEAD
    # Test one-sample bootstrap test with pairwise approach
    n_bootstraps = 10

    iscs = isc(data, pairwise=True, summary_statistic=None)
    observed, ci, p, distribution = bootstrap_isc(iscs, pairwise=True,
                                                  summary_statistic=np.median,
                                                  n_bootstraps=n_bootstraps,
                                                  ci_percentile=95,
                                                  return_distribution=True)
    assert distribution.shape == (n_bootstraps, n_voxels)

    # Check random seeds
    iscs = isc(data, pairwise=False, summary_statistic=None)
    distributions = []
    for random_state in [42, 42, None]:
        observed, ci, p, distribution = bootstrap_isc(iscs, pairwise=False,
                                                      summary_statistic=np.median,
                                                      n_bootstraps=n_bootstraps,
                                                      ci_percentile=95,
                                                      return_distribution=True,
                                                      random_state=random_state)
        distributions.append(distribution)
    assert np.array_equal(distributions[0], distributions[1])
    assert not np.array_equal(distributions[1], distributions[2])
    
    # Check output p-values
    data = correlated_timeseries(20, 60, noise=.5,
                                 random_state=42)
    iscs = isc(data, pairwise=False)
    observed, ci, p = bootstrap_isc(iscs, pairwise=False)
    assert np.all(iscs[:, :2] > .5)
    assert np.all(iscs[:, -1] < .5)
    assert p[0, 0] < .05 and p[0, 1] < .05
    assert p[0, 2] > .05
    
    iscs = isc(data, pairwise=True)
    observed, ci, p = bootstrap_isc(iscs, pairwise=True)
    assert np.all(iscs[:, :2] > .5)
    assert np.all(iscs[:, -1] < .5)
    assert p[0, 0] < .05 and p[0, 1] < .05
    assert p[0, 2] > .05
    
    # Check that ISC computation and bootstrap observed are same
    iscs = isc(data, pairwise=False)
    observed, ci, p = bootstrap_isc(iscs, pairwise=False, summary_statistic=np.median)
    assert np.array_equal(observed, isc(data, pairwise=False, summary_statistic=np.median))
    
    # Check that ISC computation and bootstrap observed are same
    iscs = isc(data, pairwise=True)
    observed, ci, p = bootstrap_isc(iscs, pairwise=True, summary_statistic=np.mean)
    assert np.array_equal(observed, isc(data, pairwise=True, summary_statistic=np.mean))
    
    
# Test permutation test with group assignments
def test_permutation_isc():
    group_assignment = [1] * 10 + [2] * 10
    
    # Set parameters for toy time series data
    n_subjects = 20
    n_TRs = 60
    n_voxels = 30
    random_state = 42

    # Create dataset with two groups in pairwise approach
    data = np.dstack((simulated_timeseries(10, n_TRs, n_voxels=n_voxels,
                                           noise=1, data_type='array',
                                           random_state=3),
                      simulated_timeseries(10, n_TRs, n_voxels=n_voxels,
                                           noise=5, data_type='array',
                                           random_state=4)))
    iscs = isc(data, pairwise=True, summary_statistic=None)

    observed, p, distribution = permutation_isc(iscs, group_assignment=group_assignment,
                                                pairwise=True,
                                                summary_statistic=np.mean,
                                                n_permutations=200,
                                                return_distribution=True)

    # Create data with two groups in leave-one-out approach
    data_1 = simulated_timeseries(10, n_TRs, n_voxels=n_voxels,
                                  noise=1, data_type='array',
                                  random_state=3)
    data_2 = simulated_timeseries(10, n_TRs, n_voxels=n_voxels,
                                  noise=10, data_type='array',
                                  random_state=4)
    iscs = np.vstack((isc(data_1, pairwise=False, summary_statistic=None),
                      isc(data_2, pairwise=False, summary_statistic=None)))

    observed, p, distribution = permutation_isc(iscs, group_assignment=group_assignment,
                                                pairwise=False,
                                                summary_statistic=np.median,
                                                n_permutations=200,
                                                return_distribution=True)

    # One-sample leave-one-out permutation test
    data = simulated_timeseries(n_subjects, n_TRs,
                                n_voxels=n_voxels, data_type='array',
                                random_state=random_state)
    iscs = isc(data, pairwise=False, summary_statistic=None)

    observed, p, distribution = permutation_isc(iscs,
                                                pairwise=False,
                                                summary_statistic=np.median,
                                                n_permutations=200,
                                                return_distribution=True)

    # One-sample pairwise permutation test
    iscs = isc(data, pairwise=True, summary_statistic=None)

    observed, p, distribution = permutation_isc(iscs,
                                                pairwise=True,
                                                summary_statistic=np.median,
                                                n_permutations=200,
                                                return_distribution=True)

    # Small one-sample pairwise exact test
    data = simulated_timeseries(12, n_TRs,
                                n_voxels=n_voxels, data_type='array',
                                random_state=random_state)
    iscs = isc(data, pairwise=False, summary_statistic=None)

    observed, p, distribution = permutation_isc(iscs,
                                                pairwise=False,
                                                summary_statistic=np.median,
                                                n_permutations=10000,
                                                return_distribution=True)

    # Small two-sample pairwise exact test (and unequal groups)
    data = np.dstack((simulated_timeseries(3, n_TRs, n_voxels=n_voxels,
                                  noise=1, data_type='array',
                                  random_state=3),
                      simulated_timeseries(4, n_TRs, n_voxels=n_voxels,
                                  noise=50, data_type='array',
                                  random_state=4)))
    iscs = isc(data, pairwise=True, summary_statistic=None)
    group_assignment = [1, 1, 1, 2, 2, 2, 2]

    observed, p, distribution = permutation_isc(iscs,
                                                group_assignment=group_assignment,
                                                pairwise=True,
                                                summary_statistic=np.mean,
                                                n_permutations=10000,
                                                return_distribution=True)

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

    observed, p, distribution = permutation_isc(iscs,
                                                group_assignment=group_assignment,
                                                pairwise=False,
                                                summary_statistic=np.mean,
                                                n_permutations=10000,
                                                return_distribution=True)
    
    # Check output p-values
    data = correlated_timeseries(20, 60, noise=.5,
                                 random_state=42)
    iscs = isc(data, pairwise=False)
    observed, p = permutation_isc(iscs, pairwise=False)
    assert np.all(iscs[:, :2] > .5)
    assert np.all(iscs[:, -1] < .5)
    assert p[0, 0] < .05 and p[0, 1] < .05
    assert p[0, 2] > .1
    
    iscs = isc(data, pairwise=True)
    observed, p = permutation_isc(iscs, pairwise=True)
    assert np.all(iscs[:, :2] > .5)
    assert np.all(iscs[:, -1] < .5)
    assert p[0, 0] < .05 and p[0, 1] < .05
    assert p[0, 2] > .1
    
    # Check that ISC computation and permutation observed are same
    iscs = isc(data, pairwise=False)
    observed, p = permutation_isc(iscs, pairwise=False, summary_statistic=np.median)
    assert np.array_equal(observed, isc(data, pairwise=False, summary_statistic=np.median))
    
    # Check that ISC computation and permuation observed are same
    iscs = isc(data, pairwise=True)
    observed, p = permutation_isc(iscs, pairwise=True, summary_statistic=np.mean)
    assert np.array_equal(observed, isc(data, pairwise=True, summary_statistic=np.mean))


def test_timeshift_isc():
    # Circular time-shift on one sample, leave-one-out
    
    # Set parameters for toy time series data
    n_subjects = 20
    n_TRs = 60
    n_voxels = 30
    random_state = 42
    
    data = simulated_timeseries(n_subjects, n_TRs,
                                n_voxels=n_voxels, data_type='array')
    observed, p, distribution = timeshift_isc(data, pairwise=False,
                                              summary_statistic=np.median,
                                              n_shifts=200,
                                              return_distribution=True)

    # Circular time-shift on one sample, pairwise
    data = simulated_timeseries(n_subjects, n_TRs,
                                n_voxels=n_voxels, data_type='array')
    observed, p, distribution = timeshift_isc(data, pairwise=True,
                                              summary_statistic=np.median,
                                              n_shifts=200,
                                              return_distribution=True)

    # Circular time-shift on one sample, leave-one-out
    data = simulated_timeseries(n_subjects, n_TRs,
                                n_voxels=n_voxels, data_type='array')
    observed, p, distribution = timeshift_isc(data, pairwise=False,
                                              summary_statistic=np.mean,
                                              n_shifts=200,
                                              return_distribution=True)
    # Check output p-values
    data = correlated_timeseries(20, 60, noise=.5,
                                 random_state=42)
    iscs = isc(data, pairwise=False)
    observed, p = timeshift_isc(data, pairwise=False)
    assert np.all(iscs[:, :2] > .5)
    assert np.all(iscs[:, -1] < .5)
    assert p[0, 0] < .05 and p[0, 1] < .05
    assert p[0, 2] > .1
    
    iscs = isc(data, pairwise=True)
    observed, p = timeshift_isc(data, pairwise=True)
    assert np.all(iscs[:, :2] > .5)
    assert np.all(iscs[:, -1] < .5)
    assert p[0, 0] < .05 and p[0, 1] < .05
    assert p[0, 2] > .1
    
    # Check that ISC computation and permutation observed are same
    iscs = isc(data, pairwise=False)
    observed, p = timeshift_isc(data, pairwise=False, summary_statistic=np.median)
    assert np.array_equal(observed, isc(data, pairwise=False, summary_statistic=np.median))
    
    # Check that ISC computation and permuation observed are same
    iscs = isc(data, pairwise=True)
    observed, p = timeshift_isc(data, pairwise=True, summary_statistic=np.mean)
    assert np.array_equal(observed, isc(data, pairwise=True, summary_statistic=np.mean))
    

# Phase randomization test
def test_phaseshift_isc():
    
    # Set parameters for toy time series data
    n_subjects = 20
    n_TRs = 60
    n_voxels = 30
    random_state = 42

    data = simulated_timeseries(n_subjects, n_TRs,
                                n_voxels=n_voxels, data_type='array')
    observed, p, distribution = phaseshift_isc(data, pairwise=True,
                                               summary_statistic=np.median,
                                               n_shifts=200,
                                               return_distribution=True)

    # Phase randomization one-sample test, leave-one-out
    data = simulated_timeseries(n_subjects, n_TRs,
                                n_voxels=n_voxels, data_type='array')
    observed, p, distribution = phaseshift_isc(data, pairwise=False,
                                               summary_statistic=np.mean,
                                               n_shifts=200,
                                               return_distribution=True)
    
    # Check output p-values
    data = correlated_timeseries(20, 60, noise=.5,
                                 random_state=42)
    iscs = isc(data, pairwise=False)
    observed, p = phaseshift_isc(data, pairwise=False)
    assert np.all(iscs[:, :2] > .5)
    assert np.all(iscs[:, -1] < .5)
    assert p[0, 0] < .05 and p[0, 1] < .05
    assert p[0, 2] > .1
    
    iscs = isc(data, pairwise=True)
    observed, p = phaseshift_isc(data, pairwise=True)
    assert np.all(iscs[:, :2] > .5)
    assert np.all(iscs[:, -1] < .5)
    assert p[0, 0] < .05 and p[0, 1] < .05
    assert p[0, 2] > .1
    
    # Check that ISC computation and permutation observed are same
    iscs = isc(data, pairwise=False)
    observed, p = phaseshift_isc(data, pairwise=False, summary_statistic=np.median)
    assert np.array_equal(observed, isc(data, pairwise=False, summary_statistic=np.median))
    
    # Check that ISC computation and permuation observed are same
    iscs = isc(data, pairwise=True)
    observed, p = phaseshift_isc(data, pairwise=True, summary_statistic=np.mean)
    assert np.array_equal(observed, isc(data, pairwise=True, summary_statistic=np.mean))


# Test ISFC 
def test_isfc_options():
    
    # Set parameters for toy time series data
    n_subjects = 20
    n_TRs = 60
    n_voxels = 30
    random_state = 42
    
    from brainiak.fcma.util import compute_correlation
    data = simulated_timeseries(n_subjects, n_TRs,
                                n_voxels=n_voxels, data_type='array')
    isfcs = isfc(data, pairwise=False, summary_statistic=None)

    # Just two subjects
    isfcs = isfc(data[..., :2], pairwise=False, summary_statistic=None)

    # ISFC with pairwise approach
    isfcs = isfc(data, pairwise=True, summary_statistic=None)

    # ISFC with summary statistics
    isfcs = isfc(data, pairwise=True, summary_statistic=np.mean)
    isfcs = isfc(data, pairwise=True, summary_statistic=np.median)

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
        assert np.allclose(isfcs[..., s].diagonal(), iscs[s, :])
        
    # Check that ISC and ISFC diagonal are identical
    iscs = isc(data, pairwise=True)
    isfcs = isfc(data, pairwise=True)
    for s in np.arange(len(iscs)):
        assert np.allclose(isfcs[..., s].diagonal(), iscs[s, :])


if __name__ == '__main__':
    test_isc_input()
    test_isc_options()
    test_isc_output()
    test_bootstrap_isc()
    test_permutation_isc()
    test_timeshift_isc()
    test_phaseshift_isc()
    test_isfc_options()
=======
    assert np.isclose(p, true_p).all(), \
        "Calculated p values (non collapse) do not match ground truth"
>>>>>>> parent of 94d10cc... Fixed location of test_isfc.py and added module imports
