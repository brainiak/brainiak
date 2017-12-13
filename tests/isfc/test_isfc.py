import brainiak.isfc
from brainiak import image, io
import numpy as np
import os


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
