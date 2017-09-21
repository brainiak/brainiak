import brainiak.isfc
from brainiak import image, io
import numpy as np
import os


def test_ISC():
    D = np.zeros((2, 5, 3))
    D[:, :, 0] = \
        [[-0.36225433, -0.43482456,  0.26723158,  0.16461712, -0.37991465],
         [-0.62305959, -0.46660116, -0.50037994,  1.81083754,  0.23499509]]
    D[:, :, 1] = \
        [[-0.31484153, -0.47486988,  0.18966625, -0.12568572, -0.40535156],
         [-0.78433298, -0.35875085, -0.6121344,  1.68267639,  0.28603493]]
    D[:, :, 2] = \
        [[-0.26593192, -0.56914734,  0.28397317,  0.30276589, -0.42637472],
         [-0.67598379, -0.51549055, -0.64196342,  1.60686666,  0.04127293]]

    ISC = brainiak.isfc.isc(D)
    assert np.isclose(ISC, [0.9540602, 0.99585304]).all(), \
        "Calculated ISC does not match ground truth"


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

    ISFC = brainiak.isfc.isfc(D)

    ground_truth = \
        [[1, 1, 0, -1],
         [1, 1, 0, -1],
         [0, 0, 1,  0],
         [-1, -1, 0, 1]]

    assert np.isclose(ISFC, ground_truth).all(), \
        "Calculated ISFC does not match ground truth"
