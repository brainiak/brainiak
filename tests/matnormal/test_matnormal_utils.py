from brainiak.matnormal.utils import (pack_trainable_vars,
                                      unpack_trainable_vars,
                                      flatten_cholesky_unique,
                                      unflatten_cholesky_unique)
import tensorflow as tf
import numpy as np
import numpy.testing as npt


def test_pack_unpack(seeded_rng):

    shapes = [[2, 3], [3], [3, 4, 2], [1, 5]]
    mats = [tf.random.stateless_normal(
        shape=shape, seed=[0, 0]) for shape in shapes]
    flatmats = pack_trainable_vars(mats)
    unflatmats = unpack_trainable_vars(flatmats, mats)
    for mat_in, mat_out in zip(mats, unflatmats):
        assert tf.math.reduce_all(tf.equal(mat_in, mat_out))


def test_cholesky_uncholesky(seeded_rng):
    size = 3
    flat_chol_length = (size*(size+1))//2
    flatchol = np.random.normal(size=[flat_chol_length])
    unflatchol = unflatten_cholesky_unique(flatchol)
    npt.assert_equal(unflatchol.shape, [3, 3])
    reflatchol = flatten_cholesky_unique(unflatchol)
    npt.assert_allclose(flatchol, reflatchol)
