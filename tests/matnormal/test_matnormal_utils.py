from brainiak.matnormal.utils import pack_trainable_vars, unpack_trainable_vars
import tensorflow as tf


def test_pack_unpack():

    shapes = [[2, 3], [3], [3, 4, 2], [1, 5]]
    mats = [tf.random.stateless_normal(shape=shape, seed=[0, 0]) for shape in shapes]
    flatmats = pack_trainable_vars(mats)
    unflatmats = unpack_trainable_vars(flatmats, mats)
    for mat_in, mat_out in zip(mats, unflatmats):
        assert tf.math.reduce_all(tf.equal(mat_in, mat_out))
