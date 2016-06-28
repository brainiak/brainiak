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


def test_fast_inv():
    from brainiak.utils.utils import fast_inv
    import numpy as np

    a = np.random.rand(6)
    with pytest.raises(TypeError) as excinfo:
        fast_inv(a)
    assert "Input matrix should be 2D array" in str(excinfo.value)
    a = np.random.rand(3, 2)
    with pytest.raises(np.linalg.linalg.LinAlgError) as excinfo:
        fast_inv(a)
    assert "Last 2 dimensions of the array must be square" in str(excinfo.value)
