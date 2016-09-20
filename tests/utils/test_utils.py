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


def test_sumexp():
    from brainiak.utils.utils import sumexp_stable
    import numpy as np

    data = np.array([[1, 1],[0, 1]])
    sums, maxs, exps = sumexp_stable(data)
    assert sums.size == data.shape[1], "Invalid sum(exp(v)) computation (wrong # samples in sums)"
    assert exps.shape[0] == data.shape[0], "Invalid exp(v) computation (wrong # features)"
    assert exps.shape[1] == data.shape[1], "Invalid exp(v) computation (wrong # samples)"
    assert maxs.size == data.shape[1], "Invalid max computation (wrong # samples in maxs)"


def test_concatenate_list():
    from brainiak.utils.utils import concatenate_list
    import numpy as np
    l = [None] * 5

    l[1] = np.array([0, 1, 2])
    l[3] = np.array([3, 4])
    try:
        r = concatenate_list(l, axis=0)
    except:
        assert True, "Could not concatenate a list"
    assert np.all(np.arange(5) == r), "Invalid concatenation of a list of arrays"