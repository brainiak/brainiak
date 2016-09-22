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


def test_cov2corr():
    from brainiak.utils.utils import cov2corr
    import numpy as np
    cov = np.array([[4,3,0],[3,9,0],[0,0,1]])
    corr = cov2corr(cov)
    assert np.isclose(corr, np.array([[1,0.5,0],[0.5,1,0],[0,0,1]])).all(),\
        "Converting from covariance matrix to correlation incorrect"


def test_ReadDesign():
    from brainiak.utils.utils import ReadDesign
    import numpy as np
    import os.path
    file_path = os.path.join(os.path.dirname(__file__), "example_design.1D")
    design = ReadDesign(fname=file_path)
    assert design, 'Failed to read design matrix'
    read = ReadDesign()
    assert read, 'Failed to initialize an instance of the class'