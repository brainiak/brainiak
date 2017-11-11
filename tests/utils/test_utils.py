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
             'AFNI1': 'example_stimtime_1_AFNI.txt'}
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
    assert np.all(np.isclose(design4, design5[::2])), (
        'design matrices sampled at different frequency do not match'
        ' at corresponding time points')
    design6 = gen_design(stimtime_files=[files['AFNI1']],
                         scan_duration=[48, 20], TR=2, style='AFNI')
    assert np.all(np.isclose(design1, design6)), (
        'design matrices generated from AFNI style and FSL style do not match')


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
