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

import numpy as np
from mpi4py import MPI
import sys

from brainiak.searchlight.searchlight import Searchlight
from brainiak.searchlight.searchlight import Diamond5x5x5

"""Diamond shape test
"""

def test_diamond5x5x5():
  diamond = Diamond5x5x5()
  x = True
  o = False
  true_diamond = [[[o,o,o,o,o],
                   [o,o,o,o,o],
                   [o,o,x,o,o],
                   [o,o,o,o,o],
                   [o,o,o,o,o]],
                  [[o,o,o,o,o],
                   [o,x,x,x,o],
                   [o,x,x,x,o],
                   [o,x,x,x,o],
                   [o,o,o,o,o]],
                  [[o,o,x,o,o],
                   [o,x,x,x,o],
                   [x,x,x,x,x],
                   [o,x,x,x,o],
                   [o,o,x,o,o]],
                  [[o,o,o,o,o],
                   [o,x,x,x,o],
                   [o,x,x,x,o],
                   [o,x,x,x,o],
                   [o,o,o,o,o]],
                  [[o,o,o,o,o],
                   [o,o,o,o,o],
                   [o,o,x,o,o],
                   [o,o,o,o,o],
                   [o,o,o,o,o]]]

  assert np.all(diamond == true_diamond)

test_diamond5x5x5()
