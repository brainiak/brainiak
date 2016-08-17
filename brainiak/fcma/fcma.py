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
"""Full Correlation Matrix Analysis (FCMA)

This implementation is based on the following publications:

.. [Wang2015] Full correlation matrix analysis (FCMA): An unbiased method for
   task-related functional connectivity",
   Yida Wang, Jonathan D Cohen, Kai Li, Nicholas B Turk-Browne.
   Journal of Neuroscience Methods, 2015.

.. [Wang2015] "Full correlation matrix analysis of fMRI data on Intel® Xeon
   Phi™ coprocessors",
   Yida Wang, Michael J. Anderson, Jonathan D. Cohen, Alexander Heinecke, Kai Li,
   Nadathur Satish, Narayanan Sundaram, Nicholas B. Turk-Browne, Theodore L. Willke.
   In Proceedings of the International Conference for High Performance Computing,
   Networking, Storage and Analysis. 2015.
"""

# Authors: Yida Wang
# (Intel Labs), 2016

import numpy as np
import nibabel as nib

# __all__ = ['fcma']

class fcma:
    def run(self):
