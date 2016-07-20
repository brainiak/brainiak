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

"""Example for using topological data analysis

Simulate some neural data and run it through the pipeline for TDA analysis

Authors: Cameron Ellis (Princeton
"""

from brainiak import tda
import numpy as np


def main():

    # Simulate data
    n = 10
    volume = np.random.randn(np.power(n, 2)).reshape(n, n)

    volume = tda.preprocess(volume)

    tda_input = tda.convert_space(volume)

    print(tda_input)

if __name__ == "__main__":
    main()

