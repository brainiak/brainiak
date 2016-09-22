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


from mpi4py import MPI
import numpy as np
import brainiak.searchlight.searchlight
import pytest

def test_create_searchlight():
  sl = brainiak.searchlight.searchlight.Searchlight()
  assert sl, "Invalid searchlight instance"

def test_two_subject_rad_1():

  def fn(a, mask):
    return np.mean(a[0]) + np.mean(a[1])

  M=5
  N=7
  K=12
  D=10
  R=1

  data = None
  mask = None
  if MPI.COMM_WORLD.Get_rank() == 0:
    data = [0.5*np.ones((D,M,N,K)), 0.5*np.ones((D,M,N,K))]
    mask = np.ones((M,N,K))

  sl = brainiak.searchlight.searchlight.Searchlight()
  output = sl.run(data, mask, fn, R)

  # Check output
  EPS = 1e-5
  if MPI.COMM_WORLD.Get_rank() == 0:
    for i in range(R,M-R):
      for j in range(R,N-R):
        for k in range(R,K-R):
          assert abs(output[i,j,k] - 1.0) < EPS, "Invalid output " + str((i,j,k))

def test_one_subject_rad_0():

  def fn(a, mask):
    return np.mean(a[0])

  M=5
  N=7
  K=12
  D=10
  R=1

  data = None
  mask = None
  if MPI.COMM_WORLD.Get_rank() == 0:
    data = [np.ones((D,M,N,K))]
    mask = np.ones((M,N,K))

  sl = brainiak.searchlight.searchlight.Searchlight()
  output = sl.run(data, mask, fn, R)

  # Check output
  EPS = 1e-5
  if MPI.COMM_WORLD.Get_rank() == 0:
    for i in range(R,M-R):
      for j in range(R,N-R):
        for k in range(R,K-R):
          assert abs(output[i,j,k] - 1.0) < EPS, "Invalid output " + str((i,j,k))

def test_null_fn():

  fn = None

  M=5
  N=7
  K=12
  D=10
  R=1

  data = None
  mask = None
  if MPI.COMM_WORLD.Get_rank() == 0:
    data = [np.ones((D,M,N,K))]
    mask = np.ones((M,N,K))

  sl = brainiak.searchlight.searchlight.Searchlight()
  with pytest.raises(TypeError):
    output = sl.run(data, mask, fn, R)

    # Check output
    EPS = 1e-5
    if MPI.COMM_WORLD.Get_rank() == 0:
      for i in range(R,M-R):
        for j in range(R,N-R):
          for k in range(R,K-R):
            assert abs(output[i,j,k] - 1.0) < EPS, "Invalid output " + str((i,j,k))



