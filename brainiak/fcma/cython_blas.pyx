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

cimport scipy.linalg.cython_blas as blas

import numpy as np

def compute_correlation(py_trans_a, py_trans_b, py_m, py_n, py_k, py_alpha, py_a, py_lda,
          py_start_voxel, py_ldb, py_beta, py_c, py_ldc, py_start_epoch):
    cdef bytes by_trans_a=py_trans_a.encode()
    cdef bytes by_trans_b=py_trans_b.encode()
    cdef char* trans_a = by_trans_a
    cdef char* trans_b = by_trans_b
    cdef int M, N, K, lda, ldb, ldc
    M = py_m
    N = py_n
    K = py_k
    lda = py_lda
    ldb = py_ldb
    ldc = py_ldc
    cdef float alpha, beta
    alpha = py_alpha
    beta = py_beta
    cdef float[:, ::1] A
    A = py_a
    cdef float[:, :, ::1] C
    C = py_c
    blas.sgemm(trans_a, trans_b, &M, &N, &K, &alpha, &A[0,0], &lda,
               &A[0,py_start_voxel], &ldb, &beta, &C[0,py_start_epoch,0], &ldc)

def compute_kernel_matrix(py_uplo, py_trans, py_n, py_k, py_alpha, py_a, py_start_voxel, py_lda,
                          py_beta, py_c, py_ldc):
    cdef bytes by_uplo=py_uplo.encode()
    cdef bytes by_trans=py_trans.encode()
    cdef char* uplo = by_uplo
    cdef char* trans = by_trans
    cdef int N, K, lda, ldc
    N = py_n
    K = py_k
    lda = py_lda
    ldc = py_ldc
    cdef float alpha, beta
    alpha = py_alpha
    beta = py_beta
    cdef float[:, :, ::1] A
    A = py_a
    cdef float[:, ::1] C
    C = py_c
    blas.ssyrk(uplo, trans, &N, &K, &alpha, &A[py_start_voxel,0,0], &lda,
               &beta, &C[0,0], &ldc)
    # complete the upper triangle of the kernel matrix
    # shrink the values for getting more stable alpha values in SVM training iteration
    py_c *= .001
    for j in range(py_c.shape[0]):
        for k in range(j):
            py_c[j,k] = py_c[k,j]
