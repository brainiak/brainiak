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

def compute_correlation(py_trans_a, py_trans_b, py_m, py_n, py_k, py_alpha, py_a, py_lda,
          int py_start_voxel, py_ldb, py_beta, py_c, py_ldc, int py_start_epoch):
    """ use blas API wrapped by scipy.linalg.cython_blas to compute correlation

    The blas APIs process matrices in column-major,
    but our matrices are in row-major,
    so we play the transpose trick here, i.e. A*B=(B^T*A^T)^T.
    The resulting matrix in shape [num_assigned_voxels, num_voxels]
    is stored in an alternate way to make sure that
    the correlation vectors of the same voxel stored continuously

    Parameters
    ----------
    py_trans_a: str
    do transpose or not for the first matrix A

    py_trans_b: str
    do transpose or not for the first matrix B

    py_m: int
    the row of the resulting matrix C
    in our case, is num_voxels

    py_n: int
    the column of the resulting matrix C
    in our case, is num_assigned_voxels

    py_k: int
    the collapsed dimension of the multiplying matrices
    i.e. the column of the first matrix after transpose if necessary
    the row of the second matrix after transpose if necessary

    py_alpha: float
    the weight applied to the first matrix A

    py_a: 2D array in shape [epoch_length, num_voxels] in our case
    the activity data of an epoch

    py_lda: int
    the stride of the first matrix A

    py_start_voxel: int
    the starting voxel of assigned voxels
    used to locate the second matrix B

    py_ldb: int
    the stride of the second matrix B

    py_beta: float
    the weight applied to the resulting matrix C

    py_c: 3D array in shape [num_selected_voxels, num_epochs, num_voxels]
    place to store the resulting correlation values

    py_ldc: int
    the stride of the resulting matrix
    in our case, num_voxels*num_epochs

    py_start_epoch: int
    the epoch over which the correlation is computed

    Returns
    -------
    py_c: 3D array in shape [num_selected_voxels, num_epochs, num_voxels]
    write the resulting correlation values in an alternate way
    for the processing epoch
    """
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
    blas.sgemm(trans_a, trans_b, &M, &N, &K, &alpha, &A[0, 0], &lda,
               &A[0, py_start_voxel], &ldb, &beta, &C[0, py_start_epoch,0], &ldc)

def compute_kernel_matrix(py_uplo, py_trans, py_n, py_k, py_alpha, py_a, int py_start_voxel, py_lda,
                          py_beta, py_c, py_ldc):
    """ use blas API wrapped by scipy.linalg.cython_blas to compute kernel matrix of SVM

    The blas APIs process matrices in column-major, but our matrices are in row-major,
    so we play the transpose trick here, i.e. A*B=(B^T*A^T)^T

    In SVM with linear kernel, the distance of two samples
    is essentially the dot product of them.
    Therefore, the kernel matrix can be obtained by matrix multiplication.
    Since the kernel matrix is symmetric, ssyrk is used,
    the other half of the matrix is assigned later.
    In our case, the dimension of samples is much larger than
    the number samples, so we proportionally shrink the values of the kernel matrix
    for getting more robust alpha values in SVM iteration.

    Parameters
    ----------
    py_uplo: str
    getting the upper or lower triangle of the matrix

    py_trans: str
    do transpose or not for the input matrix A

    py_n: int
    the row and column of the resulting matrix C
    in our case, is num_epochs

    py_k: int
    the collapsed dimension of the multiplying matrices
    i.e. the column of the first matrix after transpose if necessary
    the row of the second matrix after transpose if necessary
    in our case, is num_voxels

    py_alpha: float
    the weight applied to the input matrix A

    py_a: 3D array in shape [num_assigned_voxels, num_epochs, num_voxels] in our case
    the normalized correlation values of a voxel

    py_start_voxel: int
    the processed voxel
    used to locate the input matrix A

    py_lda: int
    the stride of the input matrix A

    py_beta: float
    the weight applied to the resulting matrix C

    py_c: 2D array in shape [num_epochs, num_epochs]
    place to store the resulting kernel matrix

    py_ldc: int
    the stride of the resulting matrix

    Returns
    -------
    py_c: 2D array in shape [num_epochs, num_epochs]
    write the resulting kernel_matrix
    for the processing voxel
    """
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
    blas.ssyrk(uplo, trans, &N, &K, &alpha, &A[py_start_voxel, 0, 0], &lda,
               &beta, &C[0, 0], &ldc)
    # shrink the values for getting more stable alpha values in SVM training iteration
    num_digits = len(str(int(py_c[0, 0])))
    if (num_digits > 2):
        proportion = 10**(2-num_digits)
        py_c *= proportion
    # complete the other half of the kernel matrix
    if (py_uplo=='L'):
        for j in range(py_c.shape[0]):
            for k in range(j):
                py_c[j, k] = py_c[k, j]
    else:
        for j in range(py_c.shape[0]):
            for k in range(j):
                py_c[k, j] = py_c[j, k]

def installed():
    """
    This is an empty method for installing cython_blas library
    """
