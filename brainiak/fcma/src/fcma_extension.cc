/*
// Copyright 2016 Intel Corporation
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //      http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.
// */
//
#include "Python.h"
#include <omp.h>
#include <iostream>
#include <sys/time.h>
#include <math.h>
#include <inttypes.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void within_subject_norm_native(py::array_t<float, py::array::c_style |
                                  py::array::forcecast> py_data, int epochsPerSubj)
{
//This function calculates the within-subject normalization of the input correlation values
//
//Input parameters are:
//py_data: the correlation value array, in shape [num_voxels, num_epochs, num_selected_voxels]
//  containing the raw correlation values
//epochsPerSubj: the number pf epochs per subject
//<Note> assuming all subjects have the same number of epochs
//
//Ouput parameters are:
//py_data: the correlation value array, in shape [num_voxels, num_epochs, num_selected_voxels]
//  containing the normalized correlation values
//
    py::buffer_info buf = py_data.request();
    float* data = (float*)buf.ptr;
    // sanity check
    if (buf.ndim != 3)
        throw std::runtime_error("The multi-subject correlation data structure must be 3D");
    size_t nSelectedVoxels = buf.shape[0];
    size_t nEpochs = buf.shape[1];
    size_t nVoxels = buf.shape[2];
    size_t nSubjs = nEpochs / epochsPerSubj;

    #pragma omp parallel for
    for(int v = 0 ; v < nSelectedVoxels*nSubjs ; v++)
    {
        int s = v % nSubjs;  // subject id
        int i = v / nSubjs;  // voxel id

        float *mat = data+i*nEpochs*nVoxels;
        #pragma simd
        for(int j = 0 ; j < nVoxels ; j++)
        {
            float mean = 0.0f;
    	    float std_dev = 0.0f;
            for(int b = s*epochsPerSubj; b < (s+1)*epochsPerSubj; b++)
            {
                float num = 1.0f + mat[b*nVoxels+j];
      	        float den = 1.0f - mat[b*nVoxels+j];
      	        num = (num <= 0.0f) ? 1e-4 : num;
      	        den = (den <= 0.0f) ? 1e-4 : den;
      	        mat[b*nVoxels+j] = 0.5f * logf(num/den);
      	        mean += mat[b*nVoxels+j];
      	        std_dev += mat[b*nVoxels+j] * mat[b*nVoxels+j];
            }
            mean = mean / (float)epochsPerSubj;
            std_dev = std_dev / (float)epochsPerSubj - mean*mean;
            float inv_std_dev = (std_dev <= 0.0f) ? 0.0f : 1.0f / sqrt(std_dev);
            for(int b = s*epochsPerSubj; b < (s+1)*epochsPerSubj; b++)
            {
                mat[b*nVoxels+j] = (mat[b*nVoxels+j] - mean) * inv_std_dev;
            }
        }
    }
    return;
}

PYBIND11_PLUGIN(fcma_extension) {
    py::module m("fcma_extension");
    m.def("normalization", &within_subject_norm_native, "Within-subejct correlation normalization");

#ifdef VERSION_INFO
    m.attr("__version__") = py::str(VERSION_INFO);
#else
    m.attr("__version__") = py::str("dev");
#endif
    return m.ptr();
}
