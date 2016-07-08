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
#include <algorithm>
#include <inttypes.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void 

factor_native(py::array_t<double, py::array::c_style | py::array::forcecast> F_arr,
              py::array_t<double, py::array::c_style | py::array::forcecast> C_arr,
              py::array_t<double, py::array::c_style | py::array::forcecast> W_arr,
              py::array_t<double, py::array::c_style | py::array::forcecast> Rx_arr,
              py::array_t<double, py::array::c_style | py::array::forcecast> Ry_arr,
              py::array_t<double, py::array::c_style | py::array::forcecast> Rz_arr,
              py::array_t<int64_t, py::array::c_style | py::array::forcecast> Ix_arr,
              py::array_t<int64_t, py::array::c_style | py::array::forcecast> Iy_arr,
              py::array_t<int64_t, py::array::c_style | py::array::forcecast> Iz_arr)
{ auto F_buf = F_arr.request(), 
       C_buf = C_arr.request(),
       W_buf = W_arr.request(),
       Rx_buf = Rx_arr.request(),
       Ry_buf = Ry_arr.request(),
       Rz_buf = Rz_arr.request(),
       Ix_buf = Ix_arr.request(),
       Iy_buf = Iy_arr.request(),
       Iz_buf = Iz_arr.request();
//This function calculates latest factors based on estimation on
//centers and widths, as well as voxels' coordinates
//
//Input parameters are:
//F_arr: The factor array, in shape [n_voxel, n_factor] 
//C_arr: The center array, in shape [n_factor, n_dim]
//W_arr: The width array, in shape [n_factor, 1]. 
//<Note> Factor width is defined as 2*\sigma^2, 
//       where sigma is the standard deviation of Gaussian
//Rx_arr: The unique value of voxel coordinates in x-dimension
//Ry_arr: The unique value of voxel coordinates in y-dimension
//Rz_arr: The unique value of voxel coordinates in z-dimension
//Ix_arr: The index to look up unique value of voxle coordinates in x-dimension
//Iy_arr: The index to look up unique value of voxel coordinates in y-dimension
//Iz_arr: The index to look up unique value of voxle coordinates in z-dimension
//
//Ouput parameters are:
//F_arr: The updated factor array
//
  if(F_buf.ndim != 2)
      throw std::runtime_error("F must be 2D");
  
  if(C_buf.ndim != 2)
      throw std::runtime_error("C must be 2D");

  if(W_buf.ndim != 2)
      throw std::runtime_error("W must be 2D"); 

  if(Rx_buf.ndim != 1)
      throw std::runtime_error("Rx must be 1D");
  
  if(Ry_buf.ndim != 1)
      throw std::runtime_error("Ry must be 1D");

  if(Rz_buf.ndim != 1)
      throw std::runtime_error("Rz must be 1D");

  if(Ix_buf.ndim != 1)
      throw std::runtime_error("Ix must be 1D");

  if(Iy_buf.ndim != 1)
      throw std::runtime_error("Iy must be 1D");

  if(Iz_buf.ndim != 1)
      throw std::runtime_error("Iz must be 1D");


  int V = F_buf.shape[0];
  int K = F_buf.shape[1];
  int D = C_buf.shape[1]; 
  int nx = Rx_buf.shape[0];
  int ny = Ry_buf.shape[0];
  int nz = Rz_buf.shape[0];

  if(Ix_buf.shape[0] != V)
      throw std::runtime_error("Ix must have V elements!");

  if(Iy_buf.shape[0] != V)
      throw std::runtime_error("Iy must have V elements!");

  if(Iz_buf.shape[0] != V)
      throw std::runtime_error("Iz must have V elements!");
    
  if(C_buf.shape[0] != K)
      throw std::runtime_error("C_buf.shape[0] must be K!");
  
  if(W_buf.shape[0] != K)
      throw std::runtime_error("W_buf.shape[0] must be K!");
 
  double *F = (double*) F_buf.ptr;
  double *C = (double*) C_buf.ptr; 
  double *W = (double*) W_buf.ptr;
  double *Rx = (double*) Rx_buf.ptr;
  double *Ry = (double*) Ry_buf.ptr;
  double *Rz = (double*) Rz_buf.ptr;
  int64_t *Ix = (int64_t*) Ix_buf.ptr;
  int64_t *Iy = (int64_t*) Iy_buf.ptr;
  int64_t *Iz = (int64_t*) Iz_buf.ptr;

  double invW[K];
  double tmp1 = 0.0;
  double tmp2 = 0.0;
  double tmp3 = 0.0;

  #pragma omp parallel for
  for(int k = 0 ; k < K; k++)
  {
      double Tx[nx];
      double Ty[ny];
      double Tz[nz];
      invW[k] = 1.0/W[k];
      for(int x = 0 ; x < nx;  x++)
      {   
         tmp1 = Rx[x]-C[k*D];
         Tx[x] = exp(-1.0*tmp1*tmp1*invW[k]);
      }

      for(int y = 0 ; y < ny;  y++)
      {   
         tmp2 = Ry[y]-C[k*D+1];
         Ty[y] = exp(-1.0*tmp2*tmp2*invW[k]);
      }
      
      for(int z = 0 ; z < nz;  z++)
      {   
         tmp3 = Rz[z]-C[k*D+2];
         Tz[z] = exp(-1.0*tmp3*tmp3*invW[k]);
      }
      
      for(int v = 0 ; v < V;  v++)
      {   
         F[v*K+k] = Tx[Ix[v]] * Ty[Iy[v]] * Tz[Iz[v]];
      }
  }
  
  return;
}

void

recon_native(py::array_t<double, py::array::c_style | py::array::forcecast> recon_arr,
              py::array_t<double, py::array::c_style | py::array::forcecast> X_arr,
              py::array_t<double, py::array::c_style | py::array::forcecast> F_arr,
              py::array_t<double, py::array::c_style | py::array::forcecast> W_arr,
              py::array_t<double, py::array::c_style | py::array::forcecast> s_arr)
{ auto recon_buf = recon_arr.request(),
       X_buf = X_arr.request(),
       F_buf = F_arr.request(),
       W_buf = W_arr.request(),
       s_buf = s_arr.request();
//This function calculates the reconstruction error 
//centers and widths, as well as voxels' coordinates
//
//Input parameters are:
//recon_arr: The reconstruction error 
//X_arr: The original fMRI data, in shape [n_voxel, n_tr] 
//F_arr: The factor array, in shape [n_voxel, n_factor] 
//W_arr: The weight array, in shape [n_factor, n_tr]
//s_arr: The subsampling coeffient 
//
//Ouput parameters are:
//recon_arr: The updated reconstruction error 
//
  double *recon = (double*) recon_buf.ptr;
  double *X = (double*) X_buf.ptr;
  double *F = (double*) F_buf.ptr;
  double *W = (double*) W_buf.ptr;
  double *s = (double*) s_buf.ptr;

  if(recon_buf.ndim != 1)
      throw std::runtime_error("recon must be 1D");

  if(X_buf.ndim != 2)
      throw std::runtime_error("recon must be 2D");

  if(F_buf.ndim != 2)
      throw std::runtime_error("F must be 2D");
  
  if(W_buf.ndim != 2)
      throw std::runtime_error("W must be 2D");

  if(s_buf.ndim != 1)
      throw std::runtime_error("s must be 1D");
  
  if(X_buf.shape[0] != F_buf.shape[0])
      throw std::runtime_error("X and F must have same dims[0]");
    
  if(X_buf.shape[1] != W_buf.shape[1])
      throw std::runtime_error("X and W must have same dims[1]");
  
  if(X_buf.shape[0]*X_buf.shape[1] != recon_buf.shape[0])
      throw std::runtime_error("X and recon must have same num of elements");
  
  #pragma omp parallel for
  for(int v = 0 ; v < X_buf.shape[0] ; v++)
  {   
    int  idx = 0;       
    float tmp = 0.0;
    for(int t = 0 ; t < X_buf.shape[1] ; t++)
    {
      idx = v * X_buf.shape[1] + t;
      tmp = X[idx];
      for(int k = 0 ; k < F_buf.shape[1] ; k++)
      {
       tmp -= W[k * X_buf.shape[1] + t] * F[v*F_buf.shape[1]+k];       
      }
    recon[idx] =  s[0] * tmp;
    }
  }
  return;
}


PYBIND11_PLUGIN(tfa_extension) {
    py::module m("tfa_extension");
    m.def("factor", &factor_native, "Calculate factor matrix");
    m.def("recon", &recon_native, "Reconstruct data");

#ifdef VERSION_INFO
    m.attr("__version__") = py::str(VERSION_INFO);
#else
    m.attr("__version__") = py::str("dev");
#endif

    return m.ptr();
}
