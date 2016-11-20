// Copyright Intel Corporation 2016

#include <iostream>
#include <vector>
#include <list>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;

inline size_t getIdx3(const std::vector<size_t> & pt,
                  const std::vector<size_t> & strides)
{
  return pt[0]*strides[0] + pt[1]*strides[1] + pt[2]*strides[2];
}

inline size_t getIdx4(const std::vector<size_t> & pt,
                  const std::vector<size_t> & strides)
{
  return pt[0]*strides[0] + pt[1]*strides[1] + pt[2]*strides[2] + pt[3]*strides[3];
}

py::array_t<double> yah(std::vector< py::array_t<double, py::array::c_style | py::array::forcecast> > data,
         py::array_t<double, py::array::c_style | py::array::forcecast> & _mask,
         size_t rad,
         py::object bcast_var)
{
  auto _mask_buf = _mask.request();
  if(_mask_buf.ndim != 3)
  {
    throw std::runtime_error("Mask must be a 3D numpy array");
  }
  double * mask = (double*) _mask_buf.ptr;

  // If region is too small, return empty array
  if(_mask_buf.shape[0] <= 2*rad ||
     _mask_buf.shape[1] <= 2*rad ||
     _mask_buf.shape[2] <= 2*rad)
  {
    std::vector<size_t> outshape = {0,0,0};
    std::vector<size_t> outstrides = {sizeof(double),sizeof(double),sizeof(double)};
    return py::array(py::buffer_info(NULL, sizeof(double), py::format_descriptor<double>::value, 3, outshape, outstrides));
  }

  // Create output array
  std::vector<size_t> outshape = {_mask_buf.shape[0]-2*rad, 
                                             _mask_buf.shape[1]-2*rad,
                                             _mask_buf.shape[2]-2*rad};

  std::vector<size_t> outstrides = {outshape[2]*outshape[1]*sizeof(double),
                                               outshape[2]*sizeof(double),
                                               sizeof(double)};
  std::vector<double> outmat(outshape[0]*outshape[1]*outshape[2]);

  std::vector<double*> subj_data;
  std::vector<std::vector<size_t> > subj_strides;
  std::vector<std::vector<size_t> > subj_shapes;

  for(size_t subj = 0 ; subj < data.size() ; subj++)
  {
    auto subj_buf = data[subj].request();
    subj_data.push_back((double*)subj_buf.ptr);
    subj_strides.push_back(subj_buf.strides);
    subj_shapes.push_back(subj_buf.shape);
  }
  #pragma omp parallel for schedule(dynamic) collapse(3)
  for(size_t i = rad ; i < _mask_buf.shape[0]-rad ; i++)
  {
    for(size_t j = rad ; j < _mask_buf.shape[1]-rad ; j++)
    {
      for(size_t k = rad ; k < _mask_buf.shape[2]-rad ; k++)
      {
        if(mask[getIdx3({i,j,k},_mask_buf.strides)/sizeof(double)] > 0) 
        {
          double sum = 0.0;
          size_t niter = 0;
          for(size_t subj = 0 ; subj < data.size() ; subj++)
          {
            size_t base_pt = getIdx4({0,i,j,k},subj_strides[subj])/sizeof(double);
            size_t cnt = 0;
            for(size_t zz = 0 ; zz < subj_shapes[subj][0] ; zz++)
            {
              for(size_t ii = i-rad ; ii < i+rad+1 ; ii++)
              {
                for(size_t jj = j-rad ; jj < j+rad+1 ; jj++)
                {
                  for(size_t kk = k-rad ; kk < k+rad+1 ; kk++)
                  {
                    sum += subj_data[subj][(zz*subj_strides[subj][0] +
                                           ii*subj_strides[subj][1] + 
                                           jj*subj_strides[subj][2] + 
                                           kk*subj_strides[subj][3]) / sizeof(double)];
                    niter++;
                  }
                }
              }
            }
          }
          outmat[getIdx3({i-rad,j-rad,k-rad},outstrides)/sizeof(double)] = sum;
        }
      }
    }
  }
  return py::array(py::buffer_info(outmat.data(), sizeof(double), py::format_descriptor<double>::value, 3, outshape, outstrides));
}


PYBIND11_PLUGIN(example_fn) {
    py::module m("example_fn");
    m.def("yah", &yah, "yah");

#ifdef VERSION_INFO
    m.attr("__version__") = py::str(VERSION_INFO);
#else
    m.attr("__version__") = py::str("dev");
#endif

    return m.ptr();
}
