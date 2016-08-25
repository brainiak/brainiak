cmake_minimum_required (VERSION 2.6)
project (FCMA_EXT)

if ($ENV{VERBOSE})
  set (CMAKE_VERBOSE_MAKEFILE ON)
endif ($ENV{VERBOSE})

# add the binary tree to the search path for include files
include_directories("${PROJECT_BINARY_DIR}")

#set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
set (CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -lomp")
# set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -DCMM_INT_IS_LONG -std=c++14 -fPIC -flto")

#if ($ENV{USE_MKL})
#  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_MKL")
#  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mkl=parallel")
#else ($ENV{USE_MKL})
#  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lopenblas -lomp")
#  set (CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -lopenblas -lomp")
#endif ($ENV{USE_MKL})

set(CMAKE_C_COMPILER ${CC})
set(CMAKE_CXX_COMPILER ${CXX})

# get all *.cc files recursively
file(GLOB_RECURSE SRC_FILES src/*.cc)

# add the library
add_subdirectory(pybind11)
pybind11_add_module(fcma_extension ${SRC_FILES})