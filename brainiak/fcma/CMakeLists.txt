cmake_minimum_required (VERSION 2.6)
project (FCMA)

if ($ENV{VERBOSE})
  set (CMAKE_VERBOSE_MAKEFILE ON)
endif ($ENV{VERBOSE})

# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("${PROJECT_BINARY_DIR}")

link_directories ("${PROJECT_SOURCE_DIR}/deps/nifticlib-2.0.0/lib")

set (LD_LIBRARY_PATH "${LD_LIBRARY_PATH}:${PROJECT_SOURCE_DIR}/deps/nifticlib-2.0.0/lib")

# configure a header file to pass some of the CMake settings
# to the source code
#configure_file (
#  "${PROJECT_SOURCE_DIR}/FCMAConfig.h.in"
#  "${PROJECT_BINARY_DIR}/FCMAConfig.h"
#  )

# should we use MKL? otherwise we use OpenBLAS
#option (USE_MKL "Use Intel MKL" ON)

set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lz -lhdf5 -lhdf5_hl -lboost_serialization -lboost_iostreams -lmpi")
set (CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -lz -lhdf5 -lhdf5_hl -lboost_serialization -lboost_iostreams -lmpi")
# set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -DCMM_INT_IS_LONG -std=c++14 -fPIC -flto")

if ($ENV{USE_MKL})
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_MKL")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mkl=parallel")
else ($ENV{USE_MKL})
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lopenblas -lomp")
  set (CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -lopenblas -lomp")
endif ($ENV{USE_MKL})

set(CMAKE_C_COMPILER ${CC})
set(CMAKE_CXX_COMPILER ${CXX})


include_directories ("${PROJECT_SOURCE_DIR}/deps/SVM-phi")
add_subdirectory (deps/SVM-phi)
include_directories ("${PROJECT_SOURCE_DIR}/deps/SVM-Simple")
add_subdirectory (deps/SVM-Simple)
include_directories ("${PROJECT_SOURCE_DIR}/deps/nifticlib-2.0.0/include")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CMAKE_INCLUDE_PATH}")  # either this or last line
link_directories ("${PROJECT_SOURCE_DIR}/deps/nifticlib-2.0.0/lib")
set (EXTRA_LIBS ${EXTRA_LIBS} svmphi svmsimple niftiio znz)

# get all *.cc files recursively
file(GLOB_RECURSE SRC_FILES src/*.cc)

# add the executable
add_subdirectory(pybind11)
pybind11_add_module(fcma ${SRC_FILES})
#add_executable (fcma ${SRC_FILES})
target_link_libraries (fcma ${EXTRA_LIBS})
