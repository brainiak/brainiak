add_custom_command(
  OUTPUT cython_blas.c
  COMMENT
    "Making ${CMAKE_CURRENT_BINARY_DIR}/cython_blas.c from ${CMAKE_CURRENT_SOURCE_DIR}/cython_blas.pyx"
  COMMAND Python::Interpreter -m cython
          "${CMAKE_CURRENT_SOURCE_DIR}/cython_blas.pyx" --output-file cython_blas.c
  DEPENDS cython_blas.pyx
  VERBATIM)

python_add_library(cython_blas MODULE cython_blas.c WITH_SOABI)
pybind11_add_module(fcma_extension MODULE src/fcma_extension.cc)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fcma_extension PUBLIC OpenMP::OpenMP_CXX)
endif()

install(TARGETS cython_blas DESTINATION ${SKBUILD_PROJECT_NAME}/fcma)
install(TARGETS fcma_extension LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME}/fcma)