pybind11_add_module(tfa_extension tfa_extension.cpp)

find_package(OpenMP REQUIRED)

target_link_libraries(tfa_extension PRIVATE OpenMP::OpenMP_CXX)

install(TARGETS tfa_extension LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME}/factoranalysis)