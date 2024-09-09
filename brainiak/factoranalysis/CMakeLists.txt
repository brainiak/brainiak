pybind11_add_module(tfa_extension tfa_extension.cpp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(tfa_extension PUBLIC OpenMP::OpenMP_CXX)
else()


endif()

install(TARGETS tfa_extension LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME}/factoranalysis)