pybind11_add_module(tfa_extension tfa_extension.cpp)

install(TARGETS tfa_extension LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME}/factoranalysis)