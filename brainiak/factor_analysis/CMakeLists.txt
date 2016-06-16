cmake_minimum_required(VERSION 2.8.12)
project(brainiak)

add_subdirectory(pybind11)
pybind11_add_module(tfa_extension tfa_extension.cpp)
