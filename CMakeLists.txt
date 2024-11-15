cmake_minimum_required(VERSION 3.15...3.26)

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES C CXX)

find_package(
  Python
  COMPONENTS Interpreter Development.Module NumPy
  REQUIRED)

set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(src/brainiak/fcma)
add_subdirectory(src/brainiak/factoranalysis)
add_subdirectory(src/brainiak/eventseg)