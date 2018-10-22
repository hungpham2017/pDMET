cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(cpp_libdmet)

set(anaconda "/nfs/roc/groups/6/gagliard/phamx494/anaconda")

# Set INCLUDE and LIB 
LINK_DIRECTORIES(${anaconda}/lib)
INCLUDE_DIRECTORIES(${anaconda}/include)

# Find the LAWrap for mkl_blas/lapack, pybind11
find_package(LAWrap CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "Found LAWrap: ${LAWrap_CONFIG}")
message(STATUS "Found pybind11: ${pybind11_CONFIG}")

# Set python
set(PYTHON_EXECUTABLE)

# Set pybind11 c++ standard
set(PYBIND11_CPP_STANDARD -std=c++11)

# Creates a python module named "module_name"
pybind11_add_module(libdmet MODULE libdmet.cpp)
#target_link_libraries(libdmet mkl_intel_lp64 mkl_sequential mkl_core) TODO: remove mkl?
target_link_libraries(libdmet lapack blas mkl_intel_lp64 mkl_sequential mkl_core)