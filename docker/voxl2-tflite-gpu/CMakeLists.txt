# libgate_tflite.so: TensorFlow Lite (from source, with the OpenCL GPU delegate) + gate_tflite.cpp.
#
#   export CMAKE_POLICY_VERSION_MINIMUM=3.5   # needed with CMake 4.x (TFLite deps are old)
#   cmake -S . -B build -DTENSORFLOW_SOURCE_DIR=/opt/tensorflow -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j"$(nproc)" --target gate_tflite benchmark_model
#
# TFLite's own CMake fetches its dependencies (abseil, flatbuffers, XNNPACK, OpenCL headers, ...)
# at configure time, so this needs internet. No OpenCL library is needed to build: the GPU delegate
# opens libOpenCL.so at run time (on the VOXL 2: the Adreno driver from qti-adreno).

cmake_minimum_required(VERSION 3.16)
project(gate_tflite C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # TFLite's static libs go into a shared library

set(TENSORFLOW_SOURCE_DIR "" CACHE PATH "TensorFlow source tree (git clone, e.g. tag v2.17.1)")
if(NOT EXISTS "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite/CMakeLists.txt")
  message(FATAL_ERROR "set -DTENSORFLOW_SOURCE_DIR to a TensorFlow source tree")
endif()

option(GT_WITH_GPU "build TFLite's GPU (OpenCL) delegate in" ON)
set(TFLITE_ENABLE_GPU ${GT_WITH_GPU} CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_XNNPACK ON CACHE BOOL "" FORCE)

add_subdirectory("${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
                 "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)

add_library(gate_tflite SHARED gate_tflite.cpp)
target_link_libraries(gate_tflite PRIVATE tensorflow-lite ${CMAKE_DL_LIBS})
if(GT_WITH_GPU)
  target_compile_definitions(gate_tflite PRIVATE GT_WITH_GPU)
endif()

install(TARGETS gate_tflite LIBRARY DESTINATION lib)
