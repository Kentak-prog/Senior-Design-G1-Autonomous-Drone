# VOXL 2 container with OpenCL (the Adreno 650 GPU) and TensorFlow Lite's GPU delegate, for our
# Python gate vision. Build ON THE DRONE (it is ARM; an x86 laptop would have to emulate it):
#
#   ./repack_deps.sh                       # once: the drone's GPU driver files -> deps/gpu-driver.tar
#   docker build -t voxl2-tflite-gpu .     # needs internet (station mode); 30-90 min the first time
#   ./run.sh opencl_check                  # the GPU is visible in the container
#
# OpenCL follows ModalAI's "OpenCL in Docker" recipe (docs.modalai.com/voxl-2-opencl-in-docker):
# the drone's Adreno user-space driver goes into the image (as a tar of the installed files rather
# than repacked .debs: see repack_deps.sh); the GPU device nodes are passed in at run time (run.sh).
# Also builds on a Mac (Apple Silicon is arm64 too): docker build --platform linux/arm64, then
# docker save | gzip, adb push, docker load on the drone. TFLite is built from source (TF_TAG): the drone's own
# qrb5165-tflite is older (2.8.0 on SDK 1.5.1) and has no Python bindings.

ARG UBUNTU=arm64v8/ubuntu:22.04

# ---------------------------------------------------------------- build: TFLite + libgate_tflite
FROM ${UBUNTU} AS build
ARG TF_TAG=v2.17.1
ARG BUILD_JOBS=6
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates python3 \
 && rm -rf /var/lib/apt/lists/*
# Some of TFLite's fetched dependencies declare a CMake minimum below 3.5; as an environment
# variable this reaches their sub-builds too (a -D option does not).
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
RUN git clone --depth 1 --branch ${TF_TAG} https://github.com/tensorflow/tensorflow.git /opt/tensorflow
COPY CMakeLists.txt gate_tflite.cpp /opt/gate_tflite/
RUN cmake -S /opt/gate_tflite -B /opt/build -DTENSORFLOW_SOURCE_DIR=/opt/tensorflow \
          -DCMAKE_BUILD_TYPE=Release \
 && cmake --build /opt/build -j${BUILD_JOBS} --target gate_tflite benchmark_model \
 && strip /opt/build/libgate_tflite.so /opt/build/tensorflow-lite/tools/benchmark/benchmark_model

# ---------------------------------------------------------------- runtime
FROM ${UBUNTU}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 libpcre3 zlib1g python3 python3-pip \
        gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir numpy opencv-python-headless onnxruntime

# The drone's Adreno GPU driver (libOpenCL.so and what it needs), bundled by repack_deps.sh. Then
# what qti-adreno's install script does: link its headers (*.adreno) into /usr/include, ldconfig.
COPY deps/gpu-driver.tar /opt/gpu-driver.tar
RUN tar -xf /opt/gpu-driver.tar -C / && rm /opt/gpu-driver.tar \
 && for h in $(find /opt/qcom/usr/include -name '*.adreno'); do \
        t="/usr/include/${h#/opt/qcom/usr/include/}"; t="${t%.adreno}"; \
        mkdir -p "$(dirname "$t")" && ln -sf "$h" "$t"; \
    done \
 && ldconfig

COPY --from=build /opt/build/libgate_tflite.so /usr/local/lib/
COPY --from=build /opt/build/tensorflow-lite/tools/benchmark/benchmark_model /usr/local/bin/
COPY opencl_check.c /opt/opencl_check.c
RUN gcc -O2 -Wall -o /usr/local/bin/opencl_check /opt/opencl_check.c -lOpenCL && ldconfig

ENV GATE_TFLITE_LIB=/usr/local/lib/libgate_tflite.so
WORKDIR /repo
CMD ["bash"]
