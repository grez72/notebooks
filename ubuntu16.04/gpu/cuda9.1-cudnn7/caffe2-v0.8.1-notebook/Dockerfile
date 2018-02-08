# grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-caffe2-v0.8.1-notebook
#
# Caffe2 jupyter notebook with gpu suppport (nvidia CUDA cuDNN).
#
# sudo nvidia-docker build -t grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-caffe2-v0.8.1-notebook https://github.com/grez72/notebooks.git#master:ubuntu16.04/gpu/cuda9.1-cudnn7/caffe2-v0.8.1-notebook
# nvidia-docker run -it --rm --shm-size=2G -p 8888:8888 grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-caffe2-v0.8.1-notebook
#
# tests
# nvidia-docker run -it --rm --shm-size=2G grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-caffe2-v0.8.1-notebook python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
# nvidia-docker run -it --rm --shm-size=2G grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-caffe2-v0.8.1-notebook python -m caffe2.python.operator_test.relu_op_test
# nvidia-docker run -it --rm --shm-size=2G grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-caffe2-v0.8.1-notebook python -c 'from caffe2.python import workspace; print(workspace.NumCudaDevices())'
# nvidia-docker run -it --rm --shm-size=2G grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-caffe2-v0.8.1-notebook bash

FROM grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-minimal-notebook

MAINTAINER George Alvarez <gre72@gmail.com, alvarez@wjh.harvard.edu>

ARG CAFFE2_VERSION=v0.8.1

USER root
WORKDIR $HOME

# =======================================================
#   Protobuf
# =======================================================

# Uninstall Conda's Protobuf, which interfered with installation
RUN conda uninstall --quiet --yes protobuf

# =======================================================
#   INSTALL caffe2 install with gpu support
# =======================================================

USER root
WORKDIR $HOME

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgflags-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libiomp-dev \
    libleveldb-dev \
    liblmdb-dev \
    # we already have a GPU enabled version of opencv 3.4 installed
    # libopencv-dev \
    libopenmpi-dev \
    libprotobuf-dev \
    libsnappy-dev \
    openmpi-bin \
    openmpi-doc \
    protobuf-compiler \
    python-dev \
    python-numpy \
    python-pip \
    python-pydot \
    python-setuptools \
    python-scipy \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    flask \
    future \
    graphviz \
    hypothesis \
    jupyter \
    matplotlib \
    numpy \
    protobuf \
    pydot \
    python-nvd3 \
    pyyaml \
    requests \
    scikit-image \
    scipy \
    setuptools \
    six \
    tornado

# ==============================================================
#  install Caffe2 with gpu support
# ==============================================================

RUN git clone --branch master --recursive https://github.com/caffe2/caffe2.git \
  && cd caffe2 \
  && git checkout $CAFFE2_VERSION

# make the following change to caffe2/cmake/Dependencies.cmake deal with an error when using opencv 3.4
RUN sed -i 's/find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)/find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs superres)/g' caffe2/cmake/Dependencies.cmake

# run installation
RUN cd caffe2 \
    && mkdir build && cd build \
    && cmake -DUSE_CUDA=ON \
    -DCUDA_ARCH_NAME=Manual \
    -DCUDA_ARCH_BIN="35 52 60 61" \
    -DCUDA_ARCH_PTX="61" \
    -USE_NATIVE_ARCH=1 \
    -DUSE_FFMPEG=ON \
    -DUSE_OPENCV=ON \
    .. \
    && make -j"$(nproc)" install \
    && ldconfig \
    && make clean

# install for anaconda python3 and python2
# RUN cd $HOME/caffe2 \
#     && python setup.py install \
#     && /bin/bash -c "source activate python2 && python setup.py install" \
#     && cd build \
#     && rm -rf build

# RUN cd $HOME/caffe2 \
#     && CMAKE_ARGS='-DUSE_CUDA=ON \
#     -DCUDA_ARCH_NAME=Manual \
#     -DCUDA_ARCH_BIN="35 52 60 61" \
#     -DCUDA_ARCH_PTX="61" \
#     -DUSE_FFMPEG=ON \
#     -DUSE_OPENCV=ON' \
#     python setup.py install
#     # && /bin/bash -c "source activate python2 && python setup.py install" \
#
#     # && cd build \
#     # && rm -rf build
#
# ENV PYTHONPATH /usr/local

# ########## INSTALLATION STEPS ###################
# RUN /bin/bash -c 'source activate python2 && cd caffe2 && make -j"$(nproc)" && cd build && make -j"$(nproc)" install'
# RUN /bin/bash -c "source activate python2 && pip install future && pip install future --upgrade"

# ENV PYTHONPATH /usr/local
# ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
# ENV LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH

# test installation
# python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
# test gpu
# python -m caffe2.python.operator_test.relu_op_test
# python -c 'from caffe2.python import workspace; print(workspace.NumCudaDevices())'

# # ==============================================================
# #  set workdir and user for jupyter notebook
# # ==============================================================
#
# WORKDIR /home/$NB_USER/work/
# USER $NB_USER
