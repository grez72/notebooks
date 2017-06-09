# grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-caffe-v1.0-notebook
#
# Caffe jupyter notebook with gpu suppport (nvidia CUDA cuDNN).
#
# sudo nvidia-docker build -t grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-caffe-v1.0-notebook https://github.com/grez72/notebooks.git#master:ubuntu14.04/gpu/cuda8-cudnn5/caffe-v1.0-notebook
# nvidia-docker run -it --rm -p 8888:8888 grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-caffe-v1.0-notebook

FROM grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-base-notebook

MAINTAINER George Alvarez <gre72@gmail.com, alvarez@wjh.harvard.edu>

ARG CAFFE_VERSION=v1.0

# ==============================================================
#  INSTALL Caffe
# ==============================================================

USER root

# install caffe dependencies
RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      git \
      wget \
      bc \
      cmake \
      libatlas-base-dev \
      libatlas-dev \
      libboost-all-dev \
      libopencv-dev \
      libprotobuf-dev \
      libgoogle-glog-dev \
      libgflags-dev \
      protobuf-compiler \
      libhdf5-dev \
      libhdf5-serial-dev \
      libleveldb-dev \
      liblmdb-dev \
      libsnappy-dev \
      python-dev \
      python-pip \
      python-numpy \
      python-setuptools \
      python-scipy && \
    rm -rf /var/lib/apt/lists/*

# install caffe
ENV CAFFE_ROOT=$HOME/caffe
WORKDIR $HOME
USER $NB_USER
RUN git clone https://github.com/BVLC/caffe.git $CAFFE_ROOT

# default python is python 3.5, so we need to use pip2 to install for python2
WORKDIR $CAFFE_ROOT/python
USER root
RUN for req in $(cat requirements.txt) pydot; do pip2 install $req; done

WORKDIR $CAFFE_ROOT
USER $NB_USER
RUN git clone https://github.com/NVIDIA/nccl.git

WORKDIR $CAFFE_ROOT/nccl
USER root
RUN make -j install

WORKDIR $CAFFE_ROOT
RUN rm -rf nccl && \
    mkdir build

WORKDIR $CAFFE_ROOT/build
RUN cmake \
    -DUSE_CUDNN=1 \
    -DUSE_NCCL=1 \
    -DUSE_MPI=1 \
    -DMPI_INCLUDE=/usr/include/openmpi \
    -DCUDA_ARCH_NAME="Manual" \
    -DCUDA_ARCH_BIN="52 60" \
    -DCUDA_ARCH_PTX="60" .. && \
    make -j"$(nproc)"

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

# additional dependencies
RUN apt-get update && apt-get install -y python-skimage python-protobuf

# make runtest

# # Download some sample data for testing
# WORKDIR $CAFFE_ROOT
# RUN ./data/mnist/get_mnist.sh
# RUN ./examples/mnist/create_mnist.sh

# ==============================================================
#  set workdir and user for jupyter notebook
# ==============================================================

WORKDIR $HOME/work/
USER $NB_USER
