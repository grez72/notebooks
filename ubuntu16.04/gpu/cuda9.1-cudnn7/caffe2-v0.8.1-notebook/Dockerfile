# grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-caffe2-v0.8.1-notebook
#
# Caffe2 jupyter notebook with gpu suppport (nvidia CUDA cuDNN).
#
# sudo nvidia-docker build -t grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-caffe2-v0.8.1-notebook https://github.com/grez72/notebooks.git#master:ubuntu16.04/gpu/cuda9.1-cudnn7/caffe2-v0.8.1-notebook
# nvidia-docker run -it --rm --shm-size=2G -p 8888:8888 grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-caffe2-v0.8.1-notebook

FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04
LABEL maintainer="aaronmarkham@fb.com"

# caffe2 install with gpu support

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
    libopencv-dev \
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

########## INSTALLATION STEPS ###################
RUN git clone --branch master --recursive https://github.com/caffe2/caffe2.git
RUN cd caffe2 && mkdir build && cd build \
    && cmake .. \
    -DCUDA_ARCH_NAME=Manual \
    -DCUDA_ARCH_BIN="35 52 60 61" \
    -DCUDA_ARCH_PTX="61" \
    -DUSE_NNPACK=OFF \
    -DUSE_ROCKSDB=OFF \
    && make -j"$(nproc)" install \
    && ldconfig \
    && make clean \
    && cd .. \
    && rm -rf build

ENV PYTHONPATH /usr/local

# FROM grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-minimal-notebook
# # FROM grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-base-notebook
#
# MAINTAINER George Alvarez <gre72@gmail.com, alvarez@wjh.harvard.edu>
#
# ARG CAFFE2_VERSION=v0.8.0
#
# USER root
# WORKDIR $HOME

# ==============================================================
#  install Caffe2 with gpu support
# ==============================================================

# Install required packages
# mkl isn't actually required, but is really recommended for Anaconda
# RUN /bin/bash -c 'source activate python2 \
#   && conda install -y mkl -c intel \
#   && conda install -y \
#     future \
#     gflags \
#     glog \
#     lmdb \
#     mkl \
#     mkl-include \
#     numpy \
#     protobuf \
#     snappy \
#     six'

# clone and build caffe2
# RUN /bin/bash -c "source activate python2 \
#   && git clone --recursive https://github.com/caffe2/caffe2.git \
#   && git checkout $CAFFE2_VERSION
#   && cd caffe2 \
#   && cd $HOME/caffe2 \
#   && make -j$(nproc) USE_NATIVE_ARCH=1 \
#   && cd build \
#   && make -j$(nproc) install \
#   && pip install future"

# Make Caffe2 in a separate folder to avoid polluting the Caffe2 source tree.
# This can be anywhere you want it
RUN git clone --recursive https://github.com/caffe2/caffe2.git && cd caffe2
RUN git checkout v0.8.0
RUN mkdir build && cd build

# Add flags to this command to control which packages you want to use. The
# options to use are at the top of CMakeLists.txt in the topmost Caffe2
# directory
RUN cmake -DUSE_CUDA=ON -DUSE_MKL=ON ..
RUN make -j$(nproc) install

# RUN /bin/bash -c "source activate python2 && apt-get update && apt-get install -y --no-install-recommends \
#     python-dev \
#     python-pip \
#     git \
#     build-essential \
#     cmake \
#     libprotobuf-dev \
#     protobuf-compiler \
#     libgoogle-glog-dev"
#
# USER $NB_USER
# # WORKDIR $HOME/work/networks
# RUN /bin/bash -c "source activate python2 && git clone --recursive https://github.com/caffe2/caffe2.git && \
#     pip install --upgrade pip && \
#     pip install numpy protobuf"
#
# ########## OPTIONAL DEPENDENCIES ################
#
# USER root
#
# RUN /bin/bash -c "source activate python2 && apt-get install -y --no-install-recommends \
#     libgflags-dev \
#     libgoogle-glog-dev \
#     libgflags2 \
#     libgtest-dev \
#     libiomp-dev \
#     libleveldb-dev \
#     liblmdb-dev \
#     libopencv-dev \
#     libopenmpi-dev \
#     libsnappy-dev \
#     openmpi-bin \
#     openmpi-doc \
#     python-numpy \
#     python-pydot \
#     python-setuptools \
#     python-scipy \
#     wget"
#
# RUN /bin/bash -c "source activate python2 && pip install \
#     flask \
#     graphviz \
#     hypothesis \
#     jupyter \
#     matplotlib \
#     pydot \
#     python-nvd3 \
#     pyyaml \
#     requests \
#     scikit-image \
#     scipy \
#     setuptools \
#     tornado"
#
# ########## INSTALLATION STEPS ###################
# RUN /bin/bash -c 'source activate python2 && cd caffe2 && make -j"$(nproc)" && cd build && make -j"$(nproc)" install'
# RUN /bin/bash -c "source activate python2 && pip install future && pip install future --upgrade"
#
# # export PYTHONPATH=/usr/local
# # export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
# # export LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
#
# ENV PYTHONPATH /usr/local
# ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
# ENV LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH

# test installation
# python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
# test gpu
# python -m caffe2.python.operator_test.relu_op_test
# python -c 'from caffe2.python import workspace; print(workspace.NumCudaDevices())'

# ==============================================================
#  make python2 the default
# ==============================================================

# RUN echo 'export PATH="/opt/conda/envs/python2/bin:$PATH"' > ~/.bashrc \
#   && /bin/bash -c "source ~/.bashrc"

# # ==============================================================
# #  caffe2 dependencies
# # ==============================================================
#
# # required
# RUN /bin/bash -c "source activate python2 && apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     cmake \
#     git \
#     python-dev \
#     python-numpy \
#     python-pip \
#     python-pydot \
#     python-setuptools \
#     python-scipy \
#     libprotobuf-dev \
#     protobuf-compiler \
#     libgoogle-glog-dev \
#     wget \
#     && rm -rf /var/lib/apt/lists/*"
#
# # optional
# RUN /bin/bash -c "source activate python2 && apt-get update && apt-get install -y --no-install-recommends \
#     libgflags-dev \
#     libgoogle-glog-dev \
#     libgflags-dev \
#     libgtest-dev \
#     libiomp-dev \
#     libleveldb-dev \
#     liblmdb-dev \
#     libopencv-dev \
#     libopenmpi-dev \
#     libsnappy-dev \
#     openmpi-bin \
#     openmpi-doc \
#     python-numpy \
#     python-pydot \
#     python-setuptools \
#     python-scipy \
#     wget"
#
# RUN /bin/bash -c "source activate python2 && pip install --upgrade pip && pip install \
#     flask \
#     graphviz \
#     hypothesis \
#     jupyter \
#     matplotlib \
#     numpy \
#     protobuf \
#     pydot \
#     python-nvd3 \
#     pyyaml \
#     requests \
#     scikit-image \
#     scipy \
#     setuptools \
#     tornado"
#
# # Caffe2 requires zeromq 4.0 or above, manually install.
# # If you do not need zeromq, skip this step.
# RUN apt-get install -q -y autoconf libtool
# RUN mkdir /tmp/zeromq-build && \
#   cd /tmp/zeromq-build && \
#   wget https://github.com/zeromq/zeromq4-1/archive/v4.1.3.tar.gz && \
#   tar xzvf v4.1.3.tar.gz --strip 1 && \
#   ./autogen.sh && \
#   ./configure --without-libsodium && \
#   make && make install && \
#   cd / && \
#   rm -rf /tmp/zeromq-build
#
# # update pip, numpy, and install mkl
# RUN /bin/bash -c "source activate python2 \
#   && pip install --upgrade pip \
#   && pip install numpy --upgrade \
#   && conda install -y mkl -c intel"
#
# # ==============================================================
# #  INSTALL CAFFE2
# # ==============================================================
#
# WORKDIR $HOME
#
# RUN /bin/bash -c "source activate python2 && cd caffe2 && make -j'$(nproc)' && cd build && make -j'$(nproc)' install"
# RUN /bin/bash -c "source activate python2 && pip install future"
#
# # RUN git clone --recursive https://github.com/caffe2/caffe2.git
#
# # currently, going to branch v0.8.1 leads to compilation error
# # so we'll just stay with the master branch:
# RUN
#
#   && /bin/bash -c "source activate python2 \
#   && git clone --recursive https://github.com/caffe2/caffe2.git \
#   && cd caffe2 \
#   && cd $HOME/caffe2 \
#   && make -j$(nproc) USE_NATIVE_ARCH=1 \
#   && cd build \
#   && make -j$(nproc) install \
#   && pip install future"
#
# ENV PYTHONPATH /usr/local:$PYTHONPATH
# ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
# ENV LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
# # RUN python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
#
# # ==============================================================
# #  set workdir and user for jupyter notebook
# # ==============================================================
#
# WORKDIR /home/$NB_USER/work/
# USER $NB_USER
