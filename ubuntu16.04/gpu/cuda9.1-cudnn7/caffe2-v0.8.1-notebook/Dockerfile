# grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-caffe2-v0.8.1-notebook
#
# Caffe2 jupyter notebook with gpu suppport (nvidia CUDA cuDNN).
#
# sudo nvidia-docker build -t grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-caffe2-v0.8.1-notebook https://github.com/grez72/notebooks.git#master:ubuntu16.04/gpu/cuda9.1-cudnn7/caffe2-v0.8.1-notebook
# nvidia-docker run -it --rm --shm-size=2G -p 8888:8888 grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-caffe2-v0.8.1-notebook

FROM grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-minimal-notebook

MAINTAINER George Alvarez <gre72@gmail.com, alvarez@wjh.harvard.edu>

ARG CAFFE2_VERSION=v0.8.0

USER root
WORKDIR $HOME

# ==============================================================
#  install Caffe2 with gpu support
# ==============================================================

RUN /bin/bash -c "source activate python2 && apt-get update && apt-get install -y --no-install-recommends \
    python-dev \
    python-pip \
    git \
    build-essential \
    cmake \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-glog-dev"

USER $NB_USER
# WORKDIR $HOME/work/networks
RUN /bin/bash -c "source activate python2 && git clone --recursive https://github.com/caffe2/caffe2.git && \
    pip install --upgrade pip && \
    pip install numpy protobuf"

########## OPTIONAL DEPENDENCIES ################

USER root

RUN /bin/bash -c "source activate python2 && apt-get install -y --no-install-recommends \
    libgflags-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libiomp-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libopenmpi-dev \
    libsnappy-dev \
    openmpi-bin \
    openmpi-doc \
    python-numpy \
    python-pydot \
    python-setuptools \
    python-scipy \
    wget"

RUN /bin/bash -c "source activate python2 && pip install \
    flask \
    graphviz \
    hypothesis \
    jupyter \
    matplotlib \
    pydot \
    python-nvd3 \
    pyyaml \
    requests \
    scikit-image \
    scipy \
    setuptools \
    tornado"

########## INSTALLATION STEPS ###################
RUN /bin/bash -c 'source activate python2 && cd caffe2 && make -j"$(nproc)" && cd build && make -j"$(nproc)" install'
RUN /bin/bash -c "source activate python2 && pip install future && pip install future --upgrade"

# export PYTHONPATH=/usr/local
# export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
# export LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

ENV PYTHONPATH /usr/local
ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH

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
