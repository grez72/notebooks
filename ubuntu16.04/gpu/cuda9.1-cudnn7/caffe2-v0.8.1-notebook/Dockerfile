# grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-caffe2-v0.8.1-notebook
#
# Caffe2 jupyter notebook with gpu suppport (nvidia CUDA cuDNN).
#
# sudo nvidia-docker build -t grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-caffe2-v0.8.1-notebook https://github.com/grez72/notebooks.git#master:ubuntu16.04/gpu/cuda9.1-cudnn7/caffe2-v0.8.1-notebook
# nvidia-docker run -it --rm --shm-size=2G -p 8888:8888 grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-caffe2-v0.8.1-notebook

FROM grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-base-notebook

MAINTAINER George Alvarez <gre72@gmail.com, alvarez@wjh.harvard.edu>

USER root

ENV CAFFE2_VERSION=0.8.1

# ==============================================================
#  caffe2 dependencies
# ==============================================================

# required
RUN /bin/bash -c "source activate python2 && apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python-dev \
    python-numpy \
    python-pip \
    python-pydot \
    python-setuptools \
    python-scipy \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-glog-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*"

# optional
RUN /bin/bash -c "source activate python2 && apt-get update && apt-get install -y --no-install-recommends \
    libgflags-dev \
    libgoogle-glog-dev \
    libgflags2 \
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

RUN /bin/bash -c "source activate python2 && pip install --upgrade pip && pip install \
    flask \
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
    tornado"

# ==============================================================
#  INSTALL CAFFE2
# ==============================================================

WORKDIR $HOME
RUN git clone --branch v${CAFFE2_VERSION} --recursive https://github.com/caffe2/caffe2.git \
RUN /bin/bash -c "source activate python2 && cd caffe2 && make -j'$(nproc)' && cd build && make -j'$(nproc)' install"
RUN /bin/bash -c "source activate python2 && pip install future"

ENV PYTHONPATH /usr/local:$PYTHONPATH
ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
# RUN python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"

# ==============================================================
#  set workdir and user for jupyter notebook
# ==============================================================

WORKDIR /home/$NB_USER/work/
USER $NB_USER
