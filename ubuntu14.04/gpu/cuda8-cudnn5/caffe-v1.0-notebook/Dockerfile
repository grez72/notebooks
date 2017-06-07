# grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-caffe-v1.0-notebook
#
# Caffe2 jupyter notebook with gpu suppport (nvidia CUDA cuDNN).
#
# sudo nvidia-docker build -t grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-caffe-v1.0-notebook https://github.com/grez72/notebooks.git#master:ubuntu14.04/gpu/cuda8-cudnn5/caffe-v1.0-notebook
# nvidia-docker run -it --rm -p 8888:8888 grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-caffe-v1.0-notebook

FROM grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-base-notebook

MAINTAINER George Alvarez <gre72@gmail.com, alvarez@wjh.harvard.edu>

# using NVIDIA's fork of caffe (v0.15.13)
ARG CAFFE_VERSION=0.16.1

# ==============================================================
#  INSTALL Caffe
# ==============================================================

USER root
WORKDIR $HOME

# install caffe dependencies
RUN pip install --upgrade pip
RUN apt-get update
RUN apt-get install -y \
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
    gfortran > /dev/null

# install nccl
RUN wget https://github.com/NVIDIA/nccl/releases/download/v1.2.3-1%2Bcuda8.0/libnccl1_1.2.3-1.cuda8.0_amd64.deb && \
    dpkg -i libnccl1_1.2.3-1.cuda8.0_amd64.deb && \
    rm libnccl1_1.2.3-1.cuda8.0_amd64.deb && \
    wget https://github.com/NVIDIA/nccl/releases/download/v1.2.3-1%2Bcuda8.0/libnccl-dev_1.2.3-1.cuda8.0_amd64.deb && \
    dpkg -i libnccl-dev_1.2.3-1.cuda8.0_amd64.deb && \
    rm libnccl-dev_1.2.3-1.cuda8.0_amd64.deb

# install caffe
WORKDIR /opt
RUN wget https://github.com/NVIDIA/caffe/archive/v${CAFFE_VERSION}.tar.gz
RUN tar -zxf v${CAFFE_VERSION}.tar.gz
RUN mv ${CAFFE_VERSION} caffe
WORKDIR /opt/caffe
RUN cat python/requirements.txt | xargs -n1 pip2 install
RUN cp Makefile.config.example Makefile.config
RUN echo "USE_CUDNN := 1" >> Makefile.config
RUN echo "USE_NCCL := 1" >> Makefile.config
RUN echo "INCLUDE_DIRS := \$(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial/" >> Makefile.config
RUN echo "LIBRARY_DIRS := \$(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial" >> Makefile.config

RUN make all -j"$(nproc)"
RUN make test
# RUN make runtest
RUN make pycaffe
RUN apt-get install -y python-skimage python-protobuf

# export PYTHONPATH
ENV CAFFE_ROOT=/opt/caffe
ENV PYTHONPATH=/opt/caffe/distribute/python:$PYTHONPATH
ENV PYTHONPATH=/opt/caffe/python:$PYTHONPATH

# necessary?
# ENV PYCAFFE_ROOT $CAFFE_ROOT/python
# ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
# ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
# RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

# download mnist data so what we can test our installation
RUN cd $CAFFE_ROOT
RUN ./data/mnist/get_mnist.sh
RUN ./examples/mnist/create_mnist.sh
# cd $CAFFE_ROOT && ./examples/mnist/train_lenet.sh

# when I run python2, >import caffe I get the following error
# "libdc1394 error: Failed to initialize libdc1394"
# http://stackoverflow.com/questions/29274638/opencv-libdc1394-error-failed-to-initialize-libdc1394
# as per SO, the following line fixes it, but I'm not sure why...
# see also http://stackoverflow.com/questions/42149458/open-cv-error-failed-to-init-raw1394-persisting-in-docker/42151836#42151836
RUN ln /dev/null /dev/raw1394
