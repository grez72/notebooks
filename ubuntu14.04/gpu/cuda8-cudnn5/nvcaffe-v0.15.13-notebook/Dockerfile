# grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-nvcaffe-v0.15.13-notebook
#
# Caffe jupyter notebook with gpu suppport (nvidia CUDA cuDNN).
#
# sudo nvidia-docker build -t grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-nvcaffe-v0.15.13-notebook https://github.com/grez72/notebooks.git#master:ubuntu14.04/gpu/cuda8-cudnn5/nvcaffe-v0.15.13-notebook
# nvidia-docker run -it --rm -p 8888:8888 grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-nvcaffe-v0.15.13-notebook

FROM grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-base-notebook

MAINTAINER George Alvarez <gre72@gmail.com, alvarez@wjh.harvard.edu>

ARG NVCAFFE_VERSION=v0.15.13
# v0.16.1 requires cudnn6
# ARG NVCAFFE_VERSION=v0.16.1

# ==============================================================
#  INSTALL Caffe from NVIDIA's fork
# ==============================================================

WORKDIR $HOME
USER root

# install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      gfortran \
      libatlas-base-dev \
      libboost-all-dev \
      libgflags-dev \
      libgoogle-glog-dev \
      libhdf5-serial-dev \
      libleveldb-dev \
      liblmdb-dev \
      libopencv-dev \
      libprotobuf-dev \
      libsnappy-dev \
      protobuf-compiler \
      python-all-dev \
      python-dev \
      python-h5py \
      python-matplotlib \
      python-numpy \
      python-opencv \
      python-pil \
      python-pip \
      python-protobuf \
      python-scipy \
      python-skimage \
      python-sklearn

# clone caffe
ENV CAFFE_HOME $HOME/caffe
USER $NB_USER
RUN git clone https://github.com/NVIDIA/caffe.git /home/jovyan/caffe/
WORKDIR $CAFFE_HOME
RUN git checkout tags/${NVCAFFE_VERSION}

# add NCCL to caffe folder
RUN git clone https://github.com/NVIDIA/nccl.git
WORKDIR $CAFFE_HOME/nccl
USER root
RUN make -j install
WORKDIR $CAFFE_HOME
RUN rm -rf nccl

# install python dependencies
RUN cat $CAFFE_HOME/python/requirements.txt | xargs -n1 pip2 install

# install caffe
RUN mkdir build
WORKDIR $CAFFE_HOME/build
RUN cmake \
    -DCUDA_ARCH_NAME="Manual" \
    -DCUDA_ARCH_BIN="52 60" \
    -DCUDA_ARCH_PTX="60" .. && \
    make -j"$(nproc)"
ENV PYCAFFE_ROOT=$CAFFE_HOME/python
ENV PYTHONPATH=$PYCAFFE_ROOT:$PYTHONPATH
ENV PATH=$CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_HOME/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

# make runtest

# ==============================================================
#  set workdir and user for jupyter notebook
# ==============================================================

WORKDIR $HOME/work/
USER $NB_USER
