# grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-caffe-v1.0-notebook
#
# Caffe jupyter notebook with gpu suppport (nvidia CUDA cuDNN).
#
# sudo nvidia-docker build -t grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-caffe-v1.0-notebook https://github.com/grez72/notebooks.git#master:ubuntu14.04/gpu/cuda8-cudnn5/caffe-v1.0-notebook
# nvidia-docker run -it --rm -p 8888:8888 grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-caffe-v1.0-notebook

FROM grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-base-notebook

MAINTAINER George Alvarez <gre72@gmail.com, alvarez@wjh.harvard.edu>

ARG NVCAFFE_VERSION=0.16.1

# ==============================================================
#  INSTALL Caffe from NVIDIA's fork
# ==============================================================

WORKDIR $HOME
USER root

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

ENV CAFFE_HOME = $HOME/caffe
RUN mkdir $CAFFE_HOME
WORKDIR $CAFFE_HOME
RUN git clone https://github.com/NVIDIA/caffe.git .
WORKDIR $CAFFE_HOME
RUN git checkout -r 0.16.1
# RUN cat $CAFFE_HOME/python/requirements.txt | xargs -n1 pip2 install
#     # pip2 install -r $CAFFE_HOME/python/requirements.txt
# RUN mkdir build
# WORKDIR $CAFFE_HOME/build
# RUN cmake .. && \
#     make -j"$(nproc)"

# ARG CAFFE_VERSION=1.0
#
# # ==============================================================
# #  INSTALL Caffe
# # ==============================================================
#
# USER root
#
# # install caffe dependencies
# RUN pip install --upgrade pip
# RUN apt-get update
# RUN apt-get install -y --no-install-recommends \
#       build-essential \
#       git \
#       wget \
#       bc \
#       cmake \
#       libatlas-base-dev \
#       libatlas-dev \
#       libboost-all-dev \
#       libopencv-dev \
#       libprotobuf-dev \
#       libgoogle-glog-dev \
#       libgflags-dev \
#       protobuf-compiler \
#       libhdf5-dev \
#       libhdf5-serial-dev \
#       libleveldb-dev \
#       liblmdb-dev \
#       libsnappy-dev \
#       python-dev \
#       python-pip \
#       python-numpy \
#       python-setuptools \
#       python-scipy && \
#     rm -rf /var/lib/apt/lists/*
#
# # install caffe
# ENV CAFFE_ROOT=/opt/caffe
# RUN mkdir $CAFFE_ROOT
# WORKDIR $CAFFE_ROOT
#
# RUN git clone -b ${CAFFE_VERSION} --depth 1 https://github.com/BVLC/caffe.git . && \
#     pip install --upgrade pip
#
# # default python is python 3.5, so we need to use pip2 to install for python2
# WORKDIR $CAFFE_ROOT/python
# RUN for req in $(cat requirements.txt) pydot; do pip2 install $req; done
#
# WORKDIR $CAFFE_ROOT
# RUN git clone https://github.com/NVIDIA/nccl.git
#
# WORKDIR $CAFFE_ROOT/nccl
# RUN make -j install
#
# WORKDIR $CAFFE_ROOT
# RUN rm -rf nccl && \
#     mkdir build
#
# WORKDIR $CAFFE_ROOT/build
# RUN cmake -DUSE_CUDNN=1 -DUSE_NCCL=1 -DUSE_MPI=1 -DMPI_INCLUDE=/usr/include/openmpi .. && \
#     make -j"$(nproc)"
#
# ENV PYCAFFE_ROOT $CAFFE_ROOT/python
# ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
# ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
# RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig
#
# # additional dependencies
# RUN apt-get update && apt-get install -y python-skimage python-protobuf
#
# # # own it
# # RUN chown -R jovyan:users $CAFFE_ROOT/examples
# #
# # # Download some sample data for testing
# # WORKDIR $CAFFE_ROOT
# # RUN ./data/mnist/get_mnist.sh
# # RUN ./examples/mnist/create_mnist.sh
#
# # ==============================================================
# #  set workdir and user for jupyter notebook
# # ==============================================================
#
# WORKDIR $HOME/work/
# USER $NB_USER
#
# # Copying examples, models, and scripts as $NB_USER, so that
# # we can modify these files / directories in jupyter notebook.
# # e.g., if we don't do this, then scripts that download data will fail
# # because they cannot create the necessary folders.
# RUN cp -R $CAFFE_ROOT/examples $HOME/work && \
#     cp -R $CAFFE_ROOT/models $HOME/work && \
#     cp -R $CAFFE_ROOT/scripts $HOME/work && \
#     cp -R $CAFFE_ROOT/data $HOME/work && \
#     cp -R $CAFFE_ROOT/include $HOME/work && \
#     cp -R $CAFFE_ROOT/python/caffe/imagenet $HOME/work
