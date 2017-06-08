# grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-caffe-v1.0-notebook
#
# Caffe jupyter notebook with gpu suppport (nvidia CUDA cuDNN).
#
# sudo nvidia-docker build -t grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-caffe-v1.0-notebook https://github.com/grez72/notebooks.git#master:ubuntu14.04/gpu/cuda8-cudnn5/caffe-v1.0-notebook
# nvidia-docker run -it --rm -p 8888:8888 grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-caffe-v1.0-notebook

FROM grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-base-notebook

MAINTAINER George Alvarez <gre72@gmail.com, alvarez@wjh.harvard.edu>

ARG CAFFE_VERSION=1.0

# ==============================================================
#  INSTALL Caffe
# ==============================================================

USER root

# install caffe dependencies
RUN pip install --upgrade pip
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
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
ENV CAFFE_ROOT=/opt/caffe
RUN mkdir $CAFFE_ROOT
WORKDIR $CAFFE_ROOT

RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/BVLC/caffe.git . && \
    pip install --upgrade pip && \
    cd python && \
    for req in $(cat requirements.txt) pydot; do pip install $req; done && cd .. && \
    for req in $(cat requirements.txt) pydot; do pip2 install $req; done && cd .. && \
    git clone https://github.com/NVIDIA/nccl.git && cd nccl && make -j install && cd .. && rm -rf nccl && \
    mkdir build && cd build && \
    cmake -DUSE_CUDNN=1 -DUSE_NCCL=1 .. && \
    make -j"$(nproc)"

# RUN git clone -b ${CAFFE_VERSION} --depth 1 https://github.com/BVLC/caffe.git
# RUN pip install --upgrade pip
#
# WORKDIR python
# RUN for req in $(cat requirements.txt) pydot; do pip install $req; done
# RUN for req in $(cat requirements.txt) pydot; do pip2 install $req; done
#
# WORKDIR $CAFFE_ROOT
# RUN git clone https://github.com/NVIDIA/nccl.git
#
# WORKDIR nccl
# RUN make -j install
#
# WORKDIR $CAFFE_ROOT
# RUN rm -rf nccl && \
#     mkdir build
#
# WORKDIR build
# RUN cmake -DUSE_CUDNN=1 -DUSE_NCCL=1 .. && \
#     make -j"$(nproc)"

# export PYCAFFE_ROOT=$CAFFE_ROOT/python
# export PYTHONPATH=$PYCAFFE_ROOT:$PYTHONPATH
# export PATH=$CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH

# ENV PYCAFFE_ROOT $CAFFE_ROOT/python
# ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
# ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
# RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

# ENV PYTHONPATH /opt/caffe/python
#
# # Add caffe binaries to path
# ENV PATH $PATH:/opt/caffe/.build_release/tools

# RUN ln -s /usr/include/python2.7/ /usr/local/include/python2.7 && \
#   ln -s /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ /usr/local/include/python2.7/numpy
#
# RUN apt-get update && apt-get install -y python-skimage python-protobuf

# verify installation:
# python -c "import caffe;print caffe.__version__"
# python2 -c "import caffe;print caffe.__version__"
# python3 -c "import caffe;print(caffe.__version__)"

# RUN cd $CAFFE_ROOT
# RUN ./data/mnist/get_mnist.sh
# RUN ./examples/mnist/create_mnist.sh
# cd $CAFFE_ROOT && ./examples/mnist/train_lenet.sh

# ==============================================================
#  set workdir and user for jupyter notebook
# ==============================================================

# WORKDIR /home/$NB_USER/work/
# USER $NB_USER
