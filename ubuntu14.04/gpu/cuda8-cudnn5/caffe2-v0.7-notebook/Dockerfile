# grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-caffe2-v0.7-notebook
#
# Caffe2 jupyter notebook with gpu suppport (nvidia CUDA cuDNN).
#
# sudo nvidia-docker build -t grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-caffe2-v0.7-notebook https://github.com/grez72/notebooks.git#master:ubuntu14.04/gpu/cuda8-cudnn5/caffe2-v0.7-notebook
# nvidia-docker run -it --rm -p 8888:8888 grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-caffe2-v0.7-notebook

FROM grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-base-notebook

MAINTAINER George Alvarez <gre72@gmail.com, alvarez@wjh.harvard.edu>

ARG CAFFE2_VERSION=v0.7.0

# ==============================================================
#  INSTALL Caffe2
# ==============================================================

USER root
WORKDIR $HOME

# Required Dependencies
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      libgoogle-glog-dev \
      libprotobuf-dev \
      protobuf-compiler \
      python-dev \
      python-pip
RUN pip install --upgrade pip
RUN pip install numpy protobuf
RUN pip2 install numpy protobuf

# GPU Support CUDA 8.0
# RUN apt-get update && sudo apt-get install wget -y --no-install-recommends
# RUN wget "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_8.0.61-1_amd64.deb"
# RUN dpkg -i cuda-repo-ubuntu1404_8.0.61-1_amd64.deb
# RUN apt-get update
# RUN apt-get install cuda

# cuDNN 5.1
# RUN CUDNN_URL="http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz"
# RUN wget ${CUDNN_URL}
# RUN tar -xzf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local
# RUN rm cudnn-8.0-linux-x64-v5.1.tgz && sudo ldconfig

# Optional Dependeincies
RUN apt-get install -y --no-install-recommends \
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
      python-pydot
RUN pip install \
      flask \
      future \
      graphviz \
      hypothesis \
      jupyter \
      matplotlib \
      pydot python-nvd3 \
      pyyaml \
      requests \
      scikit-image \
      scipy \
      setuptools \
      six \
      tornado

RUN pip2 install \
      flask \
      future \
      graphviz \
      hypothesis \
      jupyter \
      matplotlib \
      pydot python-nvd3 \
      pyyaml \
      requests \
      scikit-image \
      scipy \
      setuptools \
      six \
      tornado

# Clone and build
RUN git clone --recursive https://github.com/caffe2/caffe2.git
WORKDIR $HOME/caffe2
RUN git checkout tags/${CAFFE2_VERSION}
RUN make && cd build && make install
RUN python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"

# Test your gpu
# RUN python -m caffe2.python.operator_test.relu_op_test

# Make sure these ENV variables are set
# echo $PYTHONPATH
# export PYTHONPATH=/usr/local:$PYTHONPATH
# export PYTHONPATH=$PYTHONPATH:/home/ubuntu/caffe2/build
# echo $LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# ==============================================================
#  downgrade anaconda's hdf5 to resolve error
# ==============================================================

# conda install -c anaconda hdf5=1.8.17
# RUN conda remove hdf5 -y && conda install hdf5=1.8.17 -y
# RUN /bin/bash -c "source activate python2 && conda remove hdf5 -y && conda install hdf5=1.8.17 -y"

# ==============================================================
#  set workdir and user for jupyter notebook
# ==============================================================

# WORKDIR /home/$NB_USER/work/
# USER $NB_USER
