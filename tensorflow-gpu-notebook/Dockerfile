# https://gitlab.com/nvidia/cuda/blob/ubuntu14.04/8.0/runtime/cudnn5/Dockerfile
# docker build -t grez72/tensorflow-gpu-notebook https://github.com/grez72/notebooks.git#master:tensorflow-gpu-notebook
# FROM nvidia/cuda:8.0-runtime-ubuntu14.04
# FROM jupyterhub/singleuser

FROM jupyter/scipy-notebook

MAINTAINER George Alvarez <alvarez@wjh.harvard.edu.com>

# CUDA for GPU Support

USER root

ENV CUDA_RUN https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_linux-run

RUN apt-get update && apt-get install -q -y \
  wget \
  module-init-tools \
  build-essential

RUN cd /opt && \
  wget $CUDA_RUN && \
  chmod +x cuda_8.0.44_linux-run && \
  mkdir nvidia_installers && \
  ./cuda_8.0.44_linux-run -extract=`pwd`/nvidia_installers && \
  cd nvidia_installers && \
  ./NVIDIA-Linux-x86_64-367.48.run -s -N --no-kernel-module

RUN cd /opt/nvidia_installers && \
  ./cuda-linux64-rel-8.0.44-21122537.run -noprompt

# Ensure the CUDA libs and binaries are in the correct environment variables
ENV LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64
ENV PATH=$PATH:/usr/local/cuda-8.0/bin

RUN cd /opt/nvidia_installers &&\
    ./cuda-samples-linux-8.0.44-21122537.run -noprompt -cudaprefix=/usr/local/cuda-8.0 &&\
    cd /usr/local/cuda/samples/1_Utilities/deviceQuery &&\
    make

WORKDIR /usr/local/cuda/samples/1_Utilities/deviceQuery

# Ensure the CUDA libs and binaries are in the correct environment variables
# ENV LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-6.5/lib64
# ENV PATH=$PATH:/usr/local/cuda-6.5/bin

# RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/7fa2af80.pub
# RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

# ENV CUDNN_VERSION 5.1.10
# LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

# RUN apt-get update && apt-get install -y --no-install-recommends \
#            libcudnn5=$CUDNN_VERSION-1+cuda8.0 && \
#    rm -rf /var/lib/apt/lists/*

# DRIVERS
# RUN apt-get update && apt-get install -y software-properties-common python-software-properties python3-software-properties
# RUN add-apt-repository ppa:graphics-drivers/ppa
# RUN apt-get install -y nvidia-375

USER $NB_USER

# Install Python 3 Tensorflow
RUN conda install --quiet --yes 'tensorflow-gpu=1.1.0'

# Install Python 2 Tensorflow
RUN conda install --quiet --yes -n python2 'tensorflow-gpu=1.1.0'
