# https://gitlab.com/nvidia/cuda/blob/ubuntu14.04/8.0/runtime/cudnn5/Dockerfile
# docker build -t grez72/tensorflow-gpu-notebook https://github.com/grez72/notebooks.git#master:tensorflow-gpu-notebook
# FROM nvidia/cuda:8.0-runtime-ubuntu14.04
# FROM jupyterhub/singleuser

FROM jupyter/scipy-notebook

MAINTAINER George Alvarez <alvarez@wjh.harvard.edu.com>

# CUDA for GPU Support
# USER root
# RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/7fa2af80.pub
# RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

ENV CUDNN_VERSION 5.1.10
# LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

# RUN apt-get update && apt-get install -y --no-install-recommends \
#            libcudnn5=$CUDNN_VERSION-1+cuda8.0 && \
#    rm -rf /var/lib/apt/lists/*

# DRIVERS
# RUN apt-get update && apt-get install -y software-properties-common python-software-properties python3-software-properties
# RUN add-apt-repository ppa:graphics-drivers/ppa
# RUN apt-get update && apt-get install -y nvidia-375.66

USER $NB_USER

# Install Python 3 Tensorflow
RUN conda install --quiet --yes 'tensorflow-gpu=1.1.0'
# RUN conda install --quiet --yes 'tensorflow=1.1.0'

# Install Python 2 Tensorflow
RUN conda install --quiet --yes -n python2 'tensorflow-gpu=1.1.0'
# RUN conda install --quiet --yes -n python2 'tensorflow=1.1.0'
