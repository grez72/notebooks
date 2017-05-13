# https://gitlab.com/nvidia/cuda/blob/ubuntu14.04/8.0/runtime/cudnn5/Dockerfile
# docker build -t grez72/tensorflow-gpu-notebook https://github.com/grez72/notebooks.git#master:tensorflow-gpu-notebook
# FROM nvidia/cuda:8.0-runtime-ubuntu14.04
# FROM jupyterhub/singleuser

FROM jupyter/scipy-notebook

MAINTAINER George Alvarez <alvarez@wjh.harvard.edu.com>

# CUDA for GPU Support
USER root

RUN apt-get update && apt-get install -y build-essential
RUN apt-get --purge remove -y nvidia*

ADD ./Downloads/nvidia_installers /tmp/nvidia
RUN /tmp/nvidia/NVIDIA-Linux-x86_64-331.62.run -s -N --no-kernel-module
RUN rm -rf /tmp/selfgz7
RUN /tmp/nvidia/cuda-linux64-rel-6.0.37-18176142.run -noprompt
RUN /tmp/nvidia/cuda-samples-linux-6.0.37-18176142.run -noprompt -cudaprefix=/usr/local/cuda-6.0
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
RUN touch /etc/ld.so.conf.d/cuda.conf
RUN rm -rf /temp/*

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
