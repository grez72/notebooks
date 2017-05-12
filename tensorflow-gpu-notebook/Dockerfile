# docker build -t grez72/tensorflow-gpu-notebook https://github.com/grez72/notebooks.git#master:tensorflow-gpu-notebook
FROM jupyter/scipy-notebook

MAINTAINER George Alvarez <alvarez@wjh.harvard.edu.com>

USER $NB_USER

# Install Python 3 Tensorflow
RUN conda install --quiet --yes 'tensorflow-gpu=1.1.0'

# Install Python 2 Tensorflow
RUN conda install --quiet --yes -n python2 'tensorflow-gpu=1.1.0'

# CUDA for GPU Support
# RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/" > /etc/apt/sources.list.d/nvidia-ml.list
RUN echo 'deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/' | sudo tee /etc/apt/sources.list.d/nvidia-ml.list > /dev/null


# ENV CUDNN_VERSION 5.1.10
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
            libcudnn5=$CUDNN_VERSION-1+cuda8.0 && \
    rm -rf /var/lib/apt/lists/*
