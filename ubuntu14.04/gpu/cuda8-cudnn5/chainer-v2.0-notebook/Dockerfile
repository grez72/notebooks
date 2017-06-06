# grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-chainer-v2.0-notebook
#
# Chainer jupyter notebook with gpu suppport (nvidia CUDA cuDNN).
#
# sudo nvidia-docker build -t grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-chainer-v2.0-notebook https://github.com/grez72/notebooks.git#master:ubuntu14.04/gpu/cuda8-cudnn5/chainer-v2.0-notebook
# nvidia-docker run -it --rm -p 8888:8888 grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-chainer-v2.0-notebook

FROM grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-base-notebook

MAINTAINER George Alvarez <gre72@gmail.com, alvarez@wjh.harvard.edu>

ARG CHAINER_VERSION=v2.0.0

USER root

WORKDIR $HOME

# ==============================================================
#  INSTALL CHAINER
# ==============================================================

RUN pip install chainer==${CHAINER_VERSION}
RUN pip install cupy

RUN pip2 install chainer==${CHAINER_VERSION}
RUN pip2 install cupy

# ==============================================================
#  downgrade anaconda's hdf5 to resolve error
# ==============================================================

# conda install -c anaconda hdf5=1.8.17
RUN conda remove hdf5 -y && conda install hdf5=1.8.17 -y
RUN /bin/bash -c "source activate python2 && conda remove hdf5 -y && conda install hdf5=1.8.17 -y"

# ==============================================================
#  set workdir and user for jupyter notebook
# ==============================================================

WORKDIR /home/$NB_USER/work/
USER $NB_USER
