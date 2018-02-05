# grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-chainer-v3.3-notebook
#
# chainer jupyter notebook with gpu suppport (nvidia CUDA cuDNN).
#
# sudo nvidia-docker build -t grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-chainer-v3.3-notebook https://github.com/grez72/notebooks.git#master:ubuntu16.04/gpu/cuda9.1-cudnn7/chainer-v3.3-notebook
# nvidia-docker run -it --rm --shm-size=2G -p 8888:8888 grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-chainer-v3.3-notebook

FROM grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-minimal-notebook

MAINTAINER George Alvarez <gre72@gmail.com, alvarez@wjh.harvard.edu>

ARG CHAINER_VERSION=4.0.0b3

USER root
WORKDIR $HOME

# ==============================================================
#  INSTALL CHAINER
# ==============================================================

RUN pip install cupy==4.0.0b3 chainer==${CHAINER_VERSION} \
  && /bin/bash -c "source activate python2 \
  && pip install cupy==4.0.0b3 chainer==${CHAINER_VERSION}"

# ==============================================================
#  set workdir and user for jupyter notebook use
# ==============================================================

WORKDIR /home/$NB_USER/work/
USER $NB_USER
