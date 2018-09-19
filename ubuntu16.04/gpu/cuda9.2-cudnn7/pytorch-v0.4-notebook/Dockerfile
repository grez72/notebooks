# grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.2-cudnn7-pytorch-v0.4-notebook
#
# Pytorch jupyter notebook with gpu suppport (nvidia CUDA cuDNN).
#
# sudo nvidia-docker build -t grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.2-cudnn7-pytorch-v0.4-notebook https://github.com/grez72/notebooks.git#master:ubuntu16.04/gpu/cuda9.2-cudnn7/pytorch-v0.4-notebook
# nvidia-docker run -it --rm --shm-size=2G -p 8888:8888 grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.2-cudnn7-pytorch-v0.4-notebook

FROM grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.2-cudnn7-minimal-notebook

MAINTAINER George Alvarez <gre72@gmail.com, alvarez@wjh.harvard.edu>

ARG PYTORCH_VERSION=0.4.0
ARG CUDA_VERSION=cuda90

USER root
WORKDIR $HOME

# ==============================================================
#  INSTALL PYTORCH
# ==============================================================

RUN pip install --upgrade pip \
  && pip install numpy --upgrade \
  && conda update conda \
  && conda install -y mkl -c intel \
  && conda install pytorch torchvision cuda90 -c pytorch

# RUN conda update -y conda pytorch torchvision
RUN /bin/bash -c "source activate python2 \
  && pip install --upgrade pip \
  && pip install numpy --upgrade \
  && conda install -y mkl -c intel \
  && conda install pytorch torchvision cuda90 -c pytorch"

# ==============================================================
#  set workdir and user for jupyter notebook use
# ==============================================================

WORKDIR /home/$NB_USER/work/
USER $NB_USER
