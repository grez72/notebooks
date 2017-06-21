# grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-pytorch-v0.1-notebook
#
# Pytorch jupyter notebook with gpu suppport (nvidia CUDA cuDNN).
#
# sudo nvidia-docker build -t grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-pytorch-v0.1-notebook https://github.com/grez72/notebooks.git#master:ubuntu14.04/gpu/cuda8-cudnn5/pytorch-v0.1-notebook
# nvidia-docker run -it --rm -p 8888:8888 grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-pytorch-v0.1-notebook

FROM grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-base-notebook

MAINTAINER George Alvarez <gre72@gmail.com, alvarez@wjh.harvard.edu>

ARG PYTORCH_VERSION=0.1.12

USER root

WORKDIR $HOME

# ==============================================================
#  INSTALL PYTORCH
# ==============================================================

RUN pip install http://download.pytorch.org/whl/cu80/torch-${PYTORCH_VERSION}.post2-cp35-cp35m-linux_x86_64.whl
RUN pip install torchvision

RUN pip2 install http://download.pytorch.org/whl/cu80/torch-${PYTORCH_VERSION}.post2-cp27-none-linux_x86_64.whl
RUN pip2 install torchvision

# ==============================================================
#  set workdir and user for jupyter notebook use
# ==============================================================

WORKDIR /home/$NB_USER/work/
USER $NB_USER
