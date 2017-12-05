# grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-pytorch-v0.1-notebook
#
# Pytorch jupyter notebook with gpu suppport (nvidia CUDA cuDNN).
#
# sudo nvidia-docker build -t grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-pytorch-v0.3-notebook https://github.com/grez72/notebooks.git#master:ubuntu14.04/gpu/cuda8-cudnn5/pytorch-v0.3-notebook
# nvidia-docker run -it --rm --shm-size=512m -p 8888:8888 grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-pytorch-v0.3-notebook

FROM grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-base-notebook

MAINTAINER George Alvarez <gre72@gmail.com, alvarez@wjh.harvard.edu>

ARG PYTORCH_VERSION=0.3.0

USER root

WORKDIR $HOME

# ==============================================================
#  INSTALL PYTORCH
# ==============================================================

RUN pip install --upgrade pip
RUN pip install numpy --upgrade
RUN conda install -y pytorch=${PYTORCH_VERSION} torchvision cuda80 -c pytorch
# RUN conda update -y conda pytorch torchvision

RUN /bin/bash -c "source activate python2 && pip install --upgrade pip"
RUN /bin/bash -c "source activate python2 && pip install numpy --upgrade"
RUN /bin/bash -c "source activate python2 && conda install -y pytorch=${PYTORCH_VERSION} torchvision cuda80 -c pytorch"
# RUN /bin/bash -c "source activate python2 && conda update -y conda pytorch torchvision"

# RUN pip install --upgrade pip
# RUN pip install http://download.pytorch.org/whl/cu80/torch-${PYTORCH_VERSION}.post3-cp35-cp35m-manylinux1_x86_64.whl
# RUN pip install torchvision
#
# RUN pip2 install --upgrade pip
# RUN pip2 install http://download.pytorch.org/whl/cu80/torch-${PYTORCH_VERSION}.post3-cp27-cp27mu-manylinux1_x86_64.whl
# RUN pip2 install torchvision

# ==============================================================
#  set workdir and user for jupyter notebook use
# ==============================================================

WORKDIR /home/$NB_USER/work/
USER $NB_USER
