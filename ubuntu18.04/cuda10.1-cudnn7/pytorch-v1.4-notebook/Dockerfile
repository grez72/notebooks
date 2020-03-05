# grez72/deep-notebooks:ubuntu18.04-cuda10.1-cudnn7-pytorch-v1.4-notebook
#
# Pytorch jupyter notebook with gpu suppport (nvidia CUDA10.1 cuDNN7).
#
# docker build -t grez72/deep-notebooks:ubuntu18.04-cuda10.1-cudnn7-pytorch-v1.4-notebook https://github.com/grez72/notebooks.git#master:ubuntu18.04/cuda10.1-cudnn7/pytorch-v1.4-notebook
# docker run -it --rm --user root grez72/deep-notebooks:ubuntu18.04-cuda10.1-cudnn7-pytorch-v1.4-notebook bash
# docker run -it --rm --user root grez72/deep-notebooks:ubuntu18.04-cuda10.1-cudnn7-pytorch-v1.4-notebook nvidia-smi
# docker run -it --rm --shm-size=2G -p 8889:8888 grez72/deep-notebooks:ubuntu18.04-cuda10.1-cudnn7-pytorch-v1.4-notebook

FROM grez72/base-notebook:ubuntu18.04-cuda10.1-cudnn7-1.4

LABEL maintainer="George Alvarez <alvarez@wjh.harvard.com>"

ENV DEBIAN_FRONTEND noninteractive
ARG PYTORCH_VERSION=1.4
ARG CUDA_VERSION=10.1
ENV GLIB_VERSION=2.51.0

USER root
WORKDIR $HOME

# ==============================================================
#  preliminaries
# ==============================================================

RUN apt-get update && apt-get install -y \
	libpng-dev libjpeg-dev python3-opencv ca-certificates \
	python3-dev build-essential pkg-config git curl wget automake libtool && \
  rm -rf /var/lib/apt/lists/*

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py

# ==============================================================
#  INSTALL PYTORCH
# ==============================================================

# mamba install glib=2.51.0

RUN /bin/bash -c "source activate base \
  && pip install numpy --upgrade \
  && mamba update conda \
  && mamba install -c pytorch pytorch=$PYTORCH_VERSION torchvision cudatoolkit=10.1 glib=$GLIB_VERSION \
  && mamba install -c fastai -c pytorch fastprogress \
  && mamba install -c conda-forge mock \
  && mamba install -y mkl -c intel \
  && mamba clean --all -f -y"

# ==============================================================
#  MAKE SURE WE'RE USING OPTIMIZED PILLOW
# ==============================================================

# python -c "from PIL import features; print(features.check_feature('libjpeg_turbo'))"
# python -c "import PIL; print(PIL.__version__)"

RUN /bin/bash -c "source activate base \
  && mamba update -n base conda -y \
  && pip install --upgrade pip \
  && conda uninstall -y --force pillow pil jpeg libtiff libjpeg-turbo \
  && pip uninstall -y pillow pil jpeg libtiff libjpeg-turbo \
  && mamba install -yc conda-forge libjpeg-turbo \
  && CC='cc -mavx2' pip install --no-cache-dir -U --force-reinstall --no-binary :all: --compile pillow-simd \  
  && mamba install -y jpeg libtiff \
  && mamba clean --all -f -y"

# ==============================================================
#  INSTALL NVIDIA HELPERS
# ==============================================================

WORKDIR /home/$NB_USER

RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali

# ==============================================================
#  Install Detectron
# ==============================================================

ENV FORCE_CUDA="1"
RUN pip install 'git+https://github.com/facebookresearch/fvcore' && \
    pip install cython && \
    pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' && \
    cd /home/$NB_USER && \
    git clone https://github.com/facebookresearch/detectron2.git && \
    cd detectron2 && \
    pip install -e .

# # ==============================================================
# #  MORE HELPERS
# # ==============================================================

# RUN pip install torchlars \
#   torchsummary \
#   torchsummaryX \
#   streamlit \
#   fire \
#   pretrainedmodels \ 
#   natsort \
#   && pip install -U git+git://github.com/PyTorchLightning/pytorch-lightning.git

# # ==============================================================
# #  set workdir and user for jupyter notebook use
# # ==============================================================

# WORKDIR /home/$NB_USER/work/
# USER $NB_USER
