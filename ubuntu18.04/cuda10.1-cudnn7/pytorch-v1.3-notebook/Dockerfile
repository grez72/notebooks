# grez72/deep-notebooks:ubuntu18.04-cuda10.1-cudnn7-pytorch-v1.3-notebook
#
# Pytorch jupyter notebook with gpu suppport (nvidia CUDA10.1 cuDNN7).
#
# docker build -t grez72/deep-notebooks:ubuntu18.04-cuda10.1-cudnn7-pytorch-v1.3-notebook https://github.com/grez72/notebooks.git#master:ubuntu18.04/cuda10.1-cudnn7/pytorch-v1.3-notebook
# docker run -it --rm --user root grez72/deep-notebooks:ubuntu18.04-cuda10.1-cudnn7-pytorch-v1.3-notebook bash
# docker run -it --rm --user root grez72/deep-notebooks:ubuntu18.04-cuda10.1-cudnn7-pytorch-v1.3-notebook nvidia-smi
# docker run -it --rm --shm-size=2G -p 8889:8888 grez72/deep-notebooks:ubuntu18.04-cuda10.1-cudnn7-pytorch-v1.3-notebook

FROM grez72/base-notebook:ubuntu18.04-cuda10.1-cudnn7

LABEL maintainer="George Alvarez <alvarez@wjh.harvard.com>"

ENV DEBIAN_FRONTEND noninteractive
ARG PYTORCH_VERSION=1.3
ARG CUDA_VERSION=10.1

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

RUN pip install torch torchvision
RUN pip install numpy --upgrade \
  && conda update conda \
  && conda install -c conda-forge mock \
  && conda install -y mkl -c intel \
  && conda install -c fastai fastprogress fastai \
  && conda clean -tipsy

# ==============================================================
#  MAKE SURE WE'RE USING OPTIMIZED PILLOW
# ==============================================================

RUN conda update -n base conda -y \
  && pip install --upgrade pip \
  && conda uninstall -y --force jpeg libtiff \
  && conda install -y -c conda-forge libjpeg-turbo \
  && CC="cc -mavx2" pip install --no-cache-dir -U --force-reinstall --no-binary :all: --compile pillow-simd

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

# ==============================================================
#  MORE HELPERS
# ==============================================================

RUN pip install torchsummary torchsummaryX streamlit fire pretrainedmodels

# ==============================================================
#  set workdir and user for jupyter notebook use
# ==============================================================

WORKDIR /home/$NB_USER/work/
USER $NB_USER
