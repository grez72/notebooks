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

ARG PYTORCH_VERSION=1.3
ARG CUDA_VERSION=10.1

USER root
WORKDIR $HOME

# ==============================================================
#  INSTALL PYTORCH
# ==============================================================

RUN pip install --upgrade pip \
  && pip install numpy --upgrade \
  && conda update conda \
  && conda install -c conda-forge mock \
  && conda install -y mkl -c intel \
  && conda install pytorch=$PYTORCH_VERSION torchvision cudatoolkit=$CUDA_VERSION -c pytorch \
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

# not installing APEX because current version not compiled with cuda 10.1
# RUN git clone https://github.com/NVIDIA/apex.git \
#   && cd /home/$NB_USER/apex \
#   && python setup.py install --cuda_ext --cpp_ext

# ==============================================================
#  Install Detectron
# ==============================================================

RUN pip install 'git+https://github.com/facebookresearch/fvcore' && \
    pip install cython && \
    pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' && \
    cd /tmp && \
    git clone git@github.com:facebookresearch/detectron2.git && \
    cd detectron2 && \
    python setup.py build develop

# ==============================================================
#  MORE HELPERS
# ==============================================================

RUN pip install torchsummary torchsummaryX streamlit fire pretrainedmodels

# ==============================================================
#  set workdir and user for jupyter notebook use
# ==============================================================

WORKDIR /home/$NB_USER/work/
USER $NB_USER
