# grez72/deep-notebooks:ubuntu16.04-cuda10.0-cudnn7-pytorch-v1.0-notebook
#
# Pytorch jupyter notebook with gpu suppport (nvidia CUDA10.0 cuDNN7).
#
# docker build -t grez72/deep-notebooks:ubuntu16.04-cuda10.0-cudnn7-pytorch-v1.0-notebook https://github.com/grez72/notebooks.git#master:ubuntu16.04/gpu/cuda10.0-cudnn7/pytorch-v1.0-notebook
# docker run -it --rm --user root grez72/deep-notebooks:ubuntu16.04-cuda10.0-cudnn7-pytorch-v1.0-notebook bash
# docker run -it --rm --user root grez72/deep-notebooks:ubuntu16.04-cuda10.0-cudnn7-pytorch-v1.0-notebook nvidia-smi
# docker run -it --rm --shm-size=2G -p 8889:8888 grez72/deep-notebooks:ubuntu16.04-cuda10.0-cudnn7-pytorch-v1.0-notebook

FROM grez72/base-notebook:ubuntu16.04-cuda10.0-cudnn7

LABEL maintainer="George Alvarez <alvarez@wjh.harvard.com>"

ARG PYTORCH_VERSION=1.0
ARG CUDA_VERSION=cuda100

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
  && conda install pytorch torchvision cuda100 -c pytorch \
  && conda install -c fastai fastprogress fastai \
  && conda clean -tipsy

# RUN conda update -y conda pytorch torchvision
RUN /bin/bash -c "source activate python2 \
  && pip install --upgrade pip \
  && conda install -y mkl -c intel \
  && conda install -c conda-forge mock \
  && conda install pytorch torchvision cuda100 -c pytorch \
  && conda clean -tipsy"

# python2, couldn't update numpy for some reason:
#  && pip install numpy --upgrade \

# ==============================================================
#  MAKE SURE WE'RE USING OPTIMIZED PILLOW
# ==============================================================

# RUN conda update -n base conda -y \
#   && pip install --upgrade pip \
#   && conda uninstall -y --force jpeg libtiff \
#   && conda install -y -c conda-forge libjpeg-turbo \
#   && CC="cc -mavx2" pip install --no-cache-dir -U --force-reinstall --no-binary :all: --compile pillow-simd

# ==============================================================
#  INSTALL NVIDIA HELPERS
# ==============================================================

WORKDIR /home/$NB_USER
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali \
  && git clone https://github.com/NVIDIA/apex.git \
  && cd /home/$NB_USER/apex \
  && python setup.py install --cuda_ext --cpp_ext

# ==============================================================
#  MORE HELPERS
# ==============================================================

RUN pip install torchsummary torchsummaryX
COPY credential-helper.sh /home/jovyan/credential-helper.sh
RUN git config credential.helper "/bin/bash /home/jovyan/credential-helper.sh"

# ==============================================================
#  jupyterlab-git integration?
# ==============================================================

# https://github.com/jupyterlab/jupyterlab-git
# RUN jupyter labextension install @jupyterlab/git \
#   && pip install jupyterlab-git \
#   && jupyter serverextension enable --py jupyterlab_git

# ==============================================================
#  set workdir and user for jupyter notebook use
# ==============================================================

WORKDIR /home/$NB_USER/work/
USER $NB_USER
