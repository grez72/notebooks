# grez72/deep-notebooks:ubuntu16.04-cuda9.2-cudnn7-fastai-v0.7.0-notebook
#
# Pytorch jupyter notebook with gpu suppport (nvidia CUDA9.2 cuDNN7).
#
# sudo nvidia-docker build -t grez72/deep-notebooks:ubuntu16.04-cuda9.2-cudnn7-fastai-v0.7.0-notebook https://github.com/grez72/notebooks.git#master:ubuntu16.04/gpu/cuda9.2-cudnn7/fastai-v0.7.0-notebook
# nvidia-docker run -it --rm --shm-size=2G -p 8888:8888 grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.2-cudnn7-fastai-v0.7.0

FROM grez72/base-notebook:ubuntu16.04-cuda9.2-cudnn7

LABEL maintainer="George Alvarez <alvarez@wjh.harvard.com>"

ARG PYTORCH_VERSION=pytorch-nightly
ARG CUDA_VERSION=cuda92

USER root
WORKDIR $HOME

# ==============================================================
#  INSTALL PYTORCH
# ==============================================================

RUN pip install --upgrade pip \
  && pip install numpy --upgrade \
  && conda update conda \
  && conda install -y scipy scikit-learn bokeh pandas patsy seaborn scikit-image \
  && conda install -y -c conda-forge mock \
  && conda install -y mkl -c intel \
  && conda clean -tipsy \
  && pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cu92/torch_nightly.html \
  && pip install fastai==0.7.0

# RUN conda update -y conda pytorch torchvision
RUN /bin/bash -c "source activate python2 \
  && pip install --upgrade pip \
  && conda install -y scipy scikit-learn bokeh pandas patsy seaborn scikit-image \
  && conda install -y mkl -c intel \
  && conda install -y -c conda-forge mock \
  && conda install -y -c pytorch pytorch-nightly cuda92 \
  && conda install -y -c fastai torchvision-nightly fastprogress \
  && conda clean -tipsy"

# python2, couldn't update numpy for some reason:
#  && pip install numpy --upgrade \

# ==============================================================
#  INSTALL FASTAI
# ==============================================================

# WORKDIR $HOME
#
# RUN git clone https://github.com/fastai/fastai \
#   && cd fastai \
#   && conda env update \
#   && pip install . \
#   && wget -O $HOME/fastai/weights.tgz http://files.fast.ai/models/weights.tgz \
#   && tar xvfz $HOME/fastai/weights.tgz -C fastai

# ==============================================================
#  set workdir and user for jupyter notebook use
# ==============================================================

WORKDIR /home/$NB_USER/work/
USER $NB_USER
