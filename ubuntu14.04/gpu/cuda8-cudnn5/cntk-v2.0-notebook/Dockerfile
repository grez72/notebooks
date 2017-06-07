# grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-cntk-v2.0-notebook
#
# Microsoft Cognitive Toolkit jupyter notebook with gpu suppport (nvidia CUDA cuDNN).
#
# sudo nvidia-docker build -t grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-cntk-v2.0-notebook https://github.com/grez72/notebooks.git#master:ubuntu14.04/gpu/cuda8-cudnn5/cntk-v2.0-notebook
# nvidia-docker run -it --rm -p 8888:8888 grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-cntk-v2.0-notebook

FROM grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-base-notebook

MAINTAINER George Alvarez <gre72@gmail.com, alvarez@wjh.harvard.edu>

ARG CNTK_VERSION=v0.7.0

# ==============================================================
#  INSTALL CNTK
# ==============================================================

USER root
WORKDIR $HOME
