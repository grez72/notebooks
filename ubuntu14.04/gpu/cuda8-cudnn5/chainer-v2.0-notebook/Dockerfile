# grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-chainer-v2.0-notebook
#
# Theano jupyter notebook with gpu suppport (nvidia CUDA cuDNN).
#
# sudo nvidia-docker build -t grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-theano-v0.9-notebook https://github.com/grez72/notebooks.git#master:ubuntu14.04/gpu/cuda8-cudnn5/theano-v0.9-notebook
# nvidia-docker run -it --rm -p 8888:8888 grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-theano-v0.9-notebook

FROM grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-base-notebook

MAINTAINER George Alvarez <gre72@gmail.com, alvarez@wjh.harvard.edu>

# using theano 0.9.0rc1 because of this Lasagne error: "ImportError: cannot import name downsample"
# ARG THEANO_VERSION=rel-0.9.0
ARG THEANO_VERSION=rel-0.9.0rc1
ARG KERAS_VERSION=2.0.4
ARG LASAGNE_VERSION=0.2.dev1

USER root

WORKDIR $HOME
