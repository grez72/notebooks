# grez72/deep-notebooks:ubuntu18.04-cuda10.1-cudnn7-datascience-notebook
#
# Pytorch jupyter notebook with gpu suppport (nvidia CUDA10.1 cuDNN7).
#
# docker build -t grez72/deep-notebooks:ubuntu18.04-cuda10.1-cudnn7-datascience-notebook https://github.com/grez72/notebooks.git#master:ubuntu18.04/cuda10.1-cudnn7/datascience-notebook
# docker run -it --rm --user root grez72/deep-notebooks:ubuntu18.04-cuda10.1-cudnn7-datascience-notebook bash
# docker run -it --rm --user root grez72/deep-notebooks:ubuntu18.04-cuda10.1-cudnn7-datascience-notebook nvidia-smi
# docker run -it --rm --shm-size=2G -p 8889:8888 grez72/deep-notebooks:ubuntu18.04-cuda10.1-cudnn7-datascience-notebook

FROM grez72/deep-notebooks:ubuntu18.04-cuda10.1-cudnn7-pytorch-v1.3-notebook

LABEL maintainer="George Alvarez <alvarez@wjh.harvard.com>"

USER root
WORKDIR $HOME

# ==============================================================
#  install rapids and h204gpu
# ==============================================================

RUN apt-get update && apt-get install -yq libopenblas-dev pbzip2

RUN pip install https://s3.amazonaws.com/h2o-release/h2o4gpu/releases/stable/ai/h2o/h2o4gpu/0.3-cuda10/h2o4gpu-0.3.2-cp36-cp36m-linux_x86_64.whl

RUN conda install -y -c rapidsai -c nvidia -c conda-forge \
    -c defaults rapids=0.10 python=3.6
    
# ==============================================================
#  set workdir and user for jupyter notebook use
# ==============================================================

WORKDIR /home/$NB_USER/work/
USER $NB_USER
