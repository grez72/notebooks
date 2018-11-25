# grez72/deep-notebooks:ubuntu16.04-cuda9.2-cudnn7-pytorch-v0.4-notebook
#
# Pytorch jupyter notebook with gpu suppport (nvidia CUDA9.2 cuDNN7).
#
# docker build -t grez72/deep-notebooks:ubuntu16.04-cuda9.2-cudnn7-pytorch-v0.4-notebook https://github.com/grez72/notebooks.git#master:ubuntu16.04/gpu/cuda9.2-cudnn7/pytorch-v0.4-notebook
# docker run -it --rm --user root grez72/deep-notebooks:ubuntu16.04-cuda9.2-cudnn7-pytorch-v0.4-notebook bash
# docker run -it --rm --shm-size=2G -p 8889:8888 grez72/deep-notebooks:ubuntu16.04-cuda9.2-cudnn7-pytorch-v0.4-notebook

FROM grez72/base-notebook:ubuntu16.04-cuda9.2-cudnn7

LABEL maintainer="George Alvarez <alvarez@wjh.harvard.com>"

ARG PYTORCH_VERSION=0.4.0
ARG CUDA_VERSION=cuda92

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
  && conda install pytorch torchvision cuda92 -c pytorch \
  && conda clean -tipsy

# RUN conda update -y conda pytorch torchvision
RUN /bin/bash -c "source activate python2 \
  && pip install --upgrade pip \
  && conda install -y mkl -c intel \
  && conda install -c conda-forge mock \
  && conda install pytorch torchvision cuda92 -c pytorch \
  && conda clean -tipsy"

# python2, couldn't update numpy for some reason:
#  && pip install numpy --upgrade \

# ==============================================================
#  INSTALL FASTAI ENVIRONMENTS
# ==============================================================

USER $NB_USER

RUN conda update conda \
    && conda config --add channels intel \
    && conda create -n fastai intelpython3_core python=3 \
    && conda create -n fastai_v0.7 intelpython3_core python=3

RUN /bin/bash -c "source activate fastai \
    && conda install pip ipykernel \
    && python -m ipykernel install --user --name python3 --display-name 'FastAI'" \
    /bin/bash -c "source activate fastai_v0.7 \
    && conda install pip ipykernel \
    && python -m ipykernel install --user --name python3 --display-name 'FastAIv0.7'"

# ==============================================================
#  Install FAST AI
# ==============================================================

# RUN /bin/bash -c "source activate fastai \
#     && conda install -c pytorch pytorch-nightly cuda92 \
#     && conda install -c fastai torchvision-nightly \
#     && conda install -c fastai fastai \
#     && pip uninstall -y pillow --user \
#     && conda uninstall --force jpeg libtiff -y \
#     && conda install -c conda-forge libjpeg-turbo \
#     && CC='cc -mavx2' pip install --no-cache-dir -U --force-reinstall pillow-simd" \
#     /bin/bash -c "source activate fastai_v0.7 \
#     && conda install -y scipy scikit-learn bokeh pandas patsy seaborn scikit-image \
#     && conda install -y -c conda-forge mock \
#     && conda install -y mkl -c intel \
#     && pip install fastai==0.7.0 --user \
#     && pip install torchtext==0.2.3 --user \
#     && pip uninstall -y pillow --user \
#     && conda uninstall --force jpeg libtiff -y \
#     && conda install -c conda-forge libjpeg-turbo \
#     && CC='cc -mavx2' pip install --no-cache-dir -U --force-reinstall pillow-simd"

# ==============================================================
#  set workdir and user for jupyter notebook use
# ==============================================================

WORKDIR /home/$NB_USER/work/
USER $NB_USER
