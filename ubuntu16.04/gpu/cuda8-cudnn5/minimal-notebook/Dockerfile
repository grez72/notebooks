# grez72/deep-notebooks:ubuntu16.04-gpu-cuda8-cudnn5-minimal-notebook
#
# Dervided from scipy notebook, with cuda8/cudnn5 support, and python2
# resinstated.
#
# build it:
# nvidia-docker build -t grez72/deep-notebooks:ubuntu16.04-gpu-cuda8-cudnn5-minimal-notebook https://github.com/grez72/notebooks.git#master:ubuntu16.04/gpu/cuda8-cudnn5/minimal-notebook
#
# run it:
# nvidia-docker run -it --rm -p 8888:8888 grez72/deep-notebooks:ubuntu16.04-gpu-cuda8-cudnn5-base-notebook

FROM grez72/deep-notebooks:ubuntu16.04-gpu-cuda8-cudnn5-base-notebook
MAINTAINER George Alvarez <grez72@gmail.com, alvarez.wjh.harvard.edu>

#--------------------------------------
# Install Python 2
#--------------------------------------

USER jovyan

# make these to avoid permission error
# RUN mkdir /home/$NB_USER/.jupyter && \
#     mkdir /home/$NB_USER/.local

# Install Python 2 packages
RUN conda create -p $CONDA_DIR/envs/python2 python=2.7 \
    'ipython=4.0*' \
    'ipywidgets=4.0*' \
    'pandas=0.16*' \
    'matplotlib=1.4*' \
    'scipy=0.15*' \
    'seaborn=0.6*' \
    'scikit-learn=0.16*' \
    'scikit-image=0.11*' \
    'sympy=0.7*' \
    'cython=0.22*' \
    'patsy=0.3*' \
    'statsmodels=0.6*' \
    'cloudpickle=0.1*' \
    'dill=0.2*' \
    'numba=0.20*' \
    'bokeh=0.9*' \
    'libgfortran=1.0.0' \
    pyzmq \
    && conda clean -yt

USER root

RUN /bin/bash -c "source activate python2 && ipython kernel install"

#--------------------------------------
# Install OpenCV with CUDA Support
#--------------------------------------

#--------------------------------------
# Install Additional Packages
#--------------------------------------

# plotly, cufflinks


#--------------------------------------
# Setup for notebook use
#--------------------------------------

WORKDIR /home/$NB_USER/work/
USER $NB_USER
