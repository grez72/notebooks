# scipy-plus-notebook
#
# A copy of scipy-notebook with additional packages installed:
#
# build it:
# docker build -t grez72/scipy-plus-notebook https://github.com/grez72/notebooks.git#master:scipy-plus-notebook
#
# run it:
# docker run -it --rm -p 8888:8888 grez72/scipy-plus-notebook
# docker run --user root -it --rm -p 8888:8888 grez72/scipy-plus-notebook

FROM jupyter/scipy-notebook
MAINTAINER George Alvarez <grez72@gmail.com, alvarez.wjh.harvard.edu>

#--------------------------------------
# Install Python 2
#--------------------------------------

USER jovyan

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
    pyzmq \
    && conda clean -yt

USER root

# Install Python 2 kernel spec globally to avoid permission problems when NB_UID
# switching at runtime.
RUN $CONDA_DIR/envs/python2/bin/python \
    $CONDA_DIR/envs/python2/bin/ipython \
    kernelspec install-self

#--------------------------------------
# Install R
#--------------------------------------

# USER root
#
# # R pre-requisites
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#     libxrender1 \
#     fonts-dejavu \
#     gfortran \
#     gcc && apt-get clean
#
# USER $NB_USER
#
# # R packages
# RUN conda config --add channels r
# RUN conda install --yes \
#     'r-base=3.2*' \
#     'r-irkernel=0.4*' \
#     'r-plyr=1.8*' \
#     'r-devtools=1.8*' \
#     'r-dplyr=0.4*' \
#     'r-ggplot2=1.0*' \
#     'r-tidyr=0.2*' \
#     'r-shiny=0.12*' \
#     'r-rmarkdown=0.7*' \
#     'r-forecast=5.8*' \
#     'r-stringr=0.6*' \
#     'r-rsqlite=1.0*' \
#     'r-reshape2=1.4*' \
#     'r-nycflights13=0.1*' \
#     'r-caret=6.0*' \
#     'r-rcurl=1.95*' \
#     'r-randomforest=4.6*' && conda clean -yt

#----------------------------------------
# Clear apt cache and other empty folders
#----------------------------------------

USER root
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /boot /media /mnt /srv && \
    chmod a+w /tmp

#--------------------------------------
# Setup for notebook use
#--------------------------------------

RUN ln -s /home/$NB_USER/work/data /data
WORKDIR /home/$NB_USER/work/
USER $NB_USER
