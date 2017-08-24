# scipy-plus-notebook
#
# A copy of scipy-notebook with additional packages installed:
#
# build it:
# docker build -t grez72/scipy-plus-notebook https://github.com/grez72/notebooks.git#master:scipy-plus-notebook
#
# run it:
# docker run -it --rm -v /Applications/MATLAB_R2017a.app/bin/matlab:/bin/matlab -p 8888:8888 grez72/scipy-plus-notebook
# docker run --user root -it --rm -v /Applications/MATLAB_R2017a.app/bin/matlab:/bin/matlab -p 8888:8888 grez72/scipy-plus-notebook

FROM jupyter/scipy-notebook
MAINTAINER George Alvarez <grez72@gmail.com, alvarez.wjh.harvard.edu>

#--------------------------------------
# Install Python 3 extras
#--------------------------------------

# plotly, cufflinks

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
# Install R
#--------------------------------------

USER root

# R pre-requisites
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libxrender1 \
    fonts-dejavu \
    gfortran \
    gcc && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER $NB_USER

# R packages
# tidyverse contains ggplot2, dplyr, tidyr, readr, purrr, tibble
RUN conda install --quiet --yes \
    'r-base=3.3.2' \
    'r-irkernel=0.7*' \
    'r-plyr=1.8*' \
    'r-devtools=1.12*' \
    'r-tidyverse=1.0*' \
    'r-shiny=0.14*' \
    'r-rmarkdown=1.2*' \
    'r-forecast=7.3*' \
    'r-rsqlite=1.1*' \
    'r-reshape2=1.4*' \
    'r-nycflights13=0.2*' \
    'r-caret=6.0*' \
    'r-rcurl=1.95*' \
    'r-crayon=1.3*' \
    'r-randomforest=4.6*' && conda clean -tipsy

# consider adding:
# ggExtra, jsonlite, xml2, stringr,

#--------------------------------------
# MATLAB
#--------------------------------------

RUN pip install pymatbridge
RUN pip install matlab_kernel
RUN python -m matlab_kernel install
ENV MATLAB_EXECUTABLE=/bin/MATLAB

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

USER root
RUN chown -R $NB_USER:users /home/$NB_USER

RUN ln -s /home/$NB_USER/work/data /data

WORKDIR /home/$NB_USER/work/
USER $NB_USER
