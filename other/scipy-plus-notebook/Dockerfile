# scipy-plus-notebook
#
# A copy of scipy-notebook with additional packages installed:
#
# build it:
# docker build -t grez72/scipy-plus-notebook https://github.com/grez72/notebooks.git#master:scipy-plus-notebook
#
# run it:
# docker run -it --rm -v /Applications/MATLAB_R2017b.app/bin/matlab:/bin/matlab -v /Applications/MATLAB_R2017b.app:/matlab -p 8888:8888 -v /anaconda/lib/python3.6/site-packages/matlab:/opt/conda/lib/python3.6/site-packages/matlab grez72/scipy-plus-notebook
# docker run -it --rm -v /Applications/MATLAB_R2017b.app:/matlab -p 8888:8888 -v /anaconda/lib/python3.6/site-packages/matlab:/opt/conda/lib/python3.6/site-packages/matlab grez72/scipy-plus-notebook
# docker run --user root -it --rm -v /Applications/MATLAB_R2017a.app/bin/matlab:/bin/matlab -p 8888:8888 grez72/scipy-plus-notebook
# cd /matlab/extern/engines/python
# python setup.py install
# ln -s /matlab/bin/matlab /usr/local/bin/matlab

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

# USER root
#
# # R pre-requisites
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#     libxrender1 \
#     fonts-dejavu \
#     gfortran \
#     gcc && apt-get clean && \
#     rm -rf /var/lib/apt/lists/*
#
# USER $NB_USER
#
# # R packages
# # tidyverse contains ggplot2, dplyr, tidyr, readr, purrr, tibble
# RUN conda install --quiet --yes \
#     'r-base=3.3.2' \
#     'r-irkernel=0.7*' \
#     'r-plyr=1.8*' \
#     'r-devtools=1.12*' \
#     'r-tidyverse=1.0*' \
#     'r-shiny=0.14*' \
#     'r-rmarkdown=1.2*' \
#     'r-forecast=7.3*' \
#     'r-rsqlite=1.1*' \
#     'r-reshape2=1.4*' \
#     'r-nycflights13=0.2*' \
#     'r-caret=6.0*' \
#     'r-rcurl=1.95*' \
#     'r-crayon=1.3*' \
#     'r-randomforest=4.6*' && conda clean -tipsy

# consider adding:
# ggExtra, jsonlite, xml2, stringr,

#--------------------------------------
# MATLAB
#--------------------------------------

USER root
RUN pip install pymatbridge
RUN pip install matlab_kernel
RUN python -m matlab_kernel install
ENV MATLAB_EXECUTABLE=/bin/matlab

#--------------------------------------
# Inatall PHP and PHP Kernel
#--------------------------------------

# USER root
# Install PHP7
# RUN apt-get update
# RUN apt-get install python-software-properties software-properties-common -y
# RUN apt-get install build-essential libtool autoconf uuid-dev pkg-config git libsodium-dev -y
# RUN LC_ALL=C.UTF-8 add-apt-repository ppa:ondrej/php -y
# RUN apt-get update
# RUN apt-get install php7.0 php7.0-dev php7.0-fpm php7.0-mysql -y
#
# # Install Composer
# RUN curl -sS https://getcomposer.org/installer | sudo php -- --install-dir=/usr/local/bin --filename=composer
#
# # Install ZeroMQ
# RUN apt-get install php-zmq
#
# # Jupyter-PHP kernel
# RUN wget --quiet https://litipk.github.io/Jupyter-PHP-Installer/dist/jupyter-php-installer.signed.phar && \
#     echo "a23fe19a4be460ceb993978c1d33ecd1314b1bd424cc1d1f1c360caba5642d65 jupyter-php-installer.signed.phar" | sha256sum -c
# RUN wget --quiet https://litipk.github.io/Jupyter-PHP-Installer/dist/jupyter-php-installer.signed.phar.pubkey && \
#     echo "aee84cbfb6fe0745c51f5069e989a2e062d5ca0167ca977b24de8bdb3c58853b jupyter-php-installer.signed.phar.pubkey" | sha256sum -c
# RUN wget --quiet https://litipk.github.io/Jupyter-PHP-Installer/dist/jupyter-php-installer.phar && \
#     echo "40226359b407409819b7636cfbee82d86adea57afef45fda4b2e3ddda5022844 jupyter-php-installer.phar" | sha256sum -c
# RUN wget https://litipk.github.io/Jupyter-PHP-Installer/dist/jupyter-php-installer.phar
# RUN php jupyter-php-installer.phar install

# RUN wget https://litipk.github.io/Jupyter-PHP-Installer/dist/jupyter-php-installer.signed.phar
# RUN wget https://litipk.github.io/Jupyter-PHP-Installer/dist/jupyter-php-installer.signed.phar.pubkey
# RUN php jupyter-php-installer.signed.phar install

#----------------------------------------
# Install NodeJS and NodeJS Kernel
#----------------------------------------

USER root

# inatall ijavascript
# RUN apt-get install software-properties-common python-software-properties
# RUN add-apt-repository ppa:chronitis/jupyter
# RUN apt-get update
# RUN apt-get install ijavascript

# nodejs
# RUN curl -sL https://deb.nodesource.com/setup_8.x | bash -
# RUN apt-get install -y nodejs

# jupyter-nodejs kernel
# WORKDIR /tmp
# RUN git clone https://github.com/notablemind/jupyter-nodejs.git
# WORKDIR c
# RUN mkdir -p ~/.ipython/kernels/nodejs/
# RUN npm install && node install.js
# RUN npm run build
# RUN npm run build-ext

# RUN mkdir -p  $HOME/.ipython/kernels/nodejs

# RUN apt-get update
# RUN apt-get install libtool pkg-config build-essential autoconf automake -y
# RUN apt-get install libzmq-dev -y
#
# WORKDIR /tmp
# RUN mkdir -p  $HOME/.ipython/kernels/nodejs
# RUN wget https://github.com/notablemind/jupyter-nodejs/releases/download/v1.1.0/jupyter-nodejs-1.1.0.tgz
# RUN tar xf jupyter-nodejs-1.1.0.tgz
# WORKDIR /tmp/package
# RUN npm install && node install.js
# RUN cp -R ~/.ipython/kernels/nodejs/ /usr/local/share/jupyter/kernels/

# install jupyter-nodejs
# RUN mkdir -p  $HOME/.ipython/kernels/nodejs
# RUN wget https://github.com/notablemind/jupyter-nodejs/releases/download/v1.1.0/jupyter-nodejs-1.1.0.tgz
# RUN tar xf jupyter-nodejs-1.1.0.tgz
# WORKDIR /tmp/package
# RUN npm install && node install.js

# jupyter console --kernel nodejs

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
