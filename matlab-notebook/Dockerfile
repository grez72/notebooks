# grez72/matlab-notebook
#
# jupyter/scipy-notebook with matlab kernel installed
# This notebook requires that the host has a licensed copy of MATLAB installed,
# and python engine installed with the same version of python as the docker
# container (in this case python=3.6 for latest scipy-notebook).
#
# install python engine on the host:
# cd /usr/local/MATLAB/R2017b/extern/engines/python
# python setup.py install
#
# for pre R2017b MATLAB, see https://gist.github.com/dangom/303879ca649f7a3fe113912d7ffa62d4
#
# build docker image:
# docker build -t grez72/matlab-notebook https://github.com/grez72/notebooks.git#master:matlab-notebook
#
# To run the image, we have to map volumes from the host to the container:
#   MATLAB app:           -v /usr/local/MATLAB/R2017b:/usr/local/MATLAB/R2017b
#   MATLAB python engine: -v /home/george/anaconda3/lib/python3.6/site-packages/matlab:/opt/conda/lib/python3.6/site-packages/matlab
#   eth0 MAC address:     --mac-address=$(ifconfig eth0 | awk '/^[a-z]/ { iface=$1; mac=$NF; next } /inet addr:/ { print mac }')
#
# run it
# docker run -it --rm -v /home/george/anaconda3/lib/python3.6/site-packages/matlab:/opt/conda/lib/python3.6/site-packages/matlab -v /usr/local/MATLAB/R2017b:/usr/local/MATLAB/R2017b --mac-address=$(ifconfig eth0 | awk '/^[a-z]/ { iface=$1; mac=$NF; next } /inet addr:/ { print mac }') -p 8888:8888 grez72/matlab-notebook

FROM jupyter/scipy-notebook
MAINTAINER George Alvarez <grez72@gmail.com, alvarez.wjh.harvard.edu>

#--------------------------------------
# MATLAB
#--------------------------------------

USER $NB_USER

# make these to avoid permission error
RUN mkdir /home/$NB_USER/.jupyter && \
    mkdir -p /home/jovyan/.local/share/jupyter

RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install pymatbridge
RUN pip install matlab_kernel

USER root
RUN python -m matlab_kernel install
ENV MATLAB_EXECUTABLE=/usr/local/MATLAB/R2017b/bin/matlab

#--------------------------------------
# Setup for notebook use
#--------------------------------------

USER root
RUN chown -R $NB_USER:users /home/$NB_USER

# RUN ln -s /home/$NB_USER/work/data /data

WORKDIR /home/$NB_USER/work/
USER $NB_USER
