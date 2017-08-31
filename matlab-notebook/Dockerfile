# grez72/matlab-notebook
#
# jupyter/minimal-notebook with matlab kernel installed
# This notebook requires that the host has a licensed copy of MATLAB installed,
# and python engine installed.
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
# run it:
# docker run -it --rm -v /home/george/anaconda3/lib/python3.6/site-packages/matlab:/opt/conda/lib/python3.6/site-packages/matlab -v /usr/local/MATLAB/R2017b:/usr/local/MATLAB/R2017b -p 3000:8888 --mac-address=$(ifconfig eth0 | awk '/^[a-z]/ { iface=$1; mac=$NF; next } /inet addr:/ { print mac }') grez72/matlab-notebook
# docker run -it --rm -v /home/george/anaconda3/lib/python3.6/site-packages/matlab:/opt/conda/lib/python3.6/site-packages/matlab -v /usr/local/MATLAB/R2017b:/usr/local/MATLAB/R2017b -v /usr/local/MATLAB/R2017b/bin/matlab:/bin/matlab -p 3000:8888 --mac-address=$(ifconfig eth0 | awk '/^[a-z]/ { iface=$1; mac=$NF; next } /inet addr:/ { print mac }') grez72/matlab-notebook
#
# this one works, but I have to complie python engine after the container starts:
# docker run -it --rm -v /usr/local/MATLAB/R2017b:/MATLAB -v /usr/local/MATLAB/R2017b/bin/matlab:/bin/matlab -p 3000:8888 --mac-address=$(ifconfig eth0 | awk '/^[a-z]/ { iface=$1; mac=$NF; next } /inet addr:/ { print mac }') grez72/matlab-notebook
# cd /matlab/extern/engines/python
# python setup.py install

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
