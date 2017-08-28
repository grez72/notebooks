# matlab-notebook
#
# jupyter/minimal-notebook with matlab kernel installed
#
# build it:
# docker build -t grez72/matlab-notebook https://github.com/grez72/notebooks.git#master:matlab-notebook
#
# run it:
# docker run -it --rm --user root -v ./usr/local/MATLAB/R2017b/:/bin/matlab -p 3000:8888 grez72/matlab-notebook
# docker run -it --rm -v ./usr/local/MATLAB/R2017b/:/usr/local/MATLAB/from-host -v /home/george/matlab-docker:/var/log/matlab -p 8888:8888 --mac-address=38:d5:47:c9:33:fe grez72/matlab-notebook
# docker run -it --rm -v ./usr/local/MATLAB/R2017b/bin/matlab:/bin/matlab -v /Applications/MATLAB_R2017b.app:/matlab -p 8888:8888 -v /anaconda/lib/python3.6/site-packages/matlab:/opt/conda/lib/python3.6/site-packages/matlab grez72/scipy-plus-notebook
# docker run -it --rm -v /Applications/MATLAB_R2017b.app:/matlab -p 8888:8888 -v /anaconda/lib/python3.6/site-packages/matlab:/opt/conda/lib/python3.6/site-packages/matlab grez72/scipy-plus-notebook
# docker run --user root -it --rm -v /Applications/MATLAB_R2017a.app/bin/matlab:/bin/matlab -p 8888:8888 grez72/scipy-plus-notebook
# cd /matlab/extern/engines/python
# python setup.py install
# ln -s /matlab/bin/matlab /usr/local/bin/matlab

FROM jupyter/minimal-notebook
MAINTAINER George Alvarez <grez72@gmail.com, alvarez.wjh.harvard.edu>

#--------------------------------------
# MATLAB
#--------------------------------------

# make these to avoid permission error
# RUN mkdir /home/$NB_USER/.jupyter && \
#     mkdir /home/$NB_USER/.local

# USER root
RUN pip install numpy
RUN pip install pymatbridge
RUN pip install matlab_kernel

USER root
RUN python -m matlab_kernel install
ENV MATLAB_EXECUTABLE=/bin/matlab

# Matlab and Python dependencies
# RUN apt-get update && apt-get install -y \
#     libpng12-dev libfreetype6-dev \
#     libblas-dev liblapack-dev gfortran build-essential xorg xorg-dev python-dev pkg-config
#
# # Python libs
# RUN pip install pyzmq \
#     && pip install matplotlib \
#     && pip install numpy \
#     && pip install scipy \
#     && pip install nibabel \
#     && pip install pymatbridge \
#     && pip install jupyter \
#     && pip install jupyter_client \
#     && pip install matlab_kernel \
#     && python -m matlab_kernel.install
#
# RUN ln -s /usr/local/MATLAB/from-host/bin/matlab /usr/local/bin/matlab

#--------------------------------------
# Setup for notebook use
#--------------------------------------

USER root
RUN chown -R $NB_USER:users /home/$NB_USER

RUN ln -s /home/$NB_USER/work/data /data

WORKDIR /home/$NB_USER/work/
USER $NB_USER
