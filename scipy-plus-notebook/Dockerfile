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
# Install
#--------------------------------------

RUN pip2 install ipykernel

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
