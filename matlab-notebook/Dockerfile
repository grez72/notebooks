# matlab-notebook
#
# jupyter/minimal-notebook with matlab kernel installed
#
# build it:
# docker build -t grez72/matlab-notebook https://github.com/grez72/notebooks.git#master:matlab-notebook
#
# run it:
# docker run -it --rm -v /Applications/MATLAB_R2017b.app/bin/matlab:/bin/matlab -v /Applications/MATLAB_R2017b.app:/matlab -p 8888:8888 -v /anaconda/lib/python3.6/site-packages/matlab:/opt/conda/lib/python3.6/site-packages/matlab grez72/scipy-plus-notebook
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

USER root
RUN pip install pymatbridge
RUN pip install matlab_kernel
RUN python -m matlab_kernel install
ENV MATLAB_EXECUTABLE=/bin/matlab

#--------------------------------------
# Setup for notebook use
#--------------------------------------

USER root
RUN chown -R $NB_USER:users /home/$NB_USER

RUN ln -s /home/$NB_USER/work/data /data

WORKDIR /home/$NB_USER/work/
USER $NB_USER
