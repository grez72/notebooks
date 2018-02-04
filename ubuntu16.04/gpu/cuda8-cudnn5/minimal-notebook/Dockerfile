# grez72/deep-notebooks:ubuntu16.04-gpu-cuda8-cudnn5-minimal-notebook
#
# Dervided from scipy notebook, with cuda8/cudnn5 support, and python2
# resinstated.
#
# build it:
# nvidia-docker build -t grez72/deep-notebooks:ubuntu16.04-gpu-cuda8-cudnn5-minimal-notebook https://github.com/grez72/notebooks.git#master:ubuntu16.04/gpu/cuda8-cudnn5/minimal-notebook
#
# run it:
# nvidia-docker run -it --rm -p 8888:8888 grez72/deep-notebooks:ubuntu16.04-gpu-cuda8-cudnn5-minimal-notebook

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


RUN conda create --quiet --yes -p $CONDA_DIR/envs/python2 -c conda-forge python=2.7 ipython kernda=0.2 ipykernel && \
conda clean -tipsy

USER root

# Create a global kernelspec in the image and modify it so that it properly activates the python2 conda environment.

RUN $CONDA_DIR/envs/python2/bin/python -m ipykernel install && \
$CONDA_DIR/envs/python2/bin/kernda -o -y /usr/local/share/jupyter/kernels/python2/kernel.json

USER $NB_USER

#--------------------------------------
# Install OpenCV with CUDA Support
#--------------------------------------

RUN apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev \
  libgtk2.0-dev \
  libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
  libatlas-base-dev gfortran \
  libhdf5-serial-dev \
  python2.7-dev

RUN cd ~/ \
  && git clone https://github.com/Itseez/opencv.git \
  && cd opencv \
  && mkdir build \
  && cd build/ \
  && cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D WITH_CUDA=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D BUILD_EXAMPLES=ON .. \
  && make -j$(nproc) && sudo make install \
  && ldconfig

#--------------------------------------
# Install Additional Packages
#--------------------------------------

# plotly, cufflinks

# python2
# RUN /bin/bash -c "source activate python2 && ipython kernel install"

#--------------------------------------
# Setup for notebook use
#--------------------------------------

WORKDIR /home/$NB_USER/work/
USER $NB_USER
