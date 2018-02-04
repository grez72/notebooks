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

#--------------------------------------
# Install OpenCV with CUDA Support
#--------------------------------------

RUN apt-get update
RUN apt-get install -y cmake \
  libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev \
  libgtk2.0-dev \
  libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
  libatlas-base-dev gfortran \
  libhdf5-serial-dev \
  python3-dev python2.7-dev

# RUN apt-get install emacs23 curl git autoconf lzip cmake-curses-gui cmake libtool unzip pkg-config build-essential python-pip ipython gfortran python-dev libatlas-base-dev python-scipy libfreetype6-dev libpng3 openssh-server zlibc libcurl3-dev

# RUN apt-get install -y build-essential cmake git libgtk2.0-dev pkg-config python-dev python-numpy libdc1394-22 libdc1394-22-dev libjpeg-dev libpng12-dev libtiff5-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libxine2-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev libtbb-dev libqt4-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev v4l-utils unzip yasm git subversion pkg-config checkinstall cmake libfaac-dev libjack-jackd2-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libsdl1.2-dev libtheora-dev libva-dev libvdpau-dev libvorbis-dev libx11-dev libxfixes-dev libxvidcore-dev texi2html zlib1g-dev libgstreamer0.10-0 libgstreamer0.10-dev gstreamer0.10-tools gstreamer0.10-plugins-base libgstreamer-plugins-base0.10-dev gstreamer0.10-plugins-good gstreamer0.10-plugins-ugly gstreamer0.10-plugins-bad libgtk2.0-0 libgtk2.0-dev libjpeg8 libjpeg8-dev freeglut3-dev qt-sdk libgdal-dev libwebp-dev libopenexr-dev libeigen3-dev python-tk python3-dev python3-tk python3-numpy ant default-jdk

WORKDIR ~/
RUN git clone https://github.com/Itseez/opencv.git \
  && cd opencv \
  && mkdir build \
  && cd build/ \
  && cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D WITH_CUDA=ON \
    -D CMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
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
