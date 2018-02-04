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
# Install FFmpeg
#--------------------------------------

# RUN apt-get update -qq && apt-get -y install \
#   autoconf \
#   automake \
#   build-essential \
#   cmake \
#   git \
#   libass-dev \
#   libfreetype6-dev \
#   libsdl2-dev \
#   libtheora-dev \
#   libtool \
#   libva-dev \
#   libvdpau-dev \
#   libvorbis-dev \
#   libxcb1-dev \
#   libxcb-shm0-dev \
#   libxcb-xfixes0-dev \
#   mercurial \
#   pkg-config \
#   texinfo \
#   wget \
#   zlib1g-dev
#
# RUN mkdir -p ~/ffmpeg_sources ~/bin
#
# cd ~/ffmpeg_sources && \
# wget http://www.nasm.us/pub/nasm/releasebuilds/2.13.02/nasm-2.13.02.tar.bz2 && \
# tar xjvf nasm-2.13.02.tar.bz2 && \
# cd nasm-2.13.02 && \
# ./autogen.sh && \
# PATH="$HOME/bin:$PATH" ./configure --prefix="$HOME/ffmpeg_build" --bindir="$HOME/bin" && \
# make -j$(nproc) && \
# make install
#
# sudo apt-get install yasm
# sudo apt-get install libx264-dev
# sudo apt-get install libvpx-dev
# sudo apt-get install libfdk-aac-dev
# sudo apt-get install libmp3lame-dev
# sudo apt-get install libopus-dev
#
# cd ~/ffmpeg_sources && \
# if cd x265 2> /dev/null; then hg pull && hg update; else hg clone https://bitbucket.org/multicoreware/x265; fi && \
# cd x265/build/linux && \
# PATH="$HOME/bin:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$HOME/ffmpeg_build" -DENABLE_SHARED:bool=off ../../source && \
# PATH="$HOME/bin:$PATH" make -j$(nproc) && \
# make install
#
# # then, install the required packages
# sudo apt-get install libfaac-dev
# sudo apt-get install libmp3lame-dev
# sudo apt-get install libtheora-dev
# sudo apt-get install libvorbis-dev
# sudo apt-get install libxvidcore-dev
# sudo apt-get install libxext-dev
#
# cd ~/ffmpeg_sources && \
# wget -O ffmpeg-snapshot.tar.bz2 http://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2 && \
# tar xjvf ffmpeg-snapshot.tar.bz2 && \
# cd ffmpeg && \
# PATH="$HOME/bin:$PATH" PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure \
#   --prefix="$HOME/ffmpeg_build" \
#   --pkg-config-flags="--static" \
#   --extra-cflags="-I$HOME/ffmpeg_build/include" \
#   --extra-ldflags="-L$HOME/ffmpeg_build/lib" \
#   --extra-libs="-lpthread -lm" \
#   --bindir="$HOME/bin" \
#   --enable-gpl \
#   --enable-libass \
#   --enable-libfdk-aac \
#   --enable-libfreetype \
#   --enable-libmp3lame \
#   --enable-libopus \
#   --enable-libtheora \
#   --enable-libvorbis \
#   --enable-libvpx \
#   --enable-libx264 \
#   --enable-libx265 \
#   --enable-nonfree && \
# PATH="$HOME/bin:$PATH" make -j$(nproc) && \
# make install
# hash -r
#
# # configure the PATH param
# echo "# Add FFMpeg bin & library paths:" >> ~/.bashrc
# echo "export PATH=$HOME/bin:$PATH" >> ~/.bashrc
# # echo "export LD_LIBRARY_PATH=/usr/local/ffmpeg/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
# source ~/.bashrc

# #upgrade cmake to v3.2  [DON'T DO THIS WHILE OTHER CMAKE-USING SOFTWARE IS BEING INSTALLED]
# cd ~/
# sudo apt-get install software-properties-common
# sudo add-apt-repository ppa:george-edison55/cmake-3.x
# sudo apt-get update
# sudo apt-get upgrade
# sudo apt-get install cmake
#
# #boost (~15 min)
# cd ~/
# wget http://sourceforge.net/projects/boost/files/boost/1.59.0/boost_1_59_0.tar.gz
# tar -xvf boost_1_59_0.tar.gz
# cd boost_1_59_0
# ./bootstrap.sh
# sudo ./b2 install
#
#ffmpeg (~6min)
# cd ~/
# sudo apt-get install nasm
# wget http://ffmpeg.org/releases/ffmpeg-3.0.tar.bz2
# tar -xjvf ffmpeg-3.0.tar.bz2
# cd ffmpeg-3.0
# ./configure --enable-shared
# sudo make -j${nproc} && sudo make install

# #protobuf (17 min)
# cd ~/
# git clone https://github.com/google/protobuf.git
# cd protobuf
# ./autogen.sh
# ./configure
# make && make check
# sudo make install
# cd python
# sudo python setup.py install
#
# #glog (~2min)
# #saw error and used fix in https://github.com/google/glog/issues/52
# #had to upgrade cmake to version 3
# cd ~/
# git clone https://github.com/google/glog.git
# cd glog
# mkdir build && cd build
# export CXXFLAGS="-fPIC" && cmake .. && make VERBOSE=1
# make && sudo make install

# #gflags (~5min)
# #first install ccmake
# cd ~/
# git clone https://github.com/gflags/gflags.git
# cd gflags
# mkdir build && cd build
# ccmake ..
#         #follow instructions
#         # c to configure, didn't change anything
#         # g to generate and exit
#         # not sure if this was correct, but allowed next step
# make && sudo make install
#
# #lmdb (~1min)
# cd ~/
# git clone https://github.com/LMDB/lmdb.git
# cd lmdb/libraries/liblmdb
# sudo make && sudo make install
#
# #snappy (~4min)
# cd ~/
# wget https://github.com/google/snappy/releases/download/1.1.3/snappy-1.1.3.tar.gz
# tar -xvf snappy-1.1.3.tar.gz
# cd snappy-1.1.3
# ./configure
# make && make check
# sudo make install

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
