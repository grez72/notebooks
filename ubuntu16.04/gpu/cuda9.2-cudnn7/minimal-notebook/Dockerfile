# grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.2-cudnn7-minimal-notebook
#
# Dervided from scipy notebook, with cuda9.2/cudnn7 support, and python2
# resinstated.
#
# build it:
# nvidia-docker build -t grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.2-cudnn7-minimal-notebook https://github.com/grez72/notebooks.git#master:ubuntu16.04/gpu/cuda9.2-cudnn7/minimal-notebook
#
# run it:
# nvidia-docker run -it --rm -p 8888:8888 grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.2-cudnn7-minimal-notebook

FROM grez72/base-notebook:ubuntu16.04-cuda9.2-cudnn7

#     && conda clean -yt
LABEL maintainer="George Alvarez <grez72@gmail.com, alvarez.wjh.harvard.edu>"

#--------------------------------------
# Utilities for optimizing python code:
#--------------------------------------

# Cupy, PyCuda, Numba
RUN conda install -y -c lukepfister pycuda \
  && /bin/bash -c "source activate python2 \
  && conda install -y -c lukepfister pycuda" \
  && pip install cupy \
  && /bin/bash -c "source activate python2 \
  && pip install cupy" \
  && conda install -y numba \
  && /bin/bash -c "source activate python2 \
  && conda install -y numba"

#--------------------------------------
#   FFMEG with libx264 support
#--------------------------------------

RUN apt-get update -qq && sudo apt-get -y install \
      autoconf \
      automake \
      build-essential \
      cmake \
      git-core \
      libass-dev \
      libfreetype6-dev \
      libsdl2-dev \
      libtool \
      libva-dev \
      libvdpau-dev \
      libvorbis-dev \
      libxcb1-dev \
      libxcb-shm0-dev \
      libxcb-xfixes0-dev \
      pkg-config \
      texinfo \
      wget \
      zlib1g-dev \
      yasm \
      nasm \
      libx264-dev \
      libx265-dev libnuma-dev \
      libvpx-dev \
      libfdk-aac-dev \
      libmp3lame-dev \
      libopus-dev \
    && rm -rf /var/lib/apt/lists/*

  RUN mkdir -p ~/ffmpeg_sources

  WORKDIR ~/ffmpeg_sources

  # RUN wget https://www.nasm.us/pub/nasm/releasebuilds/2.13.03/nasm-2.13.03.tar.bz2 && \
  #     tar xjvf nasm-2.13.03.tar.bz2 && \
  #     cd nasm-2.13.03 && \
  #     ./autogen.sh && \
  #     PATH="$HOME/bin:$PATH" ./configure --prefix="$HOME/ffmpeg_build" --bindir="$HOME/bin" && \
  #     make -j16 && \
  #     make install

  RUN wget -O ffmpeg-snapshot.tar.bz2 https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2 && \
      tar xjvf ffmpeg-snapshot.tar.bz2 && \
      rm -r ffmpeg-snapshot.tar.bz2 && \
      cd ffmpeg && \
      PATH="~/bin:$PATH" PKG_CONFIG_PATH="~/ffmpeg_build/lib/pkgconfig" ./configure \
      --prefix="~/ffmpeg_build" \
      --pkg-config-flags="--static" \
      --extra-cflags="-I~/ffmpeg_build/include" \
      --extra-ldflags="-L~/ffmpeg_build/lib" \
      --extra-libs="-lpthread -lm" \
      --bindir="/bin" \
      --enable-avresample \
      --enable-gpl \
      --enable-libass \
      --enable-libfdk-aac \
      --enable-libfreetype \
      --enable-libmp3lame \
      --enable-libopus \
      --enable-libvorbis \
      --enable-libvpx \
      --enable-libx264 \
      --enable-libx265 \
      --enable-nonfree && \
      PATH="~/bin:$PATH" make -j$(nproc) && \
      make install && \
      hash -r

#--------------------------------------
# Install OpenCV with CUDA Support
#--------------------------------------

# do we need to set these?
# export PATH=/usr/local/cuda/bin:$PATH
# export LPATH=/usr/local/nvidia/lib64:$LPATH
# export LIBRARY_PATH=/usr/local/nvidia:$LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib:$LD_LIBRARY_PATH

# RUN apt-get update
RUN apt-get update && apt-get install -y cmake \
  libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev \
  libgtk2.0-dev \
  libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
  libatlas-base-dev gfortran \
  libhdf5-serial-dev \
  python3-dev python2.7-dev

WORKDIR ~/
RUN git clone https://github.com/Itseez/opencv.git \
  && cd opencv \
  && git checkout tags/3.4.2 \
  && mkdir build \
  && cd build/ \
  && cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D PYTHON2_EXECUTABLE:FILEPATH=/opt/conda/envs/python2/bin/python2.7 \
    -D PYTHON2_INCLUDE_DIR:PATH=/opt/conda/envs/python2/include/python2.7 \
    -D PYTHON2_NUMPY_INCLUDE_DIRS:PATH=/opt/conda/envs/python2/lib/python2.7/site-packages/numpy/core/include \
    -D PYTHON2_PACKAGES_PATH:PATH=/opt/conda/envs/python2/lib/python2.7/site-packages \
    -D PYTHON3_EXECUTABLE:FILEPATH=/opt/conda/bin/python3.6 \
    -D PYTHON3_INCLUDE_DIR:PATH=/opt/conda/include/python3.6m \
    -D PYTHON3_NUMPY_INCLUDE_DIRS:PATH=/opt/conda/lib/python3.6/site-packages/numpy/core/include \
    -D PYTHON3_PACKAGES_PATH:PATH=/opt/conda/lib/python3.6/site-packages \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D BUILD_opencv_dnn=OFF \
    -D BUILD_opencv_dnn_modern=OFF \
    -D BUILD_PROTOBUF=OFF \
    -D WITH_CUDA=ON \
    -D FORCE_VTK=ON \
    -D WITH_TBB=ON \
    -D WITH_V4L=ON \
    -D WITH_OPENGL=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D WITH_FFMPEG=1 \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D BUILD_EXAMPLES=OFF .. \
  && make -j$(nproc) && sudo make install \
  && ldconfig

# why this setting?:
# -D BUILD_opencv_dnn=OFF \

# Had to turn off BUILD_opencv_dnn, because it uses conflicting version of Protobuf_LIBRARY
# alternatively, we can try to set these to our local Protobuf_LIBRARY
# -D Protobuf_LIBRARY=/opt/conda/lib \
# -D Protobuf_INCLUDE_DIR=/opt/conda/include \
# -D Protobuf_PROTOC_EXECUTABLE=/opt/conda/bin/protoc \

#--------------------------------------
# Install Video and Image Utilities
#--------------------------------------

# Installing all of these dependencies together in one RUN command
# to avoid limit on number of layers in docker image
RUN conda install -y -c conda-forge av trackpy ipywidgets \
  && conda install -y filterpy \
  && /bin/bash -c "source activate python2 \
  && conda install -y -c conda-forge av trackpy ipywidgets \
  && conda install -y filterpy" \
  && jupyter nbextension enable --py --sys-prefix widgetsnbextension \
  && pip install --upgrade pip \
  && pip install \
    ffmpeg-python \
    image-attendant \
    imutils \
    https://github.com/soft-matter/pims/archive/master.zip \
    tqdm \
  && cd $HOME \
  && git clone https://github.com/DheerendraRathor/ffprobe3 \
  && cd $HOME/ffprobe3 \
  && python setup.py install \
  && /bin/bash -c "source activate python2 \
  && pip install --upgrade pip \
  && pip install \
    ffmpeg-python \
    ffprobe \
    image-attendant \
    imutils \
    https://github.com/soft-matter/pims/archive/master.zip \
    simplejson \
    tqdm" \
  && cd $HOME \
  && git clone https://github.com/jfrelinger/cython-munkres-wrapper \
  && cd $HOME/cython-munkres-wrapper \
  && python setup.py install \
  && /bin/bash -c "source activate python2 && cd $HOME/cython-munkres-wrapper && python setup.py install"

#--------------------------------------
# update pyzmq to fix python2 kernel crash
#   AttributeError: type object 'IOLoop' has no attribute 'initialized'
#--------------------------------------

RUN /bin/bash -c "source activate python2 && conda install -c conda-forge pyzmq"

#--------------------------------------
# PyTurboJPEG
#--------------------------------------

RUN conda config --add channels conda-forge \
  && conda install -c loopbio libjpeg-turbo \
  && pip install -U git+git://github.com/lilohuang/PyTurboJPEG.git \
  && mkdir -p /opt/libjpeg-turbo/lib64 \
  && ln -s /opt/conda/lib/libjpeg-turbo/lib/libturbojpeg.so /opt/libjpeg-turbo/lib64/libturbojpeg.so \
  && /bin/bash -c "source activate python2 && conda config --add channels conda-forge \
    && conda install -c loopbio libjpeg-turbo \
    && pip install -U git+git://github.com/lilohuang/PyTurboJPEG.git"

#--------------------------------------
# Parallel HDF5
#--------------------------------------

# https://drtiresome.com/2016/08/23/build-and-install-mpi-parallel-hdf5-and-h5py-from-source-on-linux/

WORKDIR /tmp

# install MPICH, so we can pip install mpi4py
RUN wget http://www.mpich.org/static/downloads/3.2.1/mpich-3.2.1.tar.gz \
  && tar -xzf mpich-3.2.1.tar.gz \
  && cd mpich-3.2.1 \
  && ./configure --enable-romio --enable-shared --with-device=ch3:sock --disable-fortran --prefix=/usr/local \
  && make -j$(nproc) \
  && make install \
  && make clean

# install mpi4py
RUN pip install mpi4py \
  && /bin/bash -c "source activate python2 && pip install mpi4py"

# test file:
# test_mpi4py.py
# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# print("hello world from process ", rank)

# run test:
# mpiexec -n 5 python test_mpi4py.py

# https://github.com/HDFGroup/
# h5cc -showconfig
RUN wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.3/src/hdf5-1.10.3.tar.gz \
  && tar -xzf hdf5-1.10.3.tar.gz \
  && cd hdf5-1.10.3 \
  && ./configure --prefix=/usr/local --enable-parallel --enable-shared --enable-build-mode=production \
  && make -j$(nproc) \
  && make install \
  && make clean

# Test config
# h5pcc -showconfig
# wget https://support.hdfgroup.org/ftp/HDF5/examples/parallel/Hyperslab_by_row.c
# h5pcc Hyperslab_by_row.c
#

# h5dump SDS_row.h5

# RUN conda uninstall h5py hdf5
# ENV HDF5_DIR=/usr/local
# ENV CC=mpicc
# ENV HDF5_MPI="ON"
# RUN pip install --no-binary=h5py h5py

# link h5py to h5pcc
# export CC=mpicc
# RUN wget https://files.pythonhosted.org/packages/74/5d/6f11a5fffc3d8884bb8d6c06abbee0b3d7c8c81bde9819979208ba823a47/h5py-2.8.0.tar.gz \
#   && tar -xzf h5py-2.8.0.tar.gz \
#   && cd h5py-2.8.0 \
#   && python setup.py configure --mpi --hdf5=/usr/local \
#   && CC=mpicc python setup.py build \
#   && python setup.py install --user

# mpiexec -n 4 --allow-run-as-root python testh_5py.py

#--------------------------------------
# Install Additional Packages
#--------------------------------------

# plotly + dependencies
RUN pip install plotly \
  && conda install -c bioconda cufflinks \
  && /bin/bash -c "source activate python2 \
  && pip install plotly \
  && conda install -c bioconda cufflinks"

#--------------------------------------
# Setup for notebook use
#--------------------------------------

WORKDIR /home/$NB_USER/work/
USER $NB_USER
