# grez72/base-notebook for deep learning
#
# Runs a fully-loaded jupyterlab notebook server with access to the GPU on
# the host machine, running Ubuntu 16.04, CUDA 10.1, CUDNN7, python2, python3,
# and including most of the packages you might need for deep learning,
# experiments except a deep learning framework! (we build those from this image)
#
# The image begins with nvidia/cuda image, then adds in jupyter/docker-stacks/scipy-notebook
# https://github.com/jupyter/docker-stacks/tree/master/scipy-notebook
#
# The Jupyter server uses a password specified in jupyter_notebook_config.py
# To use tokens instead, comment out the password line of the config file:
# c.NotebookApp.password = 'sha1:b43e2d39955f:b09acc36be8f4d051f6922e10f024db6d7dddd4b'
#
# You can then generate your own password by running the following in python:
# from notebook.auth import passwd
# print(passwd('supersecretpassword'))
#
# Then update the config file:
# c.NotebookApp.password = '...'
#
# uncomment the password line, and rebuild the image (fast because the config
# is copied at the end). You should then be able to login with your password.
#
# build it:
# docker build -t grez72/base-notebook:ubuntu18.04-cuda10.1-cudnn7 https://github.com/grez72/notebooks.git#master:ubuntu18.04/cuda10.1-cudnn7/base-notebook-v1.1
#
# test it:
# docker run -it --rm grez72/base-notebook:ubuntu18.04-cuda10.1-cudnn7 nvidia-smi
#
# generate a password:
# nvidia-docker run -it --rm grez72/base-notebook:ubuntu18.04-cuda10.1-cudnn7 python genpass.py -p 'supersecretpassword'
#
# mount it:
# docker run -it --rm --user root grez72/base-notebook:ubuntu18.04-cuda10.1-cudnn7 bash
#
# run it:
# docker run -it --rm -p 8888:8888 grez72/base-notebook:ubuntu18.04-cuda10.1-cudnn7
#
# If you are using jupyterhub, this image can be used with Dockerspawner to
# spawn a single-user notebook.

FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

LABEL maintainer="George Alvarez <alvarez@wjh.harvard.com>"

USER root

#--------------------------------------
#   OS dependencies
#--------------------------------------

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get -yq dist-upgrade \
  && apt-get install -yq --no-install-recommends \
  wget \
  bzip2 \
  ca-certificates \
  sudo \
  locales \
  fonts-liberation \
  tmux \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
  locale-gen

#--------------------------------------
#   Configure environment
#--------------------------------------

ENV CONDA_DIR=/opt/conda \
  SHELL=/bin/bash \
  NB_USER=jovyan \
  NB_UID=1000 \
  NB_GID=100 \
  LC_ALL=en_US.UTF-8 \
  LANG=en_US.UTF-8 \
  LANGUAGE=en_US.UTF-8 \
  JUPYTER_ENABLE_LAB=1
ENV PATH=$CONDA_DIR/bin:$PATH \
  HOME=/home/$NB_USER

ADD fix-permissions /usr/local/bin/fix-permissions
# RUN chmod u+x /usr/local/bin/fix-permissions
# Create jovyan user with UID=1000 and in the 'users' group
# and make sure these dirs are writable by the `users` group.
RUN groupadd wheel -g 11 && \
  echo "auth required pam_wheel.so use_uid" >> /etc/pam.d/su && \
  useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
  mkdir -p $CONDA_DIR && \
  chown $NB_USER:$NB_GID $CONDA_DIR && \
  chmod g+w /etc/passwd && \
  fix-permissions $HOME && \
  fix-permissions $CONDA_DIR

# Setup work directory
USER $NB_UID

RUN mkdir /home/$NB_USER/work && \
  fix-permissions /home/$NB_USER

#--------------------------------------
#   Install conda / python
#--------------------------------------

# Install conda as NB_USER and check the md5 sum provided on the download site
ENV MINICONDA_VERSION 4.5.4
ENV MINICONDA_CHECKSUM a946ea1d0c4a642ddf0c3a26a18bb16d
# ENV MINICONDA_VERSION 4.7.10
# ENV MINICONDA_CHECKSUM 1c945f2b3335c7b2b15130b1b2dc5cf4
RUN cd /tmp && \
  wget --quiet https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
  echo "${MINICONDA_CHECKSUM} *Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh" | md5sum -c - && \
  /bin/bash Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
  rm Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
  $CONDA_DIR/bin/conda config --system --prepend channels conda-forge && \
  $CONDA_DIR/bin/conda config --system --set auto_update_conda false && \
  $CONDA_DIR/bin/conda config --system --set show_channel_urls true && \
  $CONDA_DIR/bin/conda install --quiet --yes conda="${MINICONDA_VERSION%.*}.*" && \
  conda install -y -c conda-forge libiconv && \
  conda clean -tipsy && \
  rm -rf /home/$NB_USER/.cache/yarn && \
  fix-permissions $CONDA_DIR && \
  fix-permissions /home/$NB_USER

# $CONDA_DIR/bin/conda update --all --quiet --yes && \

# Install Tini
RUN conda install --quiet --yes 'tini=0.18.0' && \
  conda list tini | grep tini | tr -s ' ' | cut -d ' ' -f 1,2 >> $CONDA_DIR/conda-meta/pinned && \
  conda clean -tipsy && \
  fix-permissions $CONDA_DIR && \
  fix-permissions /home/$NB_USER

#--------------------------------------
#   Install Jupyter Notebook, Lab, and Hub
#--------------------------------------

USER $NB_UID

# Generate a notebook server config
# Cleanup temporary files
# Correct permissions
# Do all this in a single RUN command to avoid duplicating all of the
# files across image layers when the permissions change
RUN conda install --quiet --yes \
  'notebook=5.7.*' \
  'jupyterhub=0.9.*' \
  'jupyterlab=0.35.*' && \
  conda clean -tipsy && \
  jupyter labextension install @jupyterlab/hub-extension@^0.12.0 && \
  npm cache clean --force && \
  jupyter notebook --generate-config && \
  rm -rf $CONDA_DIR/share/jupyter/lab/staging && \
  rm -rf /home/$NB_USER/.cache/yarn && \
  fix-permissions $CONDA_DIR && \
  fix-permissions /home/$NB_USER

#--------------------------------------
# Install all OS dependencies for fully functional notebook server:
#--------------------------------------

USER root

RUN apt-get update && apt-get install -yq --no-install-recommends \
  build-essential \
  emacs \
  git \
  inkscape \
  jed \
  libsm6 \
  libxext-dev \
  libxrender1 \
  lmodern \
  netcat \
  pandoc \
  python-dev \
  texlive-fonts-extra \
  texlive-fonts-recommended \
  texlive-generic-recommended \
  texlive-latex-base \
  texlive-latex-extra \
  texlive-xetex \
  unzip \
  nano \
  && apt-get clean && \
  rm -rf /var/lib/apt/lists/*

#--------------------------------------
# FFMEG with support for: with libx264 (mp4), Nvidia NVENC/NVDEC,
#   and cuda hardware acceleration:
#   https://trac.ffmpeg.org/wiki/HWAccelIntro
#   https://trac.ffmpeg.org/wiki/HWAccelIntro#NVENC
#--------------------------------------

USER root

RUN apt-get update -qq && apt-get -y install \
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

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# double check to make sure dependencies installed
RUN apt-get -y update && apt-get install -y wget nano git build-essential yasm pkg-config

# install nvidia headers for nvidia codecs
RUN git clone https://github.com/FFmpeg/nv-codec-headers /root/nv-codec-headers && \
  cd /root/nv-codec-headers &&\
  make -j$(nproc) && \
  make install -j$(nproc) && \
  cd /root && rm -rf nv-codec-headers

# Compile and install ffmpeg from source
# This version works, but my original version had more features enabled,
# so down below I've combined them.
# RUN git clone https://github.com/FFmpeg/FFmpeg /root/ffmpeg && \
#   cd /root/ffmpeg && ./configure \
#   --enable-nonfree \
#   --disable-shared \
#   --enable-nvenc \
#   --enable-cuda \
#   --enable-cuvid \
#   --enable-libnpp \
#   --extra-cflags=-I/usr/local/cuda/include \
#   --extra-cflags=-I/usr/local/include \
#   --extra-ldflags=-L/usr/local/cuda/lib64 && \
#   make -j$(nproc) && \
#   make install -j$(nproc) && \
#   cd /root && rm -rf ffmpeg

RUN git clone https://github.com/FFmpeg/FFmpeg /root/ffmpeg \
  && cd /root/ffmpeg \
  && PATH="~/bin:$PATH" PKG_CONFIG_PATH="/root/ffmpeg/lib/pkgconfig" ./configure \
  --prefix="/root/ffmpeg" \
  --pkg-config-flags="--static" \
  --extra-cflags="-I/root/ffmpeg/include" \
  --extra-ldflags="-L/root/ffmpeg/lib" \
  --extra-libs="-lpthread -lm" \
  --bindir="/bin" \
  --disable-shared \
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
  --enable-nonfree \
  --enable-cuda \
  --enable-cuvid \
  --enable-nvenc \
  --enable-nonfree \
  --enable-libnpp \
  --extra-cflags=-I/usr/local/cuda/include \
  --extra-cflags=-I/usr/local/include \
  --extra-ldflags=-L/usr/local/cuda/lib64 \
  --extra-libs="-lpthread -lm" \
  && make -j$(nproc) \
  && make install -j$(nproc) \
  && cd /root && rm -rf ffmpeg

# Check for nvidia codecs
# ffmpeg -encoders 2>/dev/null | grep nvenc

# Test hardware acceleration
# cd /root
# wget https://www.dropbox.com/s/oyhp2pv4zaae29f/youtube_epVDutrGlDU.mp4
# mkdir test
# ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=nw=1 youtube_epVDutrGlDU.mp4
# ffmpeg -hwaccel_device 0 -hwaccel nvdec -i youtube_epVDutrGlDU.mp4 -qmin 1 -qmax 1 -qscale:v 1 test/$filename%04d.jpg
# ffmpeg -hwaccel_device 0 -hwaccel nvdec -c:v h264_cuvid -i youtube_epVDutrGlDU.mp4 output.mkv

# ffmpeg -hwaccel_device 0 -hwaccel cuvid -c:v h264_cuvid -i input -vf scale_npp=-1:720 -c:v h264_nvenc -preset slow output.mkv
# ffmpeg -y -hwaccel cuvid -c:v h264_cuvid -vsync 0 -i <input.mp4> -vf scale_npp=1920:1072 -vcodec h264_nvenc <output0.264> -vf scale_npp=1280:720 -vcodec h264_nvenc <output1.264>
# ffmpeg -y -hwaccel cuvid -c:v h264_cuvid -vsync 0 -i VP13_z4LO6M.mp4 -vf scale_npp=1920:1072 -vcodec h264_nvenc VP13_z4LO6M.264 -vf scale_npp=1280:720 -vcodec h264_nvenc VP13_z4LO6M.264
# ffmpeg -i VP13_z4LO6M.mp4 -c:v h264_nvenc -profile:v high444p -pixel_format yuv444p -preset default output.mp4
# ffmpeg -c:v h264_cuvid -i input output.mkv
# ffmpeg -c:v h264_cuvid -i youtube_epVDutrGlDU.mp4 output.mkv
# ffmpeg -hwaccel_device 0 -hwaccel cuvid -c:v h264_cuvid -i youtube_epVDutrGlDU.mp4 -i "%1" test/$filename%04d.jpg
# ffmpeg -hwaccel_device 0 -hwaccel cuvid -i youtube_epVDutrGlDU.mp4 -qmin 1 -qmax 1 -qscale:v 1 test/$filename%04d.jpg
# ffmpeg -hwaccel_device 0 -hwaccel nvdec -i youtube_epVDutrGlDU.mp4 -qmin 1 -qmax 1 -qscale:v 1 test/$filename%04d.jpg

# PyNVVL is a thin wrapper of NVIDIA Video Loader (NVVL) https://github.com/NVIDIA/nvvl
# RUN pip install pynvvl-cuda92

# Trying to compile FFmpeg with Nvidia NVENC and cuda hardware acceleration support enabled
# would also like to use: https://github.com/mitmul/pynvvl
# https://github.com/lutris/ffmpeg-nvenc
# https://trac.ffmpeg.org/wiki/HWAccelIntro

# need nasm?
# http://www.linuxfromscratch.org/blfs/view/svn/general/nasm.html

# http://developer.download.nvidia.com/compute/redist/ffmpeg/1511-patch/FFMPEG-with-NVIDIA-Acceleration-on-Ubuntu_UG_v01.pdf

# http://developer.download.nvidia.com/compute/redist/ffmpeg/1511-patch/FFMPEG-with-NVIDIA-Acceleration-on-Ubuntu_UG_v01.pdf

# echo "Installing the NVidia Video SDK"
# wget -4 http://developer.download.nvidia.com/assets/cuda/files//nvidia_video_sdk_8.2.16

# sdk_version="8.2.16"
# sdk_basename="nvidia_video_sdk_${sdk_version}"
# sdk_url="http://developer.download.nvidia.com/assets/cuda/files/${sdk_basename}.zip"
# cd $source_dir
# wget -4 $sdk_url
# unzip "${sdk_basename}.zip"
# cd $sdk_basename
# cp -a Samples/common/inc/* $inc_dir
# wget -4 http://developer.download.nvidia.com/assets/cuda/files/nvidia_video_sdk_6.0.1.zip

# WORKDIR /tmp
# RUN git clone --depth=1 https://git.videolan.org/git/ffmpeg/nv-codec-headers.git nv-codec-headers && \
#   cd nv-codec-headers && \
#   make install PREFIX=/usr && \
#   cd .. && \
#   rm -rf nv-codec-headers

# RUN mkdir -p /usr/local/src/ffmpeg_sources
# WORKDIR /usr/local/src/ffmpeg_sources

# RUN mkdir -p ~/ffmpeg_build
# && mkdir -p ~/ffmpeg_build/bin \
# && mkdir -p ~/ffmpeg_build/include \
# && cp -a /tmp/nvidia_video_sdk_6.0.1/Samples/common/inc/* ~/ffmpeg_build/include
# cp -a /tmp/nvidia_video_sdk_6.0.1/Samples/common/inc/* /usr/local/include

# WORKDIR ~/ffmpeg_build

# RUN wget https://www.nasm.us/pub/nasm/releasebuilds/2.13.03/nasm-2.13.03.tar.bz2 && \
#     tar xjvf nasm-2.13.03.tar.bz2 && \
#     cd nasm-2.13.03 && \
#     ./autogen.sh && \
#     PATH="$HOME/bin:$PATH" ./configure --prefix="$HOME/ffmpeg_build" --bindir="$HOME/bin" && \
#     make -j16 && \
#     make install

# http://developer.download.nvidia.com/compute/redist/ffmpeg/1511-patch/FFMPEG-with-NVIDIA-Acceleration-on-Ubuntu_UG_v01.pdf
# https://github.com/lutris/ffmpeg-nvenc/blob/master/build.sh

# RUN wget -O ffmpeg-snapshot.tar.bz2 https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2 && \
#     tar xjvf ffmpeg-snapshot.tar.bz2 && \
#     rm -r ffmpeg-snapshot.tar.bz2 && \
#     cd ffmpeg && \
#     PATH="~/bin:$PATH" PKG_CONFIG_PATH="~/ffmpeg_build/lib/pkgconfig" ./configure \
#     --prefix="~/ffmpeg_build" \
#     --pkg-config-flags="--static" \
#     --extra-cflags="-I~/ffmpeg_build/include -I/usr/local/cuda/include" \
#     --extra-ldflags="-L~/ffmpeg_build/lib -L/usr/local/cuda/lib64" \
#     --extra-libs="-lpthread -lm" \
#     --bindir="/bin" \
#     --enable-avresample \
#     --enable-gpl \
#     --enable-libass \
#     --enable-libfdk-aac \
#     --enable-libfreetype \
#     --enable-libmp3lame \
#     --enable-libopus \
#     --enable-libvorbis \
#     --enable-libvpx \
#     --enable-libx264 \
#     --enable-libx265 \
#     --enable-nonfree \
#     --enable-cuda \
#     --enable-cuvid \
#     --enable-nvenc \
#     --enable-nonfree \
#     --enable-libnpp \
#     && PATH="~/bin:$PATH" make -j$(nproc) \
#     && make install \
#     && hash -r

# Add these to enable FFmpeg/libav using NVIDIA GPUs
# --enable-nonfree \
# --enable-cuda \
# --enable-cuvid \
# --enable-nvenc \
# --enable-nonfree \
# --enable-libnpp \
# --extra-cflags="-I/usr/local/cuda/include" \
# --extra-ldflags="-L/usr/local/cuda/lib64" && \

#--------------------------------------
#   Add Conda Environments with Intel Python
#--------------------------------------

# RUN conda update conda \
#     && conda config --add channels intel \
#     && conda create -n python3 intelpython3_core python=3 \
#     && conda create -n python2 intelpython2_core python=2 \
#     && /bin/bash -c "source activate python3 \
#     && conda install pip ipykernel \
#     && python -m ipykernel install --user --name python3 --display-name 'Python3'" \
#     && /bin/bash -c "source activate python2 \
#     && conda install pip ipykernel \
#     && python -m ipykernel install --user --name python2 --display-name 'Python2'"

#--------------------------------------
# Setup Python 3 Environment
#--------------------------------------

USER $NB_UID
WORKDIR $HOME

# Install Python 3 packages
# Remove pyqt and qt pulled in for matplotlib since we're only ever going to
# use notebook-friendly backends in these images
RUN conda install --quiet --yes \
  "conda=${MINICONDA_VERSION}" \
  'av' \
  'conda-forge::blas=*=openblas' \
  'ipywidgets=7.2*' \
  'pandas=0.23*' \
  'numexpr=2.6*' \
  'matplotlib=2.2*' \
  'scipy=1.1*' \
  'seaborn=0.9*' \
  'scikit-learn=0.19*' \
  'scikit-image=0.14*' \
  'sk-video=1.1*' \
  'sympy=1.1*' \
  'cython=0.28*' \
  'patsy=0.5*' \
  'statsmodels=0.9*' \
  'cloudpickle=0.5*' \
  'dill=0.2*' \
  'numba=0.38*' \
  'bokeh=0.12*' \
  'sqlalchemy=1.2*' \
  'hdf5=1.10*' \
  'h5py=2.7*' \
  'vincent=0.4.*' \
  'beautifulsoup4=4.6.*' \
  'protobuf=3.*' \
  'xlrd'  && \
  conda remove --quiet --yes --force qt pyqt && \
  conda clean -tipsy && \
  npm cache clean --force && \
  rm -rf $CONDA_DIR/share/jupyter/lab/staging && \
  rm -rf /home/$NB_USER/.cache/yarn && \
  rm -rf /home/$NB_USER/.node-gyp && \
  fix-permissions $CONDA_DIR && \
  fix-permissions /home/$NB_USER

# jupyter nbextension enable --py widgetsnbextension --sys-prefix && \
# jupyter labextension install @jupyter-widgets/jupyterlab-manager@^0.38.1 && \
# jupyter labextension install jupyterlab_bokeh@^0.6.3 && \

# Install facets which does not have a pip or conda package at the moment
RUN cd /tmp && \
  git clone https://github.com/PAIR-code/facets.git && \
  cd facets && \
  jupyter nbextension install facets-dist/ --sys-prefix && \
  cd && \
  rm -rf /tmp/facets && \
  fix-permissions $CONDA_DIR && \
  fix-permissions /home/$NB_USER

# Import matplotlib the first time to build the font cache.
ENV XDG_CACHE_HOME /home/$NB_USER/.cache/
RUN MPLBACKEND=Agg python -c "import matplotlib.pyplot" && \
  fix-permissions /home/$NB_USER

# #--------------------------------------
# # Setup Python 2 Environment
# #--------------------------------------
#
# USER $NB_USER
#
# # Install Python 2 packages
# RUN conda create -p $CONDA_DIR/envs/python2 python=2.7 \
#     'av' \
#     'ipython=4.0*' \
#     'ipywidgets=4.0*' \
#     'pandas=0.16*' \
#     'matplotlib=1.4*' \
#     'scipy=0.15*' \
#     'seaborn=0.6*' \
#     'scikit-learn=0.16*' \
#     'scikit-image=0.11*' \
#     'sk-video=1.1*' \
#     'sympy=0.7*' \
#     'cython=0.22*' \
#     'patsy=0.3*' \
#     'statsmodels=0.6*' \
#     'cloudpickle=0.1*' \
#     'dill=0.2*' \
#     'numba=0.20*' \
#     'bokeh=0.9*' \
#     'libgfortran=1.0.0' \
#     'tornado=4.5.3' \
#     'pyzmq' \
#     && conda clean -yt
#
# USER root
#
# RUN /bin/bash -c "source activate python2 && ipython kernel install"

#--------------------------------------
#   Install Python Libraries
#--------------------------------------

# RUN /bin/bash -c "source activate python3 \
#     && conda install -y -c intel \
#     cython \
#     freetype \
#     hdf5 \
#     ipywidgets \
#     jinja2 \
#     libpng \
#     lmdb \
#     matplotlib \
#     mpi4py \
#     numba \
#     numexpr \
#     pandas \
#     protobuf \
#     pytables \
#     requests \
#     scikit-learn \
#     scikit-image \
#     && conda install -y \
#     future \
#     graphviz \
#     hypothesis \
#     pydot \
#     python-nvd3 \
#     sk-video \
#     && conda clean -tipsy"
#

# Merged my scipy installation & intel installation
# RUN /bin/bash -c "source activate python2 \
#     && conda install -y -c intel \
#       cython \
#       freetype \
#       hdf5 \
#       ipywidgets \
#       jinja2 \
#       libpng \
#       lmdb \
#       matplotlib \
#       mpi4py \
#       numba \
#       numexpr \
#       pandas \
#       protobuf \
#       pytables \
#       requests \
#       scikit-learn \
#       scikit-image \
#     && conda install -y \
#       'av' \
#       'bokeh=0.9*' \
#       'cloudpickle=0.1*' \
#       'cython=0.22*' \
#       'dill=0.2*' \
#       'future' \
#       'graphviz' \
#       'hypothesis' \
#       'ipython=4.0*' \
#       'ipywidgets=4.0*' \
#       'libgfortran=1.0.0' \
#       'matplotlib=1.4*' \
#       'numba=0.20*' \
#       'pandas=0.16*' \
#       'patsy=0.3*' \
#       'pydot' \
#       'python-nvd3' \
#       'pyzmq' \
#       'seaborn=0.6*' \
#       'scipy=0.15*' \
#       'scikit-learn=0.16*' \
#       'scikit-image=0.11*' \
#       'sk-video=1.1*' \
#       'sympy=0.7*' \
#       'statsmodels=0.6*' \
#       'tornado=4.5.3' \
#     && conda clean -tipsy"

#--------------------------------------
#   Updates needed for opencv
#--------------------------------------

USER root

RUN apt-get update && apt-get install -y \
  build-essential \
  cmake \
  git \
  pkg-config \
  libgtk-3-dev \
  libavcodec-dev \
  libavformat-dev \
  libswscale-dev \
  libv4l-dev \
  libxvidcore-dev \
  libx264-dev \
  libjpeg-dev \
  libpng-dev \
  libtiff-dev \
  libhdf5-serial-dev \
  gfortran \
  openexr \
  libatlas-base-dev \
  python3-dev \
  python3-numpy \
  libtbb2 \
  libtbb-dev \
  libdc1394-22-dev

#--------------------------------------
# Install OpenCV
#--------------------------------------

# do we need to set these?
# export PATH=/usr/local/cuda/bin:$PATH
# export LPATH=/usr/local/nvidia/lib64:$LPATH
# export LIBRARY_PATH=/usr/local/nvidia:$LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib:$LD_LIBRARY_PATH

# -D WITH_JPEG=ON \
# -D BUILD_JPEG=OFF \
# -D JPEG_LIBRARY=/opt/libjpeg-turbo/lib64/libturbojpeg.so \
# -D JPEG_INCLUDE_DIR=/usr/libjpeg-turbo/include \
# -D BUILD_TIFF=ON \
# -D WITH_TIFF=ON \
# -D BUILD_PNG=ON \
# -D WITH_PNG=ON \

USER $NB_USER
WORKDIR ~/

# try to force use of our custom installation of ffmpeg
RUN conda uninstall -y ffmpeg
# && /bin/bash -c "source activate python2 && conda uninstall -y ffmpeg"

USER root

# used to install v3.4.3, now installing latest 4.1.0
RUN git clone https://github.com/opencv/opencv_contrib.git ~/opencv_contrib \
  && cd ~/opencv_contrib && git checkout tags/4.1.0 \
  && git clone https://github.com/Itseez/opencv.git ~/opencv \
  && cd ~/opencv \
  && git checkout tags/4.1.0 \
  && mkdir build \
  && cd build/ \
  && cmake -D CMAKE_BUILD_TYPE=RELEASE \
  -D PYTHON3_EXECUTABLE:FILEPATH=/opt/conda/bin/python3.6 \
  -D PYTHON3_INCLUDE_DIR:PATH=/opt/conda/include/python3.6m \
  -D PYTHON3_NUMPY_INCLUDE_DIRS:PATH=/opt/conda/lib/python3.6/site-packages/numpy/core/include \
  -D PYTHON3_PACKAGES_PATH:PATH=/opt/conda/lib/python3.6/site-packages \
  -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D BUILD_opencv_dnn=ON \
  -D BUILD_opencv_dnn_modern=ON \
  -D BUILD_LIBPROTOBUF_FROM_SOURCES=ON \
  -D WITH_CUDA=ON \
  -D WITH_CUFFT=ON \
  -D WITH_CUBLAS=ON \
  -D WITH_NVCUVID=ON \
  -D FORCE_VTK=OFF \
  -D WITH_TBB=ON \
  -D WITH_V4L=ON \
  -D WITH_OPENGL=ON \
  -D ENABLE_FAST_MATH=1 \
  -D CUDA_FAST_MATH=1 \
  -D WITH_CUBLAS=1 \
  -D WITH_FFMPEG=1 \
  -D INSTALL_PYTHON_EXAMPLES=ON \
  -D BUILD_EXAMPLES=OFF .. \
  && make -j$(nproc) && make install \
  && ldconfig

RUN rm /opt/conda/lib/libfontconfig.so /opt/conda/lib/libfontconfig.so.1 /opt/conda/lib/libfontconfig.so.1.12.0

#--------------------------------------
#   LIBJPEGTURBO
#--------------------------------------

RUN cd /opt \
  && wget https://downloads.sourceforge.net/libjpeg-turbo/libjpeg-turbo-2.0.0.tar.gz \
  && tar -xzf libjpeg-turbo-2.0.0.tar.gz \
  && rm libjpeg-turbo-2.0.0.tar.gz \
  && cd libjpeg-turbo-2.0.0 \
  && mkdir build \
  && cd build \
  && cmake -DCMAKE_INSTALL_PREFIX=/usr/libjpeg-turbo \
  -DCMAKE_BUILD_TYPE=RELEASE  \
  -DENABLE_STATIC=FALSE       \
  -DCMAKE_INSTALL_DOCDIR=/usr/share/doc/libjpeg-turbo-2.0.0 \
  -DCMAKE_INSTALL_DEFAULT_LIBDIR=lib  \
  .. \
  && make -j"$(nproc)" \
  && make test \
  && make install \
  && mkdir -p /opt/libjpeg-turbo/lib64 \
  && ln -s /usr/libjpeg-turbo/lib/libturbojpeg.so /opt/libjpeg-turbo/lib64/libturbojpeg.so

#--------------------------------------
# PyTurboJPEG
#--------------------------------------

# pip install -U git+git://github.com/loopbio/PyTurboJPEG.git
# RUN conda config --add channels conda-forge \
#   && conda install -c loopbio libjpeg-turbo \
#   && mkdir -p /opt/libjpeg-turbo/lib64 \
#   && ln -s /opt/conda/lib/libjpeg-turbo/lib/libturbojpeg.so /opt/libjpeg-turbo/lib64/libturbojpeg.so \
#   && /bin/bash -c "source activate python2 && conda config --add channels conda-forge \
#     && conda install -c loopbio libjpeg-turbo"

# http://blog.loopbio.com/video-io-2-jpeg-decoding.html
# using loopbio's PyTurbojpeg fork! https://github.com/loopbio/PyTurbojpeg
RUN pip install -U git+git://github.com/loopbio/PyTurboJPEG.git
# \
#  && /bin/bash -c "source activate python2 && pip install -U git+git://github.com/loopbio/PyTurboJPEG.git"

#--------------------------------------
# Utilities for optimizing python code:
#--------------------------------------

USER $NB_USER

# Cupy, PyCuda, Numba
RUN conda install -y -c lukepfister pycuda \
  && pip install cupy \
  && conda install -y numba

#--------------------------------------
# Install Video and Image Utilities
#--------------------------------------

# Installing all of these dependencies together in one RUN command
# to avoid limit on number of layers in docker image
RUN conda install -y -c conda-forge av trackpy ipywidgets \
  && conda install -y filterpy \
  && jupyter nbextension enable --py --sys-prefix widgetsnbextension \
  && pip install --upgrade pip \
  && pip install \
  ffmpeg-python \
  image-attendant \
  imutils \
  imageio \
  https://github.com/soft-matter/pims/archive/master.zip \
  pynvvl-cuda92 \
  sk-video \
  tqdm \
  && cd $HOME \
  && git clone https://github.com/DheerendraRathor/ffprobe3 \
  && cd $HOME/ffprobe3 \
  && python setup.py install \
  && cd $HOME \
  && git clone https://github.com/jfrelinger/cython-munkres-wrapper \
  && cd $HOME/cython-munkres-wrapper \
  && python setup.py install

#--------------------------------------
# Parallel HDF5
#--------------------------------------

# https://drtiresome.com/2016/08/23/build-and-install-mpi-parallel-hdf5-and-h5py-from-source-on-linux/

USER root

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
# RUN pip install mpi4py \
#   && /bin/bash -c "source activate python2 && pip install mpi4py"

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

USER root
RUN apt-get update && apt-get install -y build-essential libcap-dev openssh-client

USER $NB_UID

# plotly + dependencies
RUN easy_install python-prctl
RUN pip install plotly msgpack msgpack-numpy lmdb \
  && pip install --upgrade git+https://github.com/tensorpack/tensorpack.git \
  && conda install -c bioconda cufflinks

RUN pip install wget pathlib addict nvidia-ml-py3 \
  && conda install -y cookiecutter \
  && conda install -c menpo pathlib \
  && conda install -y dask \
  && conda clean -tipsy

#--------------------------------------
# Install Optimized Fast PIL
#--------------------------------------

RUN conda uninstall --force jpeg libtiff -y \
  && pip uninstall -y pillow \
  && conda install -y -c conda-forge libjpeg-turbo \
  && CC="cc -mavx2" pip install --no-cache-dir -U --force-reinstall pillow-simd

#--------------------------------------
# Late Additions
#--------------------------------------

ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

USER root

# RUN jupyter labextension install dask-labextension \
#   && jupyter labextension install @jupyterlab/git \
#   && pip install jupyterlab-git \
#   && jupyter serverextension enable --py jupyterlab_git

USER $NB_UID

#--------------------------------------
#   INSTALL DeepLearning Frameworks
#--------------------------------------

# RUN conda install -y pytorch torchvision cudatoolkit=10.0 -c pytorch \
#   && conda install -y -c fastai fastai fastprogress

# RUN pip install tensorflow-gpu \
#   && pip install mxnet-cu100

#--------------------------------------
# TIME ZONE!
#--------------------------------------

# ENV TZ=America/New_York
# RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#--------------------------------------
#  Late Additions
#--------------------------------------

# modin = faster pandas; https://modin.readthedocs.io/en/latest/
RUN pip install modin

# seems we need to run this build command:
RUN jupyter lab clean && jupyter lab build

#--------------------------------------
# Install R Kernel
#--------------------------------------

USER root

# install R
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    fonts-dejavu \
    unixodbc \
    unixodbc-dev \
    r-cran-rodbc \
    gfortran \
    gcc && \
    rm -rf /var/lib/apt/lists/*

RUN R -e "install.packages(c('arm', 'caret','crayon','devtools','forecast','hexbin','htmltools', 'htmltools', 'htmlwidgets', 'irkernel', 'lme4', 'lmerTest', 'lsmeans','nycflights13','plyr', 'randomforest', 'rcurl', 'reshape2', 'rmarkdown', 'rodbc','rsqlite', 'shiny', 'sparklyr', 'tidyverse', 'unixodbc'), dependencies=TRUE, repos='http://cran.rstudio.com/')"

RUN R -e "install.packages('IRkernel')" && \
    R -e "IRkernel::installspec()"

# RUN fix-permissions /home/$NB_USER

RUN mv $HOME/.local/share/jupyter/kernels/ir* $CONDA_DIR/share/jupyter/kernels/ && \
      rm -rf $HOME/.local && \
      fix-permissions /home/$NB_USER $CONDA_DIR/share/jupyter

RUN R -e "install.packages('lmerTest')"

# python -c "from pymer4.test_install import test_install; test_install()"

# R packages including IRKernel which gets installed globally.
# RUN conda install --quiet --yes \
#     'r-base=3.6.1' \
#     'r-caret=6.0*' \
#     'r-crayon=1.3*' \
#     'r-devtools=2.1*' \
#     'r-forecast=8.7*' \
#     'r-hexbin=1.27*' \
#     'r-htmltools=0.3*' \
#     'r-htmlwidgets=1.3*' \
#     'r-irkernel=1.0*' \
#     'r-lme4' \
#     'r-lmerTest' \
#     'r-lsmeans' \
#     'r-nycflights13=1.0*' \
#     'r-plyr=1.8*' \
#     'r-randomforest=4.6*' \
#     'r-rcurl=1.95*' \
#     'r-reshape2=1.4*' \
#     'r-rmarkdown=1.14*' \
#     'r-rodbc=1.3*' \
#     'r-rsqlite=2.1*' \
#     'r-shiny=1.3*' \
#     'r-sparklyr=1.0*' \
#     'r-tidyverse=1.2*' \
#     'rpy2=2.9*' \
#     'unixodbc=2.3.*' \
#     && \
#     conda clean --all -y && \
#     fix-permissions $CONDA_DIR && \
#     fix-permissions /home/$NB_USER
#
# # Install e1071 R package (dependency of the caret R package)
# # RUN conda install --quiet --yes r-e1071
#
# RUN pip install pymer4

# python -c "from pymer4.test_install import test_install; test_install()"

USER $NB_UID

#--------------------------------------
# Install Julia Kernel
#--------------------------------------

# USER root
#
# # Julia dependencies
# # install Julia packages in /opt/julia instead of $HOME
# ENV JULIA_DEPOT_PATH=/opt/julia
# ENV JULIA_PKGDIR=/opt/julia
# ENV JULIA_VERSION=1.2.0
#
# RUN mkdir /opt/julia-${JULIA_VERSION} && \
#     cd /tmp && \
#     wget -q https://julialang-s3.julialang.org/bin/linux/x64/`echo ${JULIA_VERSION} | cut -d. -f 1,2`/julia-${JULIA_VERSION}-linux-x86_64.tar.gz && \
#     echo "926ced5dec5d726ed0d2919e849ff084a320882fb67ab048385849f9483afc47 *julia-${JULIA_VERSION}-linux-x86_64.tar.gz" | sha256sum -c - && \
#     tar xzf julia-${JULIA_VERSION}-linux-x86_64.tar.gz -C /opt/julia-${JULIA_VERSION} --strip-components=1 && \
#     rm /tmp/julia-${JULIA_VERSION}-linux-x86_64.tar.gz
# RUN ln -fs /opt/julia-*/bin/julia /usr/local/bin/julia
#
# # Show Julia where conda libraries are \
# RUN mkdir /etc/julia && \
#     echo "push!(Libdl.DL_LOAD_PATH, \"$CONDA_DIR/lib\")" >> /etc/julia/juliarc.jl && \
#     # Create JULIA_PKGDIR \
#     mkdir $JULIA_PKGDIR && \
#     chown $NB_USER $JULIA_PKGDIR && \
#     fix-permissions $JULIA_PKGDIR
#
# # Add Julia packages. Only add HDF5 if this is not a test-only build since
# # it takes roughly half the entire build time of all of the images on Travis
# # to add this one package and often causes Travis to timeout.
# #
# # Install IJulia as jovyan and then move the kernelspec out
# # to the system share location. Avoids problems with runtime UID change not
# # taking effect properly on the .local folder in the jovyan home dir.
# RUN julia -e 'import Pkg; Pkg.update()' && \
#     (test $TEST_ONLY_BUILD || julia -e 'import Pkg; Pkg.add("HDF5")') && \
#     julia -e "using Pkg; pkg\"add IJulia\"; pkg\"precompile\"" && \
#     # move kernelspec out of home \
#     mv $HOME/.local/share/jupyter/kernels/julia* $CONDA_DIR/share/jupyter/kernels/ && \
#     chmod -R go+rx $CONDA_DIR/share/jupyter && \
#     rm -rf $HOME/.local && \
#     fix-permissions $JULIA_PKGDIR $CONDA_DIR/share/jupyter

#--------------------------------------
# Notebook Setup
#--------------------------------------

USER root

EXPOSE 8888
WORKDIR $HOME/work

# Configure container startup
ENTRYPOINT ["tini", "-g", "--"]
CMD ["start-notebook.sh"]

# Add local files as late as possible to avoid cache busting
COPY credential-helper.sh /usr/local/bin/
COPY start.sh /usr/local/bin/
COPY start-notebook.sh /usr/local/bin/
COPY start-singleuser.sh /usr/local/bin/
COPY jupyter_notebook_config.py /etc/jupyter/
COPY genpass.py .
RUN fix-permissions /etc/jupyter/

WORKDIR $HOME/work

# Switch back to jovyan to avoid accidental container runs as root
USER $NB_UID
