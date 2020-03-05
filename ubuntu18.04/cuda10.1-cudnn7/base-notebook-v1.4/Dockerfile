# grez72/base-notebook for deep learning
#
# Runs a fully-loaded jupyterlab notebook server with access to the GPU on
# the host machine, running Ubuntu 18.04, CUDA 10.1, CUDNN7, python3.6,
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
# docker build -t grez72/base-notebook:ubuntu18.04-cuda10.1-cudnn7-1.4 https://github.com/grez72/notebooks.git#master:ubuntu18.04/cuda10.1-cudnn7/base-notebook-v1.4
#
# test it:
# docker run -it --rm grez72/base-notebook:ubuntu18.04-cuda10.1-cudnn7-1.4 nvidia-smi
#
# generate a password:
# nvidia-docker run -it --rm grez72/base-notebook:ubuntu18.04-cuda10.1-cudnn7-1.4 python genpass.py -p 'supersecretpassword'
#
# mount it:
# docker run -it --rm --user root grez72/base-notebook:ubuntu18.04-cuda10.1-cudnn7-1.4 bash
#
# run it:
# docker run -it --rm -p 8888:8888 grez72/base-notebook:ubuntu18.04-cuda10.1-cudnn7-1.4
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
  build-essential \
  bzip2 \
  ca-certificates \
  sudo \
  libcap-dev \
  locales \
  fonts-liberation \
  openssh-client \
  rsync \
  tmux \
  wget \
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
# ENV MINICONDA_VERSION 4.5.4
# ENV MINICONDA_CHECKSUM a946ea1d0c4a642ddf0c3a26a18bb16d

ENV MINICONDA_VERSION=4.7.12.1 \
    MINICONDA_CHECKSUM=81c773ff87af5cfac79ab862942ab6b3 \
    CONDA_VERSION=4.7.12

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

# https://github.com/QuantStack/mamba
# install mamba for faster conda installs ("mamba install ...")
RUN conda install --quiet --yes pybind11 libsolv pip && \
  conda install --quiet --yes mamba -c conda-forge && \
  conda clean -tipsy

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
# "conda=${MINICONDA_VERSION}" \
# RUN conda install --quiet --yes \  
#   'av' \
#   'conda-forge::blas=*=openblas' \
#   'ipywidgets=7.2*' \
#   'pandas=0.23*' \
#   'numexpr=2.6*' \
#   'matplotlib=2.2*' \
#   'scipy=1.1*' \
#   'seaborn=0.9*' \
#   'scikit-learn=0.19*' \
#   'scikit-image=0.14*' \
#   'sk-video=1.1*' \
#   'sympy=1.1*' \
#   'cython=0.28*' \
#   'patsy=0.5*' \
#   'statsmodels=0.9*' \
#   'cloudpickle=0.5*' \
#   'dill=0.2*' \
#   'numba=0.38*' \
#   'bokeh=0.12*' \
#   'sqlalchemy=1.2*' \
#   'hdf5=1.10*' \
#   'h5py=2.7*' \
#   'vincent=0.4.*' \
#   'beautifulsoup4=4.6.*' \
#   'protobuf=3.*' \
#   'xlrd'  && \
#   conda remove --quiet --yes --force qt pyqt && \
#   conda clean -tipsy && \
#   npm cache clean --force && \
#   rm -rf $CONDA_DIR/share/jupyter/lab/staging && \
#   rm -rf /home/$NB_USER/.cache/yarn && \
#   rm -rf /home/$NB_USER/.node-gyp && \
#   fix-permissions $CONDA_DIR && \
#   fix-permissions /home/$NB_USER

RUN /bin/bash -c "source activate base && mamba install --quiet --yes \
  av=7.0.0 \
  beautifulsoup4=4.8.* \
  bokeh=0.12* \
  cloudpickle=1.3.* \
  conda-forge::blas=*=openblas \
  cython=0.29* \
  dill=0.3* \
  hdf5=1.10.* \
  h5py=2.10.* \
  ipywidgets=7.5* \
  matplotlib=3.1.* \
  numba=0.48.* \
  numexpr=2.7.* \
  pandas=1.0.* \
  patsy=0.5.* \
  protobuf=3.11.* \
  seaborn=0.10.* \
  scikit-image=0.16.* \
  scikit-learn=0.22.* \
  scipy=1.4.* \
  sk-video=1.1* \
  sqlalchemy=1.3.* \
  statsmodels=0.11.* \
  sympy=1.5.* \
  vincent=0.4.* \
  xlrd && \
  mamba clean --all -f -y && \
  mamba install --quiet --yes dask=2.11.* && \
  mamba clean --all -f -y" && \
  npm cache clean --force && \
  rm -rf $CONDA_DIR/share/jupyter/lab/staging && \
  rm -rf /home/$NB_USER/.cache/yarn && \
  rm -rf /home/$NB_USER/.node-gyp && \
  fix-permissions $CONDA_DIR && \
  fix-permissions /home/$NB_USER

# # jupyter nbextension enable --py widgetsnbextension --sys-prefix && \
# # jupyter labextension install @jupyter-widgets/jupyterlab-manager@^0.38.1 && \
# # jupyter labextension install jupyterlab_bokeh@^0.6.3 && \

# # Install facets which does not have a pip or conda package at the moment
# RUN cd /tmp && \
#   git clone https://github.com/PAIR-code/facets.git && \
#   cd facets && \
#   jupyter nbextension install facets-dist/ --sys-prefix && \
#   cd && \
#   rm -rf /tmp/facets && \
#   fix-permissions $CONDA_DIR && \
#   fix-permissions /home/$NB_USER

# # Import matplotlib the first time to build the font cache.
# ENV XDG_CACHE_HOME /home/$NB_USER/.cache/
# RUN MPLBACKEND=Agg python -c "import matplotlib.pyplot" && \
#   fix-permissions /home/$NB_USER

# #--------------------------------------
# #   Updates needed for opencv
# #--------------------------------------

# USER root

# RUN apt-get update && apt-get install -y \
#   build-essential \
#   cmake \
#   git \
#   pkg-config \
#   libgtk-3-dev \
#   libavcodec-dev \
#   libavformat-dev \
#   libswscale-dev \
#   libv4l-dev \
#   libxvidcore-dev \
#   libx264-dev \
#   libjpeg-dev \
#   libpng-dev \
#   libtiff-dev \
#   libhdf5-serial-dev \
#   gfortran \
#   openexr \
#   libatlas-base-dev \
#   python3-dev \
#   python3-numpy \
#   libtbb2 \
#   libtbb-dev \
#   libdc1394-22-dev

# #--------------------------------------
# # Install OpenCV
# #--------------------------------------

# # do we need to set these?
# # export PATH=/usr/local/cuda/bin:$PATH
# # export LPATH=/usr/local/nvidia/lib64:$LPATH
# # export LIBRARY_PATH=/usr/local/nvidia:$LIBRARY_PATH
# # export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib:$LD_LIBRARY_PATH

# # -D WITH_JPEG=ON \
# # -D BUILD_JPEG=OFF \
# # -D JPEG_LIBRARY=/opt/libjpeg-turbo/lib64/libturbojpeg.so \
# # -D JPEG_INCLUDE_DIR=/usr/libjpeg-turbo/include \
# # -D BUILD_TIFF=ON \
# # -D WITH_TIFF=ON \
# # -D BUILD_PNG=ON \
# # -D WITH_PNG=ON \

# USER $NB_USER
# WORKDIR ~/

# # try to force use of our custom installation of ffmpeg and glib
# RUN /bin/bash -c "source activate base && mamba uninstall -y ffmpeg glib"
# # RUN conda uninstall -y ffmpeg
# # && /bin/bash -c "source activate python2 && conda uninstall -y ffmpeg"

# USER root

# # used to install v3.4.3, now installing latest 4.2.0
# RUN git clone https://github.com/opencv/opencv_contrib.git ~/opencv_contrib \
#   && cd ~/opencv_contrib && git checkout tags/4.2.0 \
#   && git clone https://github.com/Itseez/opencv.git ~/opencv \
#   && cd ~/opencv \
#   && git checkout tags/4.2.0 \
#   && mkdir build \
#   && cd build/ \
#   && cmake -D CMAKE_BUILD_TYPE=RELEASE \
#   -D PYTHON3_EXECUTABLE:FILEPATH=/opt/conda/bin/python3.6 \
#   -D PYTHON3_INCLUDE_DIR:PATH=/opt/conda/include/python3.6m \
#   -D PYTHON3_NUMPY_INCLUDE_DIRS:PATH=/opt/conda/lib/python3.6/site-packages/numpy/core/include \
#   -D PYTHON3_PACKAGES_PATH:PATH=/opt/conda/lib/python3.6/site-packages \
#   -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
#   -D CMAKE_INSTALL_PREFIX=/usr/local \
#   -D CUDA_ARCH_BIN=5.3,6.0,6.1,7.0,7.5 \
#   -D CUDA_ARCH_PTX=7.5 \
#   -D BUILD_opencv_dnn=ON \
#   -D BUILD_opencv_dnn_modern=ON \
#   -D BUILD_LIBPROTOBUF_FROM_SOURCES=ON \
#   -D WITH_CUDA=ON \
#   -D WITH_CUFFT=ON \
#   -D WITH_CUBLAS=ON \
#   -D WITH_NVCUVID=ON \
#   -D FORCE_VTK=OFF \
#   -D WITH_TBB=ON \
#   -D WITH_V4L=ON \
#   -D WITH_OPENGL=ON \
#   -D ENABLE_FAST_MATH=1 \
#   -D CUDA_FAST_MATH=1 \
#   -D WITH_CUBLAS=1 \
#   -D WITH_FFMPEG=1 \
#   -D INSTALL_PYTHON_EXAMPLES=ON \
#   -D BUILD_EXAMPLES=OFF .. \
#   && make -j$(nproc) && make install \
#   && ldconfig

# RUN rm /opt/conda/lib/libfontconfig.so /opt/conda/lib/libfontconfig.so.1 /opt/conda/lib/libfontconfig.so.1.12.0

# #--------------------------------------
# #   INSTALL NVIDIA RAPIDS
# #--------------------------------------

# RUN /bin/bash -c "source activate base \
#   && mamba install --quiet --yes -c rapidsai -c nvidia -c conda-forge -c defaults cuxfilter=0.12 python=3.6 cudatoolkit=10.1 \
#   && mamba install --quiet --yes rapids=0.12 python=3.6 cudatoolkit=10.1"

# #--------------------------------------
# #   INSTALL PIP PACKAGES
# #--------------------------------------

# # addict = turn dict into dotdict
# # modin = faster pandas; https://modin.readthedocs.io/en/latest/
# # trackpy = particle tracking; http://soft-matter.github.io/trackpy/v0.4.2/
# # filterpy = kalman filtering: https://github.com/rlabbe/filterpy

# RUN pip install \
#   addict \  
#   filterpy \
#   natsort \
#   nvidia-ml-py3 \
#   modin \
#   pathlib \
#   pynvvl-cuda92 \
#   trackpy \
#   wget 

# #--------------------------------------
# #   LIBJPEGTURBO
# #--------------------------------------

# RUN cd /opt \
#   && wget https://downloads.sourceforge.net/libjpeg-turbo/libjpeg-turbo-2.0.0.tar.gz \
#   && tar -xzf libjpeg-turbo-2.0.0.tar.gz \
#   && rm libjpeg-turbo-2.0.0.tar.gz \
#   && cd libjpeg-turbo-2.0.0 \
#   && mkdir build \
#   && cd build \
#   && cmake -DCMAKE_INSTALL_PREFIX=/usr/libjpeg-turbo \
#   -DCMAKE_BUILD_TYPE=RELEASE  \
#   -DENABLE_STATIC=FALSE       \
#   -DCMAKE_INSTALL_DOCDIR=/usr/share/doc/libjpeg-turbo-2.0.0 \
#   -DCMAKE_INSTALL_DEFAULT_LIBDIR=lib  \
#   .. \
#   && make -j"$(nproc)" \
#   && make test \
#   && make install \
#   && mkdir -p /opt/libjpeg-turbo/lib64 \
#   && ln -s /usr/libjpeg-turbo/lib/libturbojpeg.so /opt/libjpeg-turbo/lib64/libturbojpeg.so

# #--------------------------------------
# # PyTurboJPEG
# #--------------------------------------

# # pip install -U git+git://github.com/loopbio/PyTurboJPEG.git
# # RUN conda config --add channels conda-forge \
# #   && conda install -c loopbio libjpeg-turbo \
# #   && mkdir -p /opt/libjpeg-turbo/lib64 \
# #   && ln -s /opt/conda/lib/libjpeg-turbo/lib/libturbojpeg.so /opt/libjpeg-turbo/lib64/libturbojpeg.so \
# #   && /bin/bash -c "source activate python2 && conda config --add channels conda-forge \
# #     && conda install -c loopbio libjpeg-turbo"

# # http://blog.loopbio.com/video-io-2-jpeg-decoding.html
# # using loopbio's PyTurbojpeg fork! https://github.com/loopbio/PyTurbojpeg
# RUN pip install -U git+git://github.com/loopbio/PyTurboJPEG.git
# # \
# #  && /bin/bash -c "source activate python2 && pip install -U git+git://github.com/loopbio/PyTurboJPEG.git"

# #--------------------------------------
# # Utilities for optimizing python code:
# # ...skipping in this iteration
# #--------------------------------------

# # USER $NB_USER

# # Cupy, PyCuda, Numba
# # RUN /bin/bash -c "source activate base && mamba install --quiet --yes \
# #   lukepfister pycuda && \
# #   pip install cupy && \
# #   conda install -y numba && \
# #   mamba clean --all -f -y"

# # RUN conda install -y -c lukepfister pycuda \
# #   && pip install cupy \
# #   && conda install -y numba

# #--------------------------------------
# # Install Video and Image Utilities
# #--------------------------------------

# # Installing all of these dependencies together in one RUN command
# # to avoid limit on number of layers in docker image
# # RUN conda install -y -c conda-forge av trackpy ipywidgets \
# #   && conda install -y filterpy \
# RUN /bin/bash -c "source activate base && mamba install --quiet --yes \
#   -c conda-forge av ipywidgets && \
#   mamba clean --all -f -y" && \
#   jupyter nbextension enable --py --sys-prefix widgetsnbextension \
#   && pip install --upgrade pip \
#   && pip install \
#   ffmpeg-python \
#   image-attendant \
#   imutils \
#   imageio \
#   https://github.com/soft-matter/pims/archive/master.zip \
#   cupy \
#   sk-video \
#   tqdm \
#   && cd $HOME \
#   && git clone https://github.com/DheerendraRathor/ffprobe3 \
#   && cd $HOME/ffprobe3 \
#   && python setup.py install \
#   && cd $HOME \
#   && git clone https://github.com/jfrelinger/cython-munkres-wrapper \
#   && cd $HOME/cython-munkres-wrapper \
#   && python setup.py install

# #--------------------------------------
# # Install Additional Packages
# #--------------------------------------

# # plotly + dependencies
# RUN easy_install python-prctl \
#   && pip install plotly msgpack msgpack-numpy lmdb \
#   && pip install --upgrade git+https://github.com/tensorpack/tensorpack.git \
#   && /bin/bash -c "source activate base && mamba install --quiet --yes bioconda cufflinks"

# RUN /bin/bash -c "source activate base && mamba install --quiet --yes cookiecutter \
#     && mamba install --quiet --yes -c menpo pathlib \
#     && mamba install --quiet --yes cookiecutter \
#     && mamba clean --all -f -y"

# #--------------------------------------
# # Install Optimized Fast PIL
# #--------------------------------------

# RUN /bin/bash -c "source activate base && mamba uninstall --force jpeg libtiff -y \
#   && pip uninstall -y pillow \
#   && mamba install -y -c conda-forge libjpeg-turbo \
#   && mamba clean --all -f -y \
#   && CC='cc -mavx2' pip install --no-cache-dir -U --force-reinstall pillow-simd"

# # RUN conda uninstall --force jpeg libtiff -y \
# #   && pip uninstall -y pillow \
# #   && conda install -y -c conda-forge libjpeg-turbo \
# #   && CC="cc -mavx2" pip install --no-cache-dir -U --force-reinstall pillow-simd

# #--------------------------------------
# # Late Additions
# #--------------------------------------

# ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

# #--------------------------------------
# #   INSTALL DeepLearning Frameworks
# #--------------------------------------

# # RUN conda install -y pytorch torchvision cudatoolkit=10.0 -c pytorch \
# #   && conda install -y -c fastai fastai fastprogress

# # RUN pip install tensorflow-gpu \
# #   && pip install mxnet-cu100

# #--------------------------------------
# #  Late Additions
# #--------------------------------------

# # seems we need to run this build command:
# RUN jupyter lab clean && jupyter lab build

# #--------------------------------------
# # Install R Kernel
# #--------------------------------------

# # USER root

# # # install R
# # RUN apt-get update && \
# #     apt-get install -y --no-install-recommends \
# #     fonts-dejavu \
# #     unixodbc \
# #     unixodbc-dev \
# #     r-cran-rodbc \
# #     gfortran \
# #     gcc && \
# #     rm -rf /var/lib/apt/lists/*

# # RUN R -e "install.packages(c('arm', 'caret','crayon','devtools','forecast','hexbin','htmltools', 'htmltools', 'htmlwidgets', 'irkernel', 'lme4', 'lmerTest', 'lsmeans','nycflights13','plyr', 'randomforest', 'rcurl', 'reshape2', 'rmarkdown', 'rodbc','rsqlite', 'shiny', 'sparklyr', 'tidyverse', 'unixodbc'), dependencies=TRUE, repos='http://cran.rstudio.com/')"

# # RUN R -e "install.packages('IRkernel')" && \
# #     R -e "IRkernel::installspec()"

# # # RUN fix-permissions /home/$NB_USER

# # RUN mv $HOME/.local/share/jupyter/kernels/ir* $CONDA_DIR/share/jupyter/kernels/ && \
# #       rm -rf $HOME/.local && \
# #       fix-permissions /home/$NB_USER $CONDA_DIR/share/jupyter

# # RUN R -e "install.packages('lmerTest')" && \
# #       pip install tzlocal pymer4

# # python -c "from pymer4.test_install import test_install; test_install()"

# # R packages including IRKernel which gets installed globally.
# # RUN conda install --quiet --yes \
# #     'r-base=3.6.1' \
# #     'r-caret=6.0*' \
# #     'r-crayon=1.3*' \
# #     'r-devtools=2.1*' \
# #     'r-forecast=8.7*' \
# #     'r-hexbin=1.27*' \
# #     'r-htmltools=0.3*' \
# #     'r-htmlwidgets=1.3*' \
# #     'r-irkernel=1.0*' \
# #     'r-lme4' \
# #     'r-lmerTest' \
# #     'r-lsmeans' \
# #     'r-nycflights13=1.0*' \
# #     'r-plyr=1.8*' \
# #     'r-randomforest=4.6*' \
# #     'r-rcurl=1.95*' \
# #     'r-reshape2=1.4*' \
# #     'r-rmarkdown=1.14*' \
# #     'r-rodbc=1.3*' \
# #     'r-rsqlite=2.1*' \
# #     'r-shiny=1.3*' \
# #     'r-sparklyr=1.0*' \
# #     'r-tidyverse=1.2*' \
# #     'rpy2=2.9*' \
# #     'unixodbc=2.3.*' \
# #     && \
# #     conda clean --all -y && \
# #     fix-permissions $CONDA_DIR && \
# #     fix-permissions /home/$NB_USER
# #
# # # Install e1071 R package (dependency of the caret R package)
# # # RUN conda install --quiet --yes r-e1071
# #
# # RUN pip install pymer4

# # python -c "from pymer4.test_install import test_install; test_install()"

# # USER $NB_UID

# #--------------------------------------
# # Install Julia Kernel
# #--------------------------------------

# # USER root
# #
# # # Julia dependencies
# # # install Julia packages in /opt/julia instead of $HOME
# # ENV JULIA_DEPOT_PATH=/opt/julia
# # ENV JULIA_PKGDIR=/opt/julia
# # ENV JULIA_VERSION=1.2.0
# #
# # RUN mkdir /opt/julia-${JULIA_VERSION} && \
# #     cd /tmp && \
# #     wget -q https://julialang-s3.julialang.org/bin/linux/x64/`echo ${JULIA_VERSION} | cut -d. -f 1,2`/julia-${JULIA_VERSION}-linux-x86_64.tar.gz && \
# #     echo "926ced5dec5d726ed0d2919e849ff084a320882fb67ab048385849f9483afc47 *julia-${JULIA_VERSION}-linux-x86_64.tar.gz" | sha256sum -c - && \
# #     tar xzf julia-${JULIA_VERSION}-linux-x86_64.tar.gz -C /opt/julia-${JULIA_VERSION} --strip-components=1 && \
# #     rm /tmp/julia-${JULIA_VERSION}-linux-x86_64.tar.gz
# # RUN ln -fs /opt/julia-*/bin/julia /usr/local/bin/julia
# #
# # # Show Julia where conda libraries are \
# # RUN mkdir /etc/julia && \
# #     echo "push!(Libdl.DL_LOAD_PATH, \"$CONDA_DIR/lib\")" >> /etc/julia/juliarc.jl && \
# #     # Create JULIA_PKGDIR \
# #     mkdir $JULIA_PKGDIR && \
# #     chown $NB_USER $JULIA_PKGDIR && \
# #     fix-permissions $JULIA_PKGDIR
# #
# # # Add Julia packages. Only add HDF5 if this is not a test-only build since
# # # it takes roughly half the entire build time of all of the images on Travis
# # # to add this one package and often causes Travis to timeout.
# # #
# # # Install IJulia as jovyan and then move the kernelspec out
# # # to the system share location. Avoids problems with runtime UID change not
# # # taking effect properly on the .local folder in the jovyan home dir.
# # RUN julia -e 'import Pkg; Pkg.update()' && \
# #     (test $TEST_ONLY_BUILD || julia -e 'import Pkg; Pkg.add("HDF5")') && \
# #     julia -e "using Pkg; pkg\"add IJulia\"; pkg\"precompile\"" && \
# #     # move kernelspec out of home \
# #     mv $HOME/.local/share/jupyter/kernels/julia* $CONDA_DIR/share/jupyter/kernels/ && \
# #     chmod -R go+rx $CONDA_DIR/share/jupyter && \
# #     rm -rf $HOME/.local && \
# #     fix-permissions $JULIA_PKGDIR $CONDA_DIR/share/jupyter

# #--------------------------------------
# # Install specific pyzmq + tornado 
# # otherwise we get "500" error in jupyterhub
# #--------------------------------------

# RUN /bin/bash -c "source activate base \
#   && mamba install -y pyzmq=18.1.0 tornado=6.0.3 \
#   && mamba clean --all -f -y"

# #--------------------------------------
# # Notebook Setup
# #--------------------------------------

# USER root

# EXPOSE 8888
# WORKDIR $HOME/work

# # Configure container startup
# ENTRYPOINT ["tini", "-g", "--"]
# CMD ["start-notebook.sh"]

# # Add local files as late as possible to avoid cache busting
# COPY credential-helper.sh /usr/local/bin/
# COPY start.sh /usr/local/bin/
# COPY start-notebook.sh /usr/local/bin/
# COPY start-singleuser.sh /usr/local/bin/
# COPY jupyter_notebook_config.py /etc/jupyter/
# COPY genpass.py .
# RUN fix-permissions /etc/jupyter/

# WORKDIR $HOME/work

# # Switch back to jovyan to avoid accidental container runs as root
# USER $NB_UID
