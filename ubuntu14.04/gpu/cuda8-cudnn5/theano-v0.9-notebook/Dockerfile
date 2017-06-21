# grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-theano-v0.9-notebook
#
# Theano jupyter notebook with gpu suppport (nvidia CUDA cuDNN).
#
# sudo nvidia-docker build -t grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-theano-v0.9-notebook https://github.com/grez72/notebooks.git#master:ubuntu14.04/gpu/cuda8-cudnn5/theano-v0.9-notebook
# nvidia-docker run -it --rm -p 8888:8888 grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-theano-v0.9-notebook

FROM grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-base-notebook

MAINTAINER George Alvarez <gre72@gmail.com, alvarez@wjh.harvard.edu>

# using theano 0.9.0rc1 because of this Lasagne error: "ImportError: cannot import name downsample"
# ARG THEANO_VERSION=rel-0.9.0
# ARG THEANO_VERSION=rel-0.9.0rc1
ARG THEANO_VERSION=0.10.0.dev1
ARG KERAS_VERSION=2.0.4
ARG LASAGNE_VERSION=0.2.dev1

USER root

WORKDIR $HOME

# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
# pip2 install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# ==============================================================
#  INSTALL THEANO
# ==============================================================

# https://github.com/Kaixhin/dockerfiles/blob/master/cuda-theano/cuda_v8.0/Dockerfile

RUN pip install --upgrade pip

# Install git, wget, python-dev, pip, BLAS + LAPACK and other dependencies
RUN apt-get update && apt-get install -y \
    gfortran \
    git \
    wget \
    liblapack-dev \
    libopenblas-dev \
    python-dev \
    python-pip \
    python-nose \
    python-numpy \
    python-scipy

# Set CUDA_ROOT
ENV CUDA_ROOT /usr/local/cuda/bin

# Install CMake 3
RUN cd /root && wget http://www.cmake.org/files/v3.8/cmake-3.8.1.tar.gz && \
  tar -xf cmake-3.8.1.tar.gz && cd cmake-3.8.1 && \
  ./configure && \
  make -j "$(nproc)" && \
  make install

# Install Cython
RUN pip install Cython

# Clone libgpuarray repo and move into it
RUN cd /root && git clone https://github.com/Theano/libgpuarray.git && cd libgpuarray && \
# Make and move into build directory
  mkdir Build && cd Build && \
# CMake
  cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr && \
# Make
  make -j"$(nproc)" && \
  make install

# Install pygpu
RUN cd /root/libgpuarray && \
  python setup.py build_ext -L /usr/lib -I /usr/include && \
  python setup.py install

# Install bleeding-edge Theano
# RUN pip install --upgrade pip
# RUN pip install --upgrade six
# RUN pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Set up .theanorc for CUDA
# RUN echo "[global]\ndevice=cuda\nfloatX=float32\noptimizer_including=cudnn\n[lib]\ncnmem=0.1\n[nvcc]\nfastmath=True" > /root/.theanorc

# Install Theano and set up Theano config (.theanorc) for CUDA and OpenBLAS
RUN pip --no-cache-dir install git+git://github.com/Theano/Theano.git@${THEANO_VERSION} && \
	\
	echo "[global]\ndevice=gpu\nfloatX=float32\noptimizer_including=cudnn\nmode=FAST_RUN \
		\n[lib]\ncnmem=0.95 \
		\n[nvcc]\nfastmath=True \
		\n[blas]\nldflag = -L/usr/lib/openblas-base -lopenblas \
		\n[DebugMode]\ncheck_finite=1" \
	> /root/.theanorc

RUN pip2 --no-cache-dir install git+git://github.com/Theano/Theano.git@${THEANO_VERSION} && \
	\
	echo "[global]\ndevice=gpu\nfloatX=float32\noptimizer_including=cudnn\nmode=FAST_RUN \
		\n[lib]\ncnmem=0.95 \
		\n[nvcc]\nfastmath=True \
		\n[blas]\nldflag = -L/usr/lib/openblas-base -lopenblas \
		\n[DebugMode]\ncheck_finite=1" \
	> /root/.theanorc

# Install Keras (pip for python3, pip2 for python2)
RUN pip --no-cache-dir install git+git://github.com/fchollet/keras.git@${KERAS_VERSION}
RUN pip2 --no-cache-dir install git+git://github.com/fchollet/keras.git@${KERAS_VERSION}
COPY keras.json $HOME/.keras/keras.json
# COPY keras.json ./opt/conda/envs/python2/lib/python2.7/site-packages/keras/keras.json
# COPY keras.json ./opt/conda/lib/python3.5/site-packages/keras/keras.json

# Install Lasagne (pip for python3, pip2 for python2)
# RUN pip --no-cache-dir install git+git://github.com/Lasagne/Lasagne.git@${LASAGNE_VERSION}
# RUN pip2 --no-cache-dir install git+git://github.com/Lasagne/Lasagne.git@${LASAGNE_VERSION}
RUN pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
RUN pip2 install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip

# Install NoLearn
RUN pip install -r https://raw.githubusercontent.com/dnouri/nolearn/master/requirements.txt
RUN pip install git+https://github.com/dnouri/nolearn.git@master#egg=nolearn==0.7.git

RUN pip2 install -r https://raw.githubusercontent.com/dnouri/nolearn/master/requirements.txt
RUN pip2 install git+https://github.com/dnouri/nolearn.git@master#egg=nolearn==0.7.git

# ==============================================================
#  downgrade anaconda's hdf5 to resolve error
# ==============================================================

# conda install -c anaconda hdf5=1.8.17
RUN conda remove hdf5 -y && conda install hdf5=1.8.17 -y
RUN /bin/bash -c "source activate python2 && conda remove hdf5 -y && conda install hdf5=1.8.17 -y"

# ==============================================================
#  set workdir and user for jupyter notebook
# ==============================================================

WORKDIR /home/$NB_USER/work/
USER $NB_USER
