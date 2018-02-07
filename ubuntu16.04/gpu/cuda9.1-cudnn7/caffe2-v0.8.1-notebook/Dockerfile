# grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-caffe2-v0.8.1-notebook
#
# Caffe2 jupyter notebook with gpu suppport (nvidia CUDA cuDNN).
#
# sudo nvidia-docker build -t grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-caffe2-v0.8.1-notebook https://github.com/grez72/notebooks.git#master:ubuntu16.04/gpu/cuda9.1-cudnn7/caffe2-v0.8.1-notebook
# nvidia-docker run -it --rm --shm-size=2G -p 8888:8888 grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-caffe2-v0.8.1-notebook
#
# tests
# nvidia-docker run -it --rm --shm-size=2G grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-caffe2-v0.8.1-notebook python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
# nvidia-docker run -it --rm --shm-size=2G grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-caffe2-v0.8.1-notebook python -m caffe2.python.operator_test.relu_op_test
# nvidia-docker run -it --rm --shm-size=2G grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-caffe2-v0.8.1-notebook python -c 'from caffe2.python import workspace; print(workspace.NumCudaDevices())'
# nvidia-docker run -it --rm --shm-size=2G grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-caffe2-v0.8.1-notebook bash

FROM grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-base-notebook

# FROM ubuntu:16.04
# FROM jupyter/base-notebook
# FROM jupyter/minimal-notebook
# FROM jupyter/scipy-notebook

MAINTAINER George Alvarez <gre72@gmail.com, alvarez@wjh.harvard.edu>

USER root
WORKDIR $HOME

# =======================================================
#   INSTALL NVIDIA CUDA 9.1
# =======================================================

# RUN NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
#     NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
#     apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
#     apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +5 > cudasign.pub && \
#     echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub && \
#     echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
#     echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
#
# ENV CUDA_VERSION 9.1.85
#
# ENV CUDA_PKG_VERSION 9-1=$CUDA_VERSION-1
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         cuda-cudart-$CUDA_PKG_VERSION && \
#     ln -s cuda-9.1 /usr/local/cuda && \
#     rm -rf /var/lib/apt/lists/*
#
# # nvidia-docker 1.0
# LABEL com.nvidia.volumes.needed="nvidia_driver"
# LABEL com.nvidia.cuda.version="${CUDA_VERSION}"
#
# RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
#     echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf
#
# ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
# ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
#
# # nvidia-container-runtime
# ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
# ENV NVIDIA_REQUIRE_CUDA "cuda>=9.1"
#
# # https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.1/runtime/Dockerfile
# ENV NCCL_VERSION 2.1.4
#
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         cuda-libraries-$CUDA_PKG_VERSION \
#         libnccl2=$NCCL_VERSION-1+cuda9.1 && \
#     rm -rf /var/lib/apt/lists/*
#
# # https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.1/devel/Dockerfile
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         cuda-libraries-dev-$CUDA_PKG_VERSION \
#         cuda-nvml-dev-$CUDA_PKG_VERSION \
#         cuda-minimal-build-$CUDA_PKG_VERSION \
#         cuda-command-line-tools-$CUDA_PKG_VERSION \
#         libnccl-dev=$NCCL_VERSION-1+cuda9.1 && \
#     rm -rf /var/lib/apt/lists/*
#
# ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs:${LIBRARY_PATH}

# =======================================================
#   INSTALL NVIDIA cuDNN 7
# =======================================================

# https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.1/devel/cudnn7/Dockerfile
# ENV CUDNN_VERSION 7.0.5.15
# LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
#
# RUN apt-get update && apt-get install -y --no-install-recommends \
#             libcudnn7=$CUDNN_VERSION-1+cuda9.1 \
#             libcudnn7-dev=$CUDNN_VERSION-1+cuda9.1 && \
#     rm -rf /var/lib/apt/lists/*

# =======================================================
#   scipy notebook
# =======================================================

# # libav-tools for matplotlib anim
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends libav-tools && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*
#
# USER $NB_USER
#
# # Install Python 3 packages
# # Remove pyqt and qt pulled in for matplotlib since we're only ever going to
# # use notebook-friendly backends in these images
# RUN conda install --quiet --yes \
#     'nomkl' \
#     'ipywidgets=7.1*' \
#     'pandas=0.19*' \
#     'numexpr=2.6*' \
#     'matplotlib=2.0*' \
#     'scipy=0.19*' \
#     'seaborn=0.7*' \
#     'scikit-learn=0.18*' \
#     'scikit-image=0.12*' \
#     'sympy=1.0*' \
#     'cython=0.25*' \
#     'patsy=0.4*' \
#     'statsmodels=0.8*' \
#     'cloudpickle=0.2*' \
#     'dill=0.2*' \
#     'numba=0.31*' \
#     'bokeh=0.12*' \
#     'sqlalchemy=1.1*' \
#     'hdf5=1.8.17' \
#     'h5py=2.6*' \
#     'vincent=0.4.*' \
#     'beautifulsoup4=4.5.*' \
#     'protobuf=3.*' \
#     'xlrd'  && \
#     conda remove --quiet --yes --force qt pyqt && \
#     conda clean -tipsy && \
#     # Activate ipywidgets extension in the environment that runs the notebook server
#     jupyter nbextension enable --py widgetsnbextension --sys-prefix && \
#     # Also activate ipywidgets extension for JupyterLab
#     jupyter labextension install @jupyter-widgets/jupyterlab-manager@^0.33.1 && \
#     npm cache clean && \
#     rm -rf $CONDA_DIR/share/jupyter/lab/staging && \
#     fix-permissions $CONDA_DIR

RUN conda uninstall --quiet --yes protobuf

# =======================================================
#   INSTALL caffe2 install with gpu support
# =======================================================

# USER root
#
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     cmake \
#     git \
#     libgflags-dev \
#     libgoogle-glog-dev \
#     libgtest-dev \
#     libiomp-dev \
#     libleveldb-dev \
#     liblmdb-dev \
#     libopencv-dev \
#     libopenmpi-dev \
#     libprotobuf-dev \
#     libsnappy-dev \
#     openmpi-bin \
#     openmpi-doc \
#     protobuf-compiler \
#     python-dev \
#     python-numpy \
#     python-pip \
#     python-pydot \
#     python-setuptools \
#     python-scipy \
#     wget \
#     && rm -rf /var/lib/apt/lists/*
#
# # Install required packages
# # mkl isn't actually required, but is really recommended for Anaconda
# RUN conda install -y \
#     future \
#     gflags \
#     glog \
#     lmdb \
#     mkl \
#     mkl-include \
#     numpy \
#     # opencv \
#     protobuf \
#     snappy \
#     six
#
# # Clone Caffe2
#
# WORKDIR $HOME
# RUN git clone --recursive https://github.com/caffe2/caffe2.git
# RUN cd caffe2 \
#   && mkdir build && cd build \
#   # && cmake .. \
#   && cmake -DCMAKE_PREFIX_PATH=/opt/conda/envs/python2 -DCMAKE_INSTALL_PREFIX=/opt/conda/envs/python2 .. \
#   && make -j$(nproc) install


# Make Caffe2 in a separate folder to avoid polluting the Caffe2 source tree.
# This can be anywhere you want it
# RUN mkdir build && cd build
#
# # Add flags to this command to control which packages you want to use. The
# # options to use are at the top of CMakeLists.txt in the topmost Caffe2
# # directory
# RUN cmake ..
# # cmake -DUSE_CUDA=OFF -DUSE_LEVELDB=OFF -DCMAKE_PREFIX_PATH=~/anaconda2/envs/my_caffe2_env -DCMAKE_INSTALL_PREFIX=~/anaconda2/envs/my_caffe2_env ..
# RUN make -j$(nproc) install

USER root
WORKDIR $HOME

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgflags-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libiomp-dev \
    libleveldb-dev \
    liblmdb-dev \
    # libopencv-dev \
    libopenmpi-dev \
    libprotobuf-dev \
    libsnappy-dev \
    openmpi-bin \
    openmpi-doc \
    protobuf-compiler \
    python-dev \
    python-numpy \
    python-pip \
    python-pydot \
    python-setuptools \
    python-scipy \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    flask \
    future \
    graphviz \
    hypothesis \
    jupyter \
    matplotlib \
    numpy \
    protobuf \
    pydot \
    python-nvd3 \
    pyyaml \
    requests \
    scikit-image \
    scipy \
    setuptools \
    six \
    tornado

########## INSTALLATION STEPS ###################
RUN git clone --branch master --recursive https://github.com/caffe2/caffe2.git
RUN cd caffe2 && mkdir build && cd build \
    && cmake .. \
    # -DUSE_NATIVE_ARCH=On \
    -DCUDA_ARCH_NAME=Manual \
    -DCUDA_ARCH_BIN="35 52 60 61" \
    -DCUDA_ARCH_PTX="61" \
    #-DUSE_NNPACK=OFF \
    #-DUSE_ROCKSDB=OFF \
    # -DDEPRIORITIZE_ANACONDA=ON \
    && make -j"$(nproc)" install \
    && ldconfig \
    && make clean
RUN cd $HOME/caffe2 \
    && python setup.py install
    # && cd .. \
    # && rm -rf build

ENV PYTHONPATH /usr/local

# FROM grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-minimal-notebook
# # FROM grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-base-notebook
#
# MAINTAINER George Alvarez <gre72@gmail.com, alvarez@wjh.harvard.edu>
#
# ARG CAFFE2_VERSION=v0.8.0
#
# USER root
# WORKDIR $HOME

# ==============================================================
#  install Caffe2 with gpu support
# ==============================================================

# Install required packages
# mkl isn't actually required, but is really recommended for Anaconda
# RUN /bin/bash -c 'source activate python2 \
#   && conda install -y mkl -c intel \
#   && conda install -y \
#     future \
#     gflags \
#     glog \
#     lmdb \
#     mkl \
#     mkl-include \
#     numpy \
#     protobuf \
#     snappy \
#     six'

# clone and build caffe2
# RUN /bin/bash -c "source activate python2 \
#   && git clone --recursive https://github.com/caffe2/caffe2.git \
#   && git checkout $CAFFE2_VERSION
#   && cd caffe2 \
#   && cd $HOME/caffe2 \
#   && make -j$(nproc) USE_NATIVE_ARCH=1 \
#   && cd build \
#   && make -j$(nproc) install \
#   && pip install future"

# Make Caffe2 in a separate folder to avoid polluting the Caffe2 source tree.
# This can be anywhere you want it
# RUN git clone --recursive https://github.com/caffe2/caffe2.git && cd caffe2
# RUN git checkout v0.8.0
# RUN mkdir build && cd build
#
# # Add flags to this command to control which packages you want to use. The
# # options to use are at the top of CMakeLists.txt in the topmost Caffe2
# # directory
# RUN cmake -DUSE_CUDA=ON -DUSE_MKL=ON ..
# RUN make -j$(nproc) install

# RUN /bin/bash -c "source activate python2 && apt-get update && apt-get install -y --no-install-recommends \
#     python-dev \
#     python-pip \
#     git \
#     build-essential \
#     cmake \
#     libprotobuf-dev \
#     protobuf-compiler \
#     libgoogle-glog-dev"
#
# USER $NB_USER
# # WORKDIR $HOME/work/networks
# RUN /bin/bash -c "source activate python2 && git clone --recursive https://github.com/caffe2/caffe2.git && \
#     pip install --upgrade pip && \
#     pip install numpy protobuf"
#
# ########## OPTIONAL DEPENDENCIES ################
#
# USER root
#
# RUN /bin/bash -c "source activate python2 && apt-get install -y --no-install-recommends \
#     libgflags-dev \
#     libgoogle-glog-dev \
#     libgflags2 \
#     libgtest-dev \
#     libiomp-dev \
#     libleveldb-dev \
#     liblmdb-dev \
#     libopencv-dev \
#     libopenmpi-dev \
#     libsnappy-dev \
#     openmpi-bin \
#     openmpi-doc \
#     python-numpy \
#     python-pydot \
#     python-setuptools \
#     python-scipy \
#     wget"
#
# RUN /bin/bash -c "source activate python2 && pip install \
#     flask \
#     graphviz \
#     hypothesis \
#     jupyter \
#     matplotlib \
#     pydot \
#     python-nvd3 \
#     pyyaml \
#     requests \
#     scikit-image \
#     scipy \
#     setuptools \
#     tornado"
#
# ########## INSTALLATION STEPS ###################
# RUN /bin/bash -c 'source activate python2 && cd caffe2 && make -j"$(nproc)" && cd build && make -j"$(nproc)" install'
# RUN /bin/bash -c "source activate python2 && pip install future && pip install future --upgrade"
#
# # export PYTHONPATH=/usr/local
# # export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
# # export LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
#
# ENV PYTHONPATH /usr/local
# ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
# ENV LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH

# test installation
# python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
# test gpu
# python -m caffe2.python.operator_test.relu_op_test
# python -c 'from caffe2.python import workspace; print(workspace.NumCudaDevices())'

# ==============================================================
#  make python2 the default
# ==============================================================

# RUN echo 'export PATH="/opt/conda/envs/python2/bin:$PATH"' > ~/.bashrc \
#   && /bin/bash -c "source ~/.bashrc"

# # ==============================================================
# #  caffe2 dependencies
# # ==============================================================
#
# # required
# RUN /bin/bash -c "source activate python2 && apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     cmake \
#     git \
#     python-dev \
#     python-numpy \
#     python-pip \
#     python-pydot \
#     python-setuptools \
#     python-scipy \
#     libprotobuf-dev \
#     protobuf-compiler \
#     libgoogle-glog-dev \
#     wget \
#     && rm -rf /var/lib/apt/lists/*"
#
# # optional
# RUN /bin/bash -c "source activate python2 && apt-get update && apt-get install -y --no-install-recommends \
#     libgflags-dev \
#     libgoogle-glog-dev \
#     libgflags-dev \
#     libgtest-dev \
#     libiomp-dev \
#     libleveldb-dev \
#     liblmdb-dev \
#     libopencv-dev \
#     libopenmpi-dev \
#     libsnappy-dev \
#     openmpi-bin \
#     openmpi-doc \
#     python-numpy \
#     python-pydot \
#     python-setuptools \
#     python-scipy \
#     wget"
#
# RUN /bin/bash -c "source activate python2 && pip install --upgrade pip && pip install \
#     flask \
#     graphviz \
#     hypothesis \
#     jupyter \
#     matplotlib \
#     numpy \
#     protobuf \
#     pydot \
#     python-nvd3 \
#     pyyaml \
#     requests \
#     scikit-image \
#     scipy \
#     setuptools \
#     tornado"
#
# # Caffe2 requires zeromq 4.0 or above, manually install.
# # If you do not need zeromq, skip this step.
# RUN apt-get install -q -y autoconf libtool
# RUN mkdir /tmp/zeromq-build && \
#   cd /tmp/zeromq-build && \
#   wget https://github.com/zeromq/zeromq4-1/archive/v4.1.3.tar.gz && \
#   tar xzvf v4.1.3.tar.gz --strip 1 && \
#   ./autogen.sh && \
#   ./configure --without-libsodium && \
#   make && make install && \
#   cd / && \
#   rm -rf /tmp/zeromq-build
#
# # update pip, numpy, and install mkl
# RUN /bin/bash -c "source activate python2 \
#   && pip install --upgrade pip \
#   && pip install numpy --upgrade \
#   && conda install -y mkl -c intel"
#
# # ==============================================================
# #  INSTALL CAFFE2
# # ==============================================================
#
# WORKDIR $HOME
#
# RUN /bin/bash -c "source activate python2 && cd caffe2 && make -j'$(nproc)' && cd build && make -j'$(nproc)' install"
# RUN /bin/bash -c "source activate python2 && pip install future"
#
# # RUN git clone --recursive https://github.com/caffe2/caffe2.git
#
# # currently, going to branch v0.8.1 leads to compilation error
# # so we'll just stay with the master branch:
# RUN
#
#   && /bin/bash -c "source activate python2 \
#   && git clone --recursive https://github.com/caffe2/caffe2.git \
#   && cd caffe2 \
#   && cd $HOME/caffe2 \
#   && make -j$(nproc) USE_NATIVE_ARCH=1 \
#   && cd build \
#   && make -j$(nproc) install \
#   && pip install future"
#
# ENV PYTHONPATH /usr/local:$PYTHONPATH
# ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
# ENV LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
# # RUN python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
#
# # ==============================================================
# #  set workdir and user for jupyter notebook
# # ==============================================================
#
# WORKDIR /home/$NB_USER/work/
# USER $NB_USER
