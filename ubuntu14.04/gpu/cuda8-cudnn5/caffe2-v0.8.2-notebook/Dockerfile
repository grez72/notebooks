# grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-caffe2-v0.8.1-notebook
#
# Caffe2 jupyter notebook with gpu suppport (nvidia CUDA cuDNN).
#
# sudo nvidia-docker build -t grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-caffe2-v0.8.1-notebook https://github.com/grez72/notebooks.git#master:ubuntu14.04/gpu/cuda8-cudnn5/caffe2-v0.8.1-notebook
# nvidia-docker run -it --rm -p 8888:8888 grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-caffe2-v0.8.1-notebook
# nvidia-docker run -it grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-caffe2-v0.8.1-notebook python -m caffe2.python.operator_test.relu_op_test

FROM grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-base-notebook

MAINTAINER George Alvarez <gre72@gmail.com, alvarez@wjh.harvard.edu>

USER root

ARG CAFFE2_VERSION=0.8.0

# ==============================================================
#  nvidia cuda8-cudnn5
# ==============================================================

# USER root
#
# LABEL com.nvidia.volumes.needed="nvidia_driver"
#
# RUN NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
#     NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
#     apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/7fa2af80.pub && \
#     apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +2 > cudasign.pub && \
#     echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub && \
#     echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64 /" > /etc/apt/sources.list.d/cuda.list
#
# ENV CUDA_VERSION 8.0.61
# LABEL com.nvidia.cuda.version="${CUDA_VERSION}"
#
# ENV CUDA_PKG_VERSION 8-0=$CUDA_VERSION-1
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         cuda-nvrtc-$CUDA_PKG_VERSION \
#         cuda-nvgraph-$CUDA_PKG_VERSION \
#         cuda-cusolver-$CUDA_PKG_VERSION \
#         cuda-cublas-8-0=8.0.61.1-1 \
#         cuda-cufft-$CUDA_PKG_VERSION \
#         cuda-curand-$CUDA_PKG_VERSION \
#         cuda-cusparse-$CUDA_PKG_VERSION \
#         cuda-npp-$CUDA_PKG_VERSION \
#         cuda-cudart-$CUDA_PKG_VERSION && \
#     ln -s cuda-8.0 /usr/local/cuda && \
#     rm -rf /var/lib/apt/lists/*
#
# RUN echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/cuda.conf && \
#     ldconfig
#
# RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
#     echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf
#
# ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
# ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
#
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         cuda-core-$CUDA_PKG_VERSION \
#         cuda-misc-headers-$CUDA_PKG_VERSION \
#         cuda-command-line-tools-$CUDA_PKG_VERSION \
#         cuda-nvrtc-dev-$CUDA_PKG_VERSION \
#         cuda-nvml-dev-$CUDA_PKG_VERSION \
#         cuda-nvgraph-dev-$CUDA_PKG_VERSION \
#         cuda-cusolver-dev-$CUDA_PKG_VERSION \
#         cuda-cublas-dev-8-0=8.0.61.1-1 \
#         cuda-cufft-dev-$CUDA_PKG_VERSION \
#         cuda-curand-dev-$CUDA_PKG_VERSION \
#         cuda-cusparse-dev-$CUDA_PKG_VERSION \
#         cuda-npp-dev-$CUDA_PKG_VERSION \
#         cuda-cudart-dev-$CUDA_PKG_VERSION \
#         cuda-driver-dev-$CUDA_PKG_VERSION && \
#     rm -rf /var/lib/apt/lists/*
#
# ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs:${LIBRARY_PATH}
#
# RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
#
# ENV CUDNN_VERSION 5.1.10
# LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
#
# RUN apt-get update && apt-get install -y --no-install-recommends \
#             libcudnn5=$CUDNN_VERSION-1+cuda8.0 \
#             libcudnn5-dev=$CUDNN_VERSION-1+cuda8.0 && \
#     rm -rf /var/lib/apt/lists/*

# ==============================================================
#  caffe2 install with gpu support
# ==============================================================

RUN /bin/bash -c "source activate python2 && apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python-dev \
    python-numpy \
    python-pip \
    python-pydot \
    python-setuptools \
    python-scipy \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-glog-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*"

# RUN /bin/bash -c "source activate python2 && git clone --recursive https://github.com/caffe2/caffe2.git && \
#     pip install --upgrade pip && \
#     pip install numpy protobuf"

########## OPTIONAL DEPENDENCIES ################

RUN /bin/bash -c "source activate python2 && apt-get update && apt-get install -y --no-install-recommends \
    libgflags-dev \
    libgoogle-glog-dev \
    libgflags2 \
    libgtest-dev \
    libiomp-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libopenmpi-dev \
    libsnappy-dev \
    openmpi-bin \
    openmpi-doc \
    python-numpy \
    python-pydot \
    python-setuptools \
    python-scipy \
    wget"

RUN /bin/bash -c "source activate python2 && pip install --upgrade pip && pip install \
    flask \
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
    tornado"

########## INSTALLATION STEPS ###################
# RUN git clone --branch v0.8.0 --recursive https://github.com/caffe2/caffe2.git
# RUN cd caffe2 \
#   && make -j"$(nproc)" \
#   && cd build \
#   && make -j"$(nproc)" install

RUN /bin/bash -c "source activate python2 && cd caffe2 && make -j'$(nproc)' && cd build && make -j'$(nproc)' install"
RUN /bin/bash -c "source activate python2 && pip install future"

ENV PYTHONPATH /usr/local
ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
# RUN python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"

########## TROUBLESHOOTING ######################
# this is added due to some issues that were found

# libdc1394 error
# for some reason once opencv is installed you may get errors with libdc1394
# http://stackoverflow.com/questions/12689304/ctypes-error-libdc1394-error-failed-to-initialize-libdc1394
# RUN ln /dev/null /dev/raw1394
# But that's not persistent http://stackoverflow.com/questions/31768441/how-to-persist-ln-in-docker-with-ubuntu
# CMD sh -c 'ln -s /dev/null /dev/raw1394'
# or when instantiating the container use:
# --device /dev/null:/dev/raw1394

# FROM grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-base-notebook
#
# MAINTAINER George Alvarez <gre72@gmail.com, alvarez@wjh.harvard.edu>
#
# ARG CAFFE2_VERSION=v0.7.0
#
# # ==============================================================
# #  INSTALL Caffe2
# # ==============================================================
#
# USER root
# WORKDIR $HOME
#
# # Required Dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#       build-essential \
#       cmake \
#       git \
#       libgoogle-glog-dev \
#       libprotobuf-dev \
#       protobuf-compiler \
#       python-dev \
#       python-pip
# RUN pip install --upgrade pip && \
#     pip install numpy protobuf && \
#     pip2 install numpy protobuf
#
# RUN conda install libgcc
#
# # Caffe2 works best with openmpi 1.8.5 or above (which has cuda support).
# # If you do not need openmpi, skip this step.
# RUN cd /tmp && \
#   wget http://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-1.10.0.tar.gz && \
#   tar xzvf openmpi-1.10.0.tar.gz && \
#   cd /tmp/openmpi-1.10.0 && \
#   ./configure --with-cuda --with-threads && \
#   make && make install && \
#   cd / && \
#   rm -rf /tmp/openmpi-1.10.0 && \
#   rm /tmp/openmpi-1.10.0.tar.gz
#
# # Caffe2 requires zeromq 4.0 or above, manually install.
# # If you do not need zeromq, skip this step.
# # RUN apt-get install -q -y autoconf libtool
# # RUN mkdir /tmp/zeromq-build && \
# #   cd /tmp/zeromq-build && \
# #   wget https://github.com/zeromq/zeromq4-1/archive/v4.1.3.tar.gz && \
# #   tar xzvf v4.1.3.tar.gz --strip 1 && \
# #   ./autogen.sh && \
# #   ./configure --without-libsodium && \
# #   make && make install && \
# #   cd / && \
# #   rm -rf /tmp/zeromq-build
#
# # pip self upgrade
# RUN pip install --upgrade pip
#
# # Python dependencies
# RUN pip install \
#       matplotlib \
#       numpy \
#       protobuf
#
# RUN pip2 install \
#       matplotlib \
#       numpy \
#       protobuf
#
# # GPU Support CUDA 8.0
# # RUN apt-get update && sudo apt-get install wget -y --no-install-recommends
# # RUN wget "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_8.0.61-1_amd64.deb"
# # RUN dpkg -i cuda-repo-ubuntu1404_8.0.61-1_amd64.deb
# # RUN apt-get update
# # RUN apt-get install cuda
#
# # cuDNN 5.1
# # RUN CUDNN_URL="http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz"
# # RUN wget ${CUDNN_URL}
# # RUN tar -xzf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local
# # RUN rm cudnn-8.0-linux-x64-v5.1.tgz && sudo ldconfig
#
# # Optional Dependeincies
# RUN apt-get install -y --no-install-recommends \
#       gfortran \
#       libgflags-dev \
#       libgoogle-glog-dev \
#       libgflags2 \
#       libgtest-dev \
#       libiomp-dev \
#       libleveldb-dev \
#       liblmdb-dev \
#       libopencv-dev \
#       libopenmpi-dev \
#       libsnappy-dev \
#       openmpi-bin \
#       openmpi-doc \
#       python-numpy \
#       python-pydot \
#       python-setuptools \
#       python-scipy \
#       wget
#
# RUN pip install \
#       flask \
#       future \
#       graphviz \
#       hypothesis \
#       jupyter \
#       matplotlib \
#       pydot \
#       python-nvd3 \
#       pyyaml \
#       requests \
#       scikit-image \
#       scipy \
#       setuptools \
#       six \
#       tornado
#
# RUN pip2 install \
#       flask \
#       future \
#       graphviz \
#       hypothesis \
#       jupyter \
#       matplotlib \
#       pydot \
#       python-nvd3 \
#       pyyaml \
#       requests \
#       scipy \
#       setuptools \
#       six \
#       tornado
#
# # This is intentional. scikit-image has to be after scipy.
# RUN pip install \
#       scikit-image
#
# RUN pip2 install \
#       scikit-image
#
# # clone it
# RUN git clone --recursive https://github.com/caffe2/caffe2.git
# WORKDIR $HOME/caffe2
# RUN git checkout tags/${CAFFE2_VERSION}
# RUN make && cd build && make install
# # RUN mkdir build
# # WORKDIR $HOME/caffe2/build
# # # RUN sed -i -e 's/VERSION_GREATER 4.8/VERSION_GREATER 5.0/g' CMakeLists.txt
# # RUN cmake .. && \
# #     make
#
# ENV PYTHONPATH /usr/local
# ENV PYTHONPATH $PYTHONPATH:$HOME/caffe2/build
# ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
# ENV LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
# RUN python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
#
# # Test your gpu
# # RUN python -m caffe2.python.operator_test.relu_op_test
#
# # Make sure these ENV variables are set
# # echo $PYTHONPATH
# # ENV PYTHONPATH=/usr/local:$PYTHONPATH
# # ENV PYTHONPATH=$PYTHONPATH:$HOME/caffe2/build
# # echo $LD_LIBRARY_PATH
# # ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
#
# # this is added due to some issues that were found
#
# # ==============================================================
# #  troubleshooting: downgrade anaconda's hdf5 to resolve error
# # ==============================================================
#
# # libdc1394 error
# # for some reason once opencv is installed you may get errors with libdc1394
# # http://stackoverflow.com/questions/12689304/ctypes-error-libdc1394-error-failed-to-initialize-libdc1394
# # RUN ln /dev/null /dev/raw1394
# # But that's not persistent http://stackoverflow.com/questions/31768441/how-to-persist-ln-in-docker-with-ubuntu
# # CMD sh -c 'ln -s /dev/null /dev/raw1394'
# # or when instantiating the container use:
# # --device /dev/null:/dev/raw1394
#
# # ==============================================================
# #  troubleshooting: downgrade anaconda's hdf5 to resolve error
# # ==============================================================
#
# # conda install -c anaconda hdf5=1.8.17
# # RUN conda remove hdf5 -y && conda install hdf5=1.8.17 -y
# # RUN /bin/bash -c "source activate python2 && conda remove hdf5 -y && conda install hdf5=1.8.17 -y"
#
# # ==============================================================
# #  set workdir and user for jupyter notebook
# # ==============================================================
#
# WORKDIR /home/$NB_USER/work/
# USER $NB_USER
