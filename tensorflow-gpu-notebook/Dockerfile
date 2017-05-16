# nvidia-cuda-notebook
# Modified nvidia-cuda-devel to work as single-user notebook with jupyterhub
# Starting with the ubuntu14.04 cuda8 runtime and development Dockerfiles
# https://gitlab.com/nvidia/cuda/blob/ubuntu14.04/8.0/runtime/cudnn5/Dockerfile
# https://gitlab.com/nvidia/cuda/blob/ubuntu14.04/8.0/devel/Dockerfile
# nvidia-docker build -t grez72/cuda-tensorflow-gpu-notebook https://github.com/grez72/notebooks.git#master:tensorflow-gpu-notebook
# nvidia-docker run -it --rm -p 8888:8888 grez72/cuda-tensorflow-gpu-notebook

FROM jupyter/scipy-notebook

LABEL com.nvidia.volumes.needed="nvidia_driver"

USER root

RUN NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
    NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/7fa2af80.pub && \
    apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +2 > cudasign.pub && \
    echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub && \
    echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64 /" > /etc/apt/sources.list.d/cuda.list

ENV CUDA_VERSION 8.0.61
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"

ENV CUDA_PKG_VERSION 8-0=$CUDA_VERSION-1
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-core-$CUDA_PKG_VERSION \
        cuda-misc-headers-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION \
        cuda-nvrtc-dev-$CUDA_PKG_VERSION \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        cuda-nvgraph-dev-$CUDA_PKG_VERSION \
        cuda-cusolver-dev-$CUDA_PKG_VERSION \
        cuda-cublas-dev-8-0=8.0.61.1-1 \
        cuda-cufft-dev-$CUDA_PKG_VERSION \
        cuda-curand-dev-$CUDA_PKG_VERSION \
        cuda-cusparse-dev-$CUDA_PKG_VERSION \
        cuda-npp-dev-$CUDA_PKG_VERSION \
        cuda-cudart-dev-$CUDA_PKG_VERSION \
        cuda-driver-dev-$CUDA_PKG_VERSION && \
    ln -s cuda-8.0 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

RUN echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/cuda.conf && \
    ldconfig

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

USER root

RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/7fa2af80.pub
RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

ENV CUDNN_VERSION 5.1.10
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update
RUN apt-get install --force-yes -y --no-install-recommends cuda-samples-$CUDA_PKG_VERSION
RUN apt-get install --force-yes -y --no-install-recommends libcudnn5=5.1.10-1+cuda8.0
RUN apt-get install --force-yes -y --no-install-recommends libcudnn5-dev=5.1.10-1+cuda8.0
RUN rm -rf /var/lib/apt/lists/*

# make copies of cuDNN so tensorflow can find them when installing from source
RUN cp /usr/lib/x86_64-linux-gnu/libcudnn* /usr/local/cuda/lib64
RUN cp /usr/include/cudnn.h /usr/local/cuda/include

# test devices
WORKDIR /usr/local/cuda/samples/1_Utilities/deviceQuery

RUN make

# CMD defines the default command to be run in the container
# CMD is overridden by supplying a command + arguments to
# `docker run`, e.g. `nvcc --version` or `bash`
# CMD ./deviceQuery

WORKDIR /home/$NB_USER/work

# Install Python 3 Tensorflow
# RUN conda install --quiet --yes 'tensorflow-gpu=1.1*'

# Install Python 2 Tensorflow
# RUN conda install --quiet --yes -n python2 'tensorflow-gpu=1.1*'

# ==============================================================
#  INSTALL TENSORFLOW FROM SOURCE (python3)
# ==============================================================

# install JAVA8
RUN echo "deb http://ppa.launchpad.net/webupd8team/java/ubuntu trusty main" | tee /etc/apt/sources.list.d/webupd8team-java.list
RUN echo "deb-src http://ppa.launchpad.net/webupd8team/java/ubuntu trusty main" | tee -a /etc/apt/sources.list.d/webupd8team-java.list
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys EEA14886
RUN apt-get update
RUN echo oracle-java8-installer shared/accepted-oracle-license-v1-1 select true | sudo /usr/bin/debconf-set-selections
RUN apt-get install -y oracle-java8-installer

# install other dependencies
RUN apt-get update
RUN apt-get install -y curl git python-dev python3-dev build-essential python-pip python3-pip python-virtualenv swig python-wheel libcurl3-dev

# install BAZEL
RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
RUN curl https://storage.googleapis.com/bazel-apt/doc/apt-key.pub.gpg | sudo apt-key add -
RUN apt-get update
RUN apt-get install -y bazel
RUN apt-get upgrade -y bazel

# install tensorflow
RUN git clone https://github.com/tensorflow/tensorflow
WORKDIR /home/$NB_USER/work/tensorflow

# Next we configure the installation
# Normally we call ./configure, and do an interactive setup
# The interactive setup sets environment variables used by bazel.
# Currently, that series of questions is as follows:

# Here we set those configuration variables:
# https://github.com/tensorflow/tensorflow/blob/master/configure

ENV PYTHON_BIN_PATH=/opt/conda/bin/python
ENV PYTHON_LIB_PATH=/opt/conda/lib/python3.5/site-packages
ENV TF_NEED_CUDA=1
ENV TF_NEED_GCP=0
ENV TF_NEED_OPENCL=0
ENV TF_CUDA_CLANG=0
ENV CUDA_TOOLKIT_PATH=/usr/local/cuda
ENV CUDNN_INSTALL_PATH=/usr/local/cuda-8.0
ENV TF_CUDA_VERSION=8.0
ENV TF_CUDA_COMPUTE_CAPABILITIES=3.5,5.2,6.1
ENV GCC_HOST_COMPILER_PATH=/usr/bin/gcc
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++
ENV TF_CUDNN_VERSION=5
ENV CUDA_TOOLKIT_PATH=/usr/local/cuda
ENV CUDNN_INSTALL_PATH=$CUDA_TOOLKIT_PATH
ENV CC_OPT_FLAGS=-march=native

# Do you wish to build TensorFlow with MKL support? [y/N]
ENV TF_NEED_MKL=0

# "Do you wish to use jemalloc as the malloc implementation? [Y/n] "
ENV TF_NEED_JEMALLOC=1

# "Do you wish to build TensorFlow with Hadoop File System support? [y/N] "
ENV TF_NEED_HDFS=0

# "Do you wish to build TensorFlow with the XLA just-in-time compiler (experimental)? [y/N] "
ENV TF_ENABLE_XLA=0

# Do you wish to build TensorFlow with VERBS support? [y/N]
ENV TF_NEED_VERBS=0

# Now when we run ./configure, there should be no unset variables, and so it should just proceed.
# However, tensorflow is continuously adding new options, and so if RUN ./configure errors,
# you might comment out everthing from here down, then build and run the image. Then use $docker ps
# to get your image id, and $ docker exec -it <id> bash to mount it, and run $ ./configure within
# the container.  Whatever questions you answer will tell you which variables need to be set.
# You can reference https://github.com/tensorflow/tensorflow/blob/master/configure to determine
# what the actual variable names should be, which you can then set in this file, and re-build the image,
# uncommenting everything from here down so tensorflow will be built.
RUN ./configure

RUN bazel build -c opt --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
RUN bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
RUN pip3 install /tmp/tensorflow_pkg/*

# ==============================================================
#  build tensorflow for python2
# ==============================================================

# python2 dependencies
RUN apt-get update && sudo apt-get install -y \
        build-essential \
        curl \
        libcurl3-dev \
        git \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-dev \
        python-numpy \
        python-pip \
        software-properties-common \
        swig \
        zip \
        zlib1g-dev

# update config
# /usr/bin/python2
# /usr/local/lib/python2.7/dist-packages
ENV PYTHON_BIN_PATH=/usr/bin/python2
ENV PYTHON_LIB_PATH=/usr/local/lib/python2.7/dist-packages
RUN ./configure

RUN bazel build -c opt --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
RUN bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
RUN pip2 install /tmp/tensorflow_pkg/*27*

# cleanup
WORKDIR /home/$NB_USER/work/
RUN rm -R tensorflow
USER $NB_USER
