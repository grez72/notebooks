# grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-tensorflow-v1.2-notebook
#
# Tensorflow jupyternotebook with gpu suppport (nvidia CUDA cuDNN),
# and optional Keras or TFLearn wrappers.
#
# sudo nvidia-docker build -t grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-tensorflow-v1.2-notebook https://github.com/grez72/notebooks.git#master:ubuntu14.04/gpu/cuda8-cudnn5/tensorflow-v1.2-notebook
# nvidia-docker run -it --rm -p 8888:8888 grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-tensorflow-v1.2-notebook

FROM grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-base-notebook

MAINTAINER George Alvarez <gre72@gmail.com, alvarez@wjh.harvard.edu>

ARG TENSORFLOW_VERSION=1.2
ARG KERAS_VERSION=2.0.4
ARG TFLEARN_VERSION=0.3.1
ARG DEFAULT_PYTHON_VERSION=3.5
ARG BAZEL_VERSION=0.5.4

USER root

WORKDIR /home/$NB_USER/work

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
RUN apt-get install -y curl git python-dev python3-dev build-essential python-pip python3-pip python-virtualenv swig python-wheel libcurl3-dev unzip

# install BAZEL
WORKDIR $HOME
RUN wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh \
    && chmod +x bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh \
    && ./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh \
    && bazel

# RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
# RUN curl https://storage.googleapis.com/bazel-apt/doc/apt-key.pub.gpg | sudo apt-key add -
# RUN apt-get update
# RUN wget https://github.com/bazelbuild/bazel/releases/download/0.5.4/bazel-0.5.4-installer-linux-x86_64.sh
# RUN chmod +x bazel-0.5.4-installer-linux-x86_64.sh
# RUN ./bazel-0.5.4-installer-linux-x86_64.sh
# RUN bazel
# RUN cp /usr/local/lib/bazel/bin/bazel-complete.bash /etc/bash_completion.d/bazel-complete.sh
# RUN /bin/bash -c "source /etc/bash_completion.d/bazel-complete.sh"
# RUN apt-get install -y bazel
# RUN apt-get upgrade -y bazel
# RUN bazel

# install tensorflow
WORKDIR $HOME
RUN git clone https://github.com/tensorflow/tensorflow
WORKDIR $HOME/tensorflow
RUN git checkout remotes/origin/r${TENSORFLOW_VERSION}

# Next we configure the installation
# Normally we call ./configure, and do an interactive setup
# The interactive setup sets environment variables used by bazel.
# Currently, that series of questions is as follows:

# Here we set those configuration variables:
# https://github.com/tensorflow/tensorflow/blob/master/configure

ENV PYTHON_BIN_PATH=/opt/conda/bin/python
ENV PYTHON_LIB_PATH=/opt/conda/lib/python${DEFAULT_PYTHON_VERSION}/site-packages
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

# Do you wish to build TensorFlow with MPI support? [y/N]
ENV TF_NEED_MPI=0

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
RUN rm -R tensorflow

# ==============================================================
#  install keras and tensorflow
# ==============================================================

# install keras
# RUN pip install keras==${KERAS_VERSION}
RUN pip install keras==${KERAS_VERSION}
RUN pip2 install keras==${KERAS_VERSION}
COPY keras.json $HOME/.keras/keras.json

# install tflearn
# RUN pip install git+https://github.com/tflearn/tflearn.git
# RUN pip2 install git+https://github.com/tflearn/tflearn.git
RUN pip install tflearn==${TFLEARN_VERSION}
RUN pip2 install tflearn==${TFLEARN_VERSION}

# ==============================================================
#  downgrade anaconda's hdf5 to resolve error
# ==============================================================

# conda install -c anaconda hdf5=1.8.17
RUN conda remove hdf5 -y && conda install hdf5=1.8.17 -y
RUN /bin/bash -c "source activate python2 && conda remove hdf5 -y && conda install hdf5=1.8.17 -y"

# cleanup
WORKDIR /home/$NB_USER/work/
USER $NB_USER
