# grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-tensorflow-v1.9-notebook
#
# Tensorflow jupyternotebook with gpu suppport (nvidia CUDA cuDNN),
# and optional TFLearn wrappers.
#
# sudo nvidia-docker build -t grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-tensorflow-v1.9-notebook https://github.com/grez72/notebooks.git#master:ubuntu16.04/gpu/cuda9.1-cudnn7/tensorflow-v1.9-notebook
# nvidia-docker run -it --rm --shm-size=2G -p 8888:8888 grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-tensorflow-v1.9-notebook

FROM grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-minimal-notebook

MAINTAINER George Alvarez <gre72@gmail.com, alvarez@wjh.harvard.edu>

ARG TENSORFLOW_VERSION=1.9
ARG TFLEARN_VERSION=0.3.2
ARG DEFAULT_PYTHON_VERSION=3.6
ARG BAZEL_VERSION=0.10.0
ARG CUDA_VERSION=9.1
ARG CUDNN_VERSION=7

USER root
WORKDIR $HOME

# ==============================================================
#  Upgrades Before Installing Tensorflow
# ==============================================================

RUN pip install --upgrade pip setuptools \
  && pip install numpy --upgrade \
  && conda update conda \
  && conda update setuptools \
  && conda install -y mkl -c intel

RUN /bin/bash -c "source activate python2 \
  && pip install --upgrade pip setuptools \
  && pip install numpy --upgrade --ignore-installed \
  && conda update -n base conda \
  && conda update setuptools \
  && conda install -y mkl -c intel"
#
# # ==============================================================
# #  INSTALL TENSORFLOW DEPENDENCIES
# # ==============================================================
#
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         build-essential \
#         curl \
#         libfreetype6-dev \
#         libpng12-dev \
#         libzmq3-dev \
#         pkg-config \
#         python \
#         python-dev \
#         rsync \
#         software-properties-common \
#         unzip \
#         && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*
#
# # ==============================================================
# #  INSTALL TENSORFLOW FROM SOURCE (python3)
# # ==============================================================
#
# # install JAVA8
# RUN echo "deb http://ppa.launchpad.net/webupd8team/java/ubuntu trusty main" | tee /etc/apt/sources.list.d/webupd8team-java.list
# RUN echo "deb-src http://ppa.launchpad.net/webupd8team/java/ubuntu trusty main" | tee -a /etc/apt/sources.list.d/webupd8team-java.list
# RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys EEA14886
# RUN apt-get update
# RUN echo oracle-java8-installer shared/accepted-oracle-license-v1-1 select true | sudo /usr/bin/debconf-set-selections
# RUN apt-get install -y oracle-java8-installer
#
# # install other dependencies
# RUN apt-get update
# RUN apt-get install -y curl git python-dev python3-dev build-essential python-pip python3-pip python-virtualenv swig python-wheel libcurl3-dev
#
# # install BAZEL
# WORKDIR $HOME
# RUN wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh \
#     && chmod +x bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh \
#     && ./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh \
#     && bazel
#
# # install tensorflow
# WORKDIR $HOME
# RUN git clone https://github.com/tensorflow/tensorflow
# WORKDIR $HOME/tensorflow
# # RUN git checkout remotes/origin/r${TENSORFLOW_VERSION}
#
# # Next we configure the installation
# # Normally we call ./configure, and do an interactive setup
# # The interactive setup sets environment variables used by bazel.
# # Currently, that series of questions is as follows:
#
# # Here we set those configuration variables:
# # https://github.com/tensorflow/tensorflow/blob/master/configure
#
# ENV PYTHON_BIN_PATH=/opt/conda/bin/python
# ENV PYTHON_LIB_PATH=/opt/conda/lib/python${DEFAULT_PYTHON_VERSION}/site-packages
# ENV TF_NEED_CUDA=1
# ENV TF_NEED_GCP=0
# ENV TF_NEED_OPENCL=0
# ENV TF_CUDA_CLANG=0
# ENV CUDA_TOOLKIT_PATH=/usr/local/cuda
# ENV CUDNN_INSTALL_PATH=/usr/local/cuda-${CUDA_VERSION}
# ENV TF_CUDA_VERSION=${CUDA_VERSION}
# ENV TF_CUDA_COMPUTE_CAPABILITIES=3.5,5.2,6.1
# ENV GCC_HOST_COMPILER_PATH=/usr/bin/gcc
# ENV CC=/usr/bin/gcc
# ENV CXX=/usr/bin/g++
# ENV TF_CUDNN_VERSION=${CUDNN_VERSION}
# ENV CUDA_TOOLKIT_PATH=/usr/local/cuda
# ENV CUDNN_INSTALL_PATH=$CUDA_TOOLKIT_PATH
# ENV CC_OPT_FLAGS=-march=native
#
# # Do you wish to build TensorFlow with MKL support? [y/N]
# ENV TF_NEED_MKL=1
#
# # "Do you wish to use jemalloc as the malloc implementation? [Y/n] "
# ENV TF_NEED_JEMALLOC=1
#
# # "Do you wish to build TensorFlow with Hadoop File System support? [y/N] "
# ENV TF_NEED_HDFS=0
#
# # "Do you wish to build TensorFlow with the XLA just-in-time compiler (experimental)? [y/N] "
# ENV TF_ENABLE_XLA=0
#
# # Do you wish to build TensorFlow with VERBS support? [y/N]
# ENV TF_NEED_VERBS=0
#
# # Do you wish to build TensorFlow with MPI support? [y/N]
# ENV TF_NEED_MPI=0
#
# # Do you wish to build TensorFlow with Amazon S3 File System support? [Y/n]:
# ENV TF_NEED_S3=0
#
# # Do you wish to build TensorFlow with GDR support?
# ENV TF_NEED_GDR=0
#
# # Do you wish to build TensorFlow with Apache Kafka Platform support?
# ENV TF_NEED_KAFKA=0
#
# # Do you wish to build TensorFlow with OpenCL SYCL support?
# ENV TF_NEED_OPENCL_SYCL=0
#
# # Do you wish to build TensorFlow with TensorRT support?
# ENV TF_NEED_TENSORRT=0
#
# # Would you like to interactively configure ./WORKSPACE for Android builds?
# ENV TF_SET_ANDROID_WORKSPACE=0
#
# # https://github.com/tensorflow/tensorflow/blob/8c1f3b3fcdc5f897dbce3b7315c1b7000069f9c4/configure.py
# # Now when we run ./configure, there should be no unset variables, and so it should just proceed.
# # However, tensorflow is continuously adding new options, and so if RUN ./configure errors,
# # you might comment out everthing from here down, then build and run the image. Then use $docker ps
# # to get your image id, and $ docker exec -it <id> bash to mount it, and run $ ./configure within
# # the container.  Whatever questions you answer will tell you which variables need to be set.
# # You can reference https://github.com/tensorflow/tensorflow/blob/master/configure to determine
# # what the actual variable names should be, which you can then set in this file, and re-build the image,
# # uncommenting everything from here down so tensorflow will be built.
# RUN ./configure
#
# RUN bazel build -c opt --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
# RUN bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
# RUN pip install /tmp/tensorflow_pkg/*
#
# # ==============================================================
# #  build tensorflow for python2
# # ==============================================================
#
# # python2 dependencies
# RUN apt-get update && sudo apt-get install -y \
#         build-essential \
#         curl \
#         libcurl3-dev \
#         git \
#         libfreetype6-dev \
#         libpng12-dev \
#         libzmq3-dev \
#         pkg-config \
#         python-dev \
#         python-numpy \
#         python-pip \
#         software-properties-common \
#         swig \
#         zip \
#         zlib1g-dev
#
# # update config
# # /usr/bin/python2
# # /usr/local/lib/python2.7/dist-packages
# ENV PYTHON_BIN_PATH=/opt/conda/envs/python2/bin/python
# ENV PYTHON_LIB_PATH=/opt/conda/envs/python2/lib/python2.7/site-packages
# RUN ./configure
#
# RUN bazel build --verbose_failures -c opt --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
# RUN bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
# RUN /bin/bash -c "source activate python2 && pip install /tmp/tensorflow_pkg/*27* \
#   && rm -R tensorflow"
#
# # ==============================================================
# #  install tflearn
# # ==============================================================
#
# # Kearas is now build into tensorflow
# # install keras
# # RUN pip install keras==${KERAS_VERSION}
# # RUN pip install keras==${KERAS_VERSION}
# # RUN pip2 install keras==${KERAS_VERSION}
# # COPY keras.json $HOME/.keras/keras.json
#
# # install tflearn
# RUN pip install tflearn==${TFLEARN_VERSION} \
#   && /bin/bash -c "source activate python2 \
#   && pip install tflearn==${TFLEARN_VERSION}"
#
# # ==============================================================
# #  downgrade anaconda's hdf5 to resolve error
# # ==============================================================
#
# # conda install -c anaconda hdf5=1.8.17
# # RUN conda remove hdf5 -y && conda install hdf5=1.8.17 -y
# # RUN /bin/bash -c "source activate python2 && conda remove hdf5 -y && conda install hdf5=1.8.17 -y"
#
# # TensorBoard
# EXPOSE 6006
#
# # ==============================================================
# #  set workdir and user for jupyter notebook use
# # ==============================================================
#
# # cleanup
# WORKDIR /home/$NB_USER/work/
# USER $NB_USER
