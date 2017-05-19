# caffe-notebook installation with nvidia gpu support (cuda 8, cuDNN 5)
# built combining instructions from here:
# https://hub.docker.com/r/kaixhin/caffe-deps/~/dockerfile/
# with nvidia's
# http://www.nvidia.com/object/caffe-installation.html

# nvidia-docker build -t grez72/cuda-caffe-gpu-notebook https://github.com/grez72/notebooks.git#master:caffe-gpu-notebook
# nvidia-docker run -it --rm -p 8888:8888 grez72/cuda-caffe-gpu-notebook
# nvidia-docker run --user root -it --rm -p 8888:8888 grez72/cuda-caffe-gpu-notebook

FROM grez72/cuda-v8.0-gpu-notebook

USER root

# install caffe dependencies
RUN pip install --upgrade pip
RUN apt-get update
RUN apt-get install -y \
  git \
  wget \
  bc \
  cmake \
  libatlas-base-dev \
  libatlas-dev \
  libboost-all-dev \
  libopencv-dev \
  libprotobuf-dev \
  libgoogle-glog-dev \
  libgflags-dev \
  protobuf-compiler \
  libhdf5-dev \
  libhdf5-serial-dev \
  libleveldb-dev \
  liblmdb-dev \
  libsnappy-dev \
  python-dev \
  python-pip \
  python-numpy \
  python-skimage \
  python-protobuf \
  gfortran > /dev/null

# install nccl
RUN wget https://github.com/NVIDIA/nccl/releases/download/v1.2.3-1%2Bcuda8.0/libnccl1_1.2.3-1.cuda8.0_amd64.deb && \
  dpkg -i libnccl1_1.2.3-1.cuda8.0_amd64.deb && \
  rm libnccl1_1.2.3-1.cuda8.0_amd64.deb && \
  wget https://github.com/NVIDIA/nccl/releases/download/v1.2.3-1%2Bcuda8.0/libnccl-dev_1.2.3-1.cuda8.0_amd64.deb && \
  dpkg -i libnccl-dev_1.2.3-1.cuda8.0_amd64.deb && \
  rm libnccl-dev_1.2.3-1.cuda8.0_amd64.deb

# install caffe
WORKDIR /opt
RUN wget https://github.com/NVIDIA/caffe/archive/v0.15.13.tar.gz
RUN tar -zxf v0.15.13.tar.gz
RUN mv caffe-0.15.13 caffe
WORKDIR /opt/caffe
RUN cat python/requirements.txt | xargs -n1 pip install
RUN cp Makefile.config.example Makefile.config
RUN echo "USE_CUDNN := 1" >> Makefile.config
RUN echo "USE_NCCL := 1" >> Makefile.config
RUN echo "INCLUDE_DIRS := \$(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial/" >> Makefile.config
RUN echo "LIBRARY_DIRS := \$(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial" >> Makefile.config

RUN make all -j"$(nproc)"
# RUN make test && make runtest
RUN make pycaffe

# export PYTHONPATH
ENV CAFFE_ROOT=/opt/caffe
ENV PYTHONPATH=/opt/caffe/distribute/python:$PYTHONPATH
ENV PYTHONPATH=/opt/caffe/python:$PYTHONPATH

# necessary?
# ENV PYCAFFE_ROOT $CAFFE_ROOT/python
# ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
# ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
# RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

# download mnist data so what we can test our installation
WORKDIR $CAFFE_ROOT
RUN ./data/mnist/get_mnist.sh
RUN ./examples/mnist/create_mnist.sh
# cd $CAFFE_ROOT && ./examples/mnist/train_lenet.sh

# when I run python2, >import caffe I get the following error
# "libdc1394 error: Failed to initialize libdc1394"
# http://stackoverflow.com/questions/29274638/opencv-libdc1394-error-failed-to-initialize-libdc1394
# as per SO, the following line fixes it, but I'm not sure why...
# see also http://stackoverflow.com/questions/42149458/open-cv-error-failed-to-init-raw1394-persisting-in-docker/42151836#42151836
RUN ln /dev/null /dev/raw1394

# setup for notebook use
WORKDIR /home/$NB_USER/work/
USER $NB_USER
