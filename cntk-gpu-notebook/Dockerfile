 # nvidia-cuda-notebook
# Modified nvidia-cuda-devel to work as single-user notebook with jupyterhub
# Starting with the ubuntu14.04 cuda8 runtime and development Dockerfiles
# https://gitlab.com/nvidia/cuda/blob/ubuntu14.04/8.0/runtime/cudnn5/Dockerfile
# https://gitlab.com/nvidia/cuda/blob/ubuntu14.04/8.0/devel/Dockerfile
# https://github.com/Microsoft/CNTK-docker/blob/v2.0/ubuntu-14.04/version_2/gpu/runtime/python-3/Dockerfile
# sudo nvidia-docker build -t grez72/cntk-gpu-notebook https://github.com/grez72/notebooks.git#master:cntk-gpu-notebook
# sudo nvidia-docker run -it --rm -p 8888:8888 grez72/cntk-gpu-notebook

FROM jupyter/scipy-notebook
MAINTAINER George Alvarez <alvarez@wjh.harvard.edu>

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
RUN cp -P /usr/lib/x86_64-linux-gnu/libcudnn* /usr/local/cuda/lib64
RUN cp -P /usr/include/cudnn.h /usr/local/cuda/include

# test devices
WORKDIR /usr/local/cuda/samples/1_Utilities/deviceQuery

RUN make

# CMD defines the default command to be run in the container
# CMD is overridden by supplying a command + arguments to
# `docker run`, e.g. `nvcc --version` or `bash`
# CMD ./deviceQuery

WORKDIR /home/$NB_USER/work

# ==============================================================
#  INSTALL MICROSOFT COGNITIVE TOOLKIT!
# ==============================================================

# RUN apt-get update && \
#     apt-get dist-upgrade -y && \
#     apt-get install -y libtool-bin

ENV CNTK_VERSION="2.0"

RUN apt-get update && apt-get dist-upgrade -y && apt-get install -y --no-install-recommends \
    # General
        ca-certificates \
        wget \
        libtool-bin \
        && \
    # Clean-up
    apt-get -y autoremove \
        && \
    rm -rf /var/lib/apt/lists/*

# Get CNTK Binary Distribution
RUN CNTK_VERSION_DASHED=$(echo $CNTK_VERSION | tr . -) && \
    CNTK_SHA256="dab691a81602efe7c2c3729bff86e5934489c9f72cfc6911ae18bc909adf83b4" && \
    wget -q https://cntk.ai/BinaryDrop/CNTK-${CNTK_VERSION_DASHED}-Linux-64bit-GPU.tar.gz && \
    echo "$CNTK_SHA256 CNTK-${CNTK_VERSION_DASHED}-Linux-64bit-GPU.tar.gz" | sha256sum --check --strict - && \
    tar -xzf CNTK-${CNTK_VERSION_DASHED}-Linux-64bit-GPU.tar.gz && \
    rm -f CNTK-${CNTK_VERSION_DASHED}-Linux-64bit-GPU.tar.gz && \
    /bin/bash /cntk/Scripts/install/linux/install-cntk.sh --py-version 35 --docker

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libtool \
#     libatlas3-base \
#     curl \
#     autoconf
#
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         autotools-dev \
#         build-essential \
#         cmake \
#         git \
#         gfortran-multilib \
#         libavcodec-dev \
#         libavformat-dev \
#         libjasper-dev \
#         libjpeg-dev \
#         libpng-dev \
#         liblapacke-dev \
#         libswscale-dev \
#         libtiff-dev \
#         pkg-config \
#         wget \
#         zlib1g-dev \
#         # Protobuf
#         ca-certificates \
#         curl \
#         unzip \
#         # For Kaldi
#         python-dev \
#         automake \
#         libtool \
#         autoconf \
#         subversion \
#         # For Kaldi's dependencies
#         libapr1 libaprutil1 libltdl-dev libltdl7 libserf-1-1 libsigsegv2 libsvn1 m4 \
#         # For Java Bindings
#         openjdk-7-jdk \
#         # For SWIG
#         libpcre++-dev && \
#     rm -rf /var/lib/apt/lists/*
#
# RUN OPENMPI_VERSION=1.10.3 && \
#     wget -q -O - https://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-${OPENMPI_VERSION}.tar.gz | tar -xzf - && \
#     cd openmpi-${OPENMPI_VERSION} && \
#     ./configure --prefix=/usr/local/mpi && \
#     make -j"$(nproc)" install && \
#     rm -rf /openmpi-${OPENMPI_VERSION}
#
# ENV PATH /usr/local/mpi/bin:$PATH
# ENV LD_LIBRARY_PATH /usr/local/mpi/lib:$LD_LIBRARY_PATH
#
# RUN LIBZIP_VERSION=1.1.2 && \
#     wget -q -O - http://nih.at/libzip/libzip-${LIBZIP_VERSION}.tar.gz | tar -xzf - && \
#     cd libzip-${LIBZIP_VERSION} && \
#     ./configure && \
#     make -j"$(nproc)" install && \
#     rm -rf /libzip-${LIBZIP_VERSION}
#
# ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
#
# RUN wget -q -O - https://github.com/NVlabs/cub/archive/1.4.1.tar.gz | tar -C /usr/local -xzf -
#
# RUN OPENCV_VERSION=3.1.0 && \
#     wget -q -O - https://github.com/Itseez/opencv/archive/${OPENCV_VERSION}.tar.gz | tar -xzf - && \
#     cd opencv-${OPENCV_VERSION} && \
#     cmake -DWITH_CUDA=OFF -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local/opencv-${OPENCV_VERSION} . && \
#     make -j"$(nproc)" install && \
#     rm -rf /opencv-${OPENCV_VERSION}
#
# RUN OPENBLAS_VERSION=0.2.18 && \
#     wget -q -O - https://github.com/xianyi/OpenBLAS/archive/v${OPENBLAS_VERSION}.tar.gz | tar -xzf - && \
#     cd OpenBLAS-${OPENBLAS_VERSION} && \
#     make -j"$(nproc)" USE_OPENMP=1 | tee make.log && \
#     grep -qF 'OpenBLAS build complete. (BLAS CBLAS LAPACK LAPACKE)' make.log && \
#     grep -qF 'Use OpenMP in the multithreading.' make.log && \
#     make PREFIX=/usr/local/openblas install && \
#     rm -rf /OpenBLAS-${OPENBLAS_VERSION}
#
# ENV LD_LIBRARY_PATH /usr/local/openblas/lib:$LD_LIBRARY_PATH
#
# # Install Boost
# RUN BOOST_VERSION=1_60_0 && \
#     BOOST_DOTTED_VERSION=$(echo $BOOST_VERSION | tr _ .) && \
#     wget -q -O - https://sourceforge.net/projects/boost/files/boost/${BOOST_DOTTED_VERSION}/boost_${BOOST_VERSION}.tar.gz/download | tar -xzf - && \
#     cd boost_${BOOST_VERSION} && \
#     ./bootstrap.sh --prefix=/usr/local/boost-${BOOST_DOTTED_VERSION} --with-libraries=filesystem,system,test  && \
#     ./b2 -d0 -j"$(nproc)" install  && \
#     rm -rf /boost_${BOOST_VERSION}
#
# # Install Protobuf
# RUN PROTOBUF_VERSION=3.1.0 \
#     PROTOBUF_STRING=protobuf-$PROTOBUF_VERSION && \
#     wget -O - --no-verbose https://github.com/google/protobuf/archive/v${PROTOBUF_VERSION}.tar.gz | tar -xzf - && \
#     cd $PROTOBUF_STRING && \
#     ./autogen.sh && \
#     ./configure CFLAGS=-fPIC CXXFLAGS=-fPIC --disable-shared --prefix=/usr/local/$PROTOBUF_STRING && \
#     make -j $(nproc) install && \
#     cd .. && \
#     rm -rf $PROTOBUF_STRING
#
# # Install CNTK custom MKL
# RUN CNTK_CUSTOM_MKL_VERSION=3 && \
#     mkdir /usr/local/CNTKCustomMKL && \
#     wget --no-verbose -O - https://www.cntk.ai/mkl/CNTKCustomMKL-Linux-$CNTK_CUSTOM_MKL_VERSION.tgz | \
#     tar -xzf - -C /usr/local/CNTKCustomMKL
#
# # Install Kaldi
# ENV KALDI_VERSION=c024e8aa
# ENV KALDI_PATH /usr/local/kaldi-$KALDI_VERSION
#
# # RUN bash -c "source activate python2"
# # RUN bash -c "source activate python2" && apt-get install -y libtool libatlas3-base subversion
# RUN apt-get update && \
#     apt-get dist-upgrade -y && \
#     apt-get install -y libtool-bin
#
# RUN mv /bin/sh /bin/sh.orig && \
#    ln -s -f /bin/bash /bin/sh && \
#    mkdir $KALDI_PATH && \
#    wget --no-verbose -O - https://github.com/kaldi-asr/kaldi/archive/$KALDI_VERSION.tar.gz | tar -xzf - --strip-components=1 -C $KALDI_PATH && \
#    cd $KALDI_PATH && \
#    cd tools && \
#    perl -pi -e 's/^# (OPENFST_VERSION = 1.4.1)$/\1/' Makefile && \
#    ./extras/check_dependencies.sh && \
#    make -j $(nproc) all && \
#    cd ../src && \
#    ./configure --openblas-root=/usr/local/openblas --shared && \
#    make -j $(nproc) depend && \
#    make -j $(nproc) all && \
# # Remove some unneeded stuff in $KALDI_PATH to reduce size
#    find $KALDI_PATH -name '*.o' -print0 | xargs -0 rm && \
#    for dir in $KALDI_PATH/src/*bin; do make -C $dir clean; done && \
#    mv -f /bin/sh.orig /bin/sh
#
# ## PYTHON
#
# # Swig
# RUN cd /root && \
#     wget -q http://prdownloads.sourceforge.net/swig/swig-3.0.10.tar.gz -O - | tar xvfz - && \
#     cd swig-3.0.10 && \
#     ./configure --without-perl5 && \
#     make -j $(nproc) && \
#     make install
#
# # Anaconda
# RUN wget -q https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh && \
#     bash Anaconda3-4.2.0-Linux-x86_64.sh -b && \
#     rm Anaconda3-4.2.0-Linux-x86_64.sh
#
# RUN CONDA_ENV_PATH=/tmp/conda-linux-cntk-py35-environment.yml; \
#     wget -q https://raw.githubusercontent.com/Microsoft/CNTK/master/Scripts/install/linux/conda-linux-cntk-py35-environment.yml -O "$CONDA_ENV_PATH" && \
#     /root/anaconda3/bin/conda env create -p /root/anaconda3/envs/cntk-py35 --file "$CONDA_ENV_PATH" && \
#     rm -f "$CONDA_ENV_PATH"
#
# ENV PATH /root/anaconda3/envs/cntk-py35/bin:$PATH
#
# # NCCL
# ENV NCCL_VERSION=1.3.0-1
# ENV NCCL_PATH /usr/local/nccl-${NCCL_VERSION}
# RUN git clone --depth=1 -b v${NCCL_VERSION} https://github.com/NVIDIA/nccl.git && \
#     cd nccl && \
#     make CUDA_HOME=/usr/local/cuda PREFIX=$NCCL_PATH install && \
#     rm -rf /nccl
#
# WORKDIR /cntk
#
# RUN mkdir -p /usr/local/cudnn/cuda/include && \
#     ln -s /usr/include/cudnn.h /usr/local/cudnn/cuda/include/cudnn.h && \
#     mkdir -p /usr/local/cudnn/cuda/lib64 && \
#     ln -s /etc/alternatives/libcudnn_so /usr/local/cudnn/cuda/lib64/libcudnn.so && \
#     ln -s /usr/local/cuda/lib64/stubs/libnvidia-ml.so /usr/local/cuda/lib64/stubs/libnvidia-ml.so.1
#
# # Build CNTK
# RUN git clone --depth=1 --recursive -b master https://github.com/Microsoft/CNTK.git . && \
#     CONFIGURE_OPTS="\
#       --1bitsgd=yes \
#       --with-cuda=/usr/local/cuda \
#       --with-gdk-include=/usr/local/cuda/include \
#       --with-gdk-nvml-lib=/usr/local/cuda/lib64/stubs \
#       --with-kaldi=${KALDI_PATH} \
#       --with-py35-path=/root/anaconda3/envs/cntk-py35 \
#       --with-cudnn=/usr/local/cudnn \
#       --with-nccl=${NCCL_PATH}" && \
#     mkdir -p build/gpu/release && \
#     cd build/gpu/release && \
#     ../../../configure $CONFIGURE_OPTS --with-openblas=/usr/local/openblas && \
#     make -j"$(nproc)" all && \
#     cd ../../.. && \
#     mkdir -p build-mkl/gpu/release && \
#     cd build-mkl/gpu/release && \
#     ../../../configure $CONFIGURE_OPTS --with-mkl=/usr/local/CNTKCustomMKL && \
#     make -j"$(nproc)" all
#
# RUN cd Examples/Image/DataSets/CIFAR-10 && \
#     python install_cifar10.py && \
#     cd ../../../..
#
# RUN cd Examples/Image/DataSets/MNIST && \
#     python install_mnist.py && \
#     cd ../../../..
#
# ENV PATH=/cntk/build/gpu/release/bin:$PATH PYTHONPATH=/cntk/bindings/python LD_LIBRARY_PATH=/cntk/bindings/python/cntk/libs:$LD_LIBRARY_PATH

# cleanup
WORKDIR /home/$NB_USER/work/
USER $NB_USER
