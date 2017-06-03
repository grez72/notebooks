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

ENV CNTK_VERSION="2.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    # General
        ca-certificates \
        wget \
        && \
    # Clean-up
    apt-get -y autoremove \
        && \
    rm -rf /var/lib/apt/lists/*

# RUN apt-get install libjpeg62-turbo-dev

# Get CNTK Binary Distribution
WORKDIR $HOME
RUN CNTK_VERSION_DASHED=$(echo $CNTK_VERSION | tr . -) && \
    CNTK_SHA256="dab691a81602efe7c2c3729bff86e5934489c9f72cfc6911ae18bc909adf83b4" && \
    wget -q https://cntk.ai/BinaryDrop/CNTK-${CNTK_VERSION_DASHED}-Linux-64bit-GPU.tar.gz && \
    echo "$CNTK_SHA256 CNTK-${CNTK_VERSION_DASHED}-Linux-64bit-GPU.tar.gz" | sha256sum --check --strict - && \
    tar -xzf CNTK-${CNTK_VERSION_DASHED}-Linux-64bit-GPU.tar.gz && \
    rm -f CNTK-${CNTK_VERSION_DASHED}-Linux-64bit-GPU.tar.gz
    # /bin/bash $HOME/cntk/Scripts/install/linux/install-cntk.sh --py-version 35 --docker
    # /bin/bash $HOME/cntk/Scripts/install/linux/install-cntk.sh --py-version 35 --docker && \
    # /bin/bash $HOME/cntk/Scripts/install/linux/install-cntk.sh --py-version 27 --docker

# cleanup
WORKDIR /home/$NB_USER/work/
USER $NB_USER
