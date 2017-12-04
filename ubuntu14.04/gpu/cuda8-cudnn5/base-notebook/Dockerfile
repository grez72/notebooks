# base-notebook for deep learning
#
# Modified nvidia-cuda-devel to work as single-user notebook with jupyterhub
# Starting with jupyter/scipy-notebook, then incorporating the ubuntu14.04
# cuda8 runtime and development Dockerfile instructions:
# https://gitlab.com/nvidia/cuda/blob/ubuntu14.04/8.0/runtime/cudnn5/Dockerfile
# https://gitlab.com/nvidia/cuda/blob/ubuntu14.04/8.0/devel/Dockerfile
# ...then including opencv
#
# build it:
# nvidia-docker build -t grez72/cuda-v8.0-gpu-notebook https://github.com/grez72/notebooks.git#master:ubuntu14.04/gpu/cuda8-cudnn5/base-notebook
#
# run it:
# nvidia-docker run -it --rm -p 8888:8888 grez72/cuda-v8.0-gpu-notebook
# nvidia-docker run --user root -it --rm -p 8888:8888 grez72/cuda-v8.0-gpu-notebook

FROM ubuntu:14.04

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

RUN make -j"$(nproc)"

# CMD defines the default command to be run in the container
# CMD is overridden by supplying a command + arguments to
# `docker run`, e.g. `nvcc --version` or `bash`
# CMD ./deviceQuery

# RUN conda install opencv
# RUN /bin/bash -c "source activate python2 && conda install opencv"
#
# #--------------------------------------
# # Packages to Enhance SciPi
# #--------------------------------------
#
# # RUN yes | pip install --upgrade pip
# # RUN yes | pip install --upgrade pyDOE
# # RUN yes | pip2 install --upgrade pyDOE
#
# #--------------------------------------
# # Setup for notebook use
# #--------------------------------------
#
# WORKDIR /home/$NB_USER/work/
# USER $NB_USER
