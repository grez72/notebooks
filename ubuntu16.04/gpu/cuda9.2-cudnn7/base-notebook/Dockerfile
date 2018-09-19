# base-noteook for deep learning cuda9.2-cudnn7
#
# build it:
# nvidia-docker build -t grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.2-cudnn7-base-notebook https://github.com/grez72/notebooks.git#master:ubuntu16.04/gpu/cuda9.2-cudnn7/base-notebook
#
# run it:
# nvidia-docker run -it --rm grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.2-cudnn7-base-notebook bash
# nvidia-docker run -it --rm -p 8888:8888 grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.2-cudnn7-base-notebook

FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04

#--------------------------------------
#   Jupyterhub/Jupyterlab
#--------------------------------------

WORKDIR /lab

RUN pip install jupyter -U && pip install jupyterlab

EXPOSE 8888

ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]

# starting from scipy-noteboo notebook latest as of 2018/09/19
# FROM jupyter/scipy-notebook:137a295ff71b
#
# # =======================================================
# #   INSTALL NVIDIA CUDA 9.2
# # =======================================================
#
# # https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.2/base/Dockerfile
#
# USER root
#
# RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates apt-transport-https gnupg-curl && \
#     rm -rf /var/lib/apt/lists/* && \
#     NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
#     NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
#     apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
#     apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +5 > cudasign.pub && \
#     echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub && \
#     echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
#     echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
#
# ENV CUDA_VERSION 9.2.148
#
# ENV CUDA_PKG_VERSION 9-2=$CUDA_VERSION-1
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         cuda-cudart-$CUDA_PKG_VERSION && \
#     ln -s cuda-9.2 /usr/local/cuda && \
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
# ENV NVIDIA_REQUIRE_CUDA "cuda>=9.2"
#
# # https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.2/runtime/Dockerfile
# ENV NCCL_VERSION 2.2.13
#
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         cuda-libraries-$CUDA_PKG_VERSION \
#         cuda-nvtx-$CUDA_PKG_VERSION \
#         libnccl2=$NCCL_VERSION-1+cuda9.2 && \
#     apt-mark hold libnccl2 && \
#     rm -rf /var/lib/apt/lists/*
#
# # https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.2/devel/Dockerfile
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         cuda-libraries-dev-$CUDA_PKG_VERSION \
#         cuda-nvml-dev-$CUDA_PKG_VERSION \
#         cuda-minimal-build-$CUDA_PKG_VERSION \
#         cuda-command-line-tools-$CUDA_PKG_VERSION \
#         libnccl-dev=$NCCL_VERSION-1+cuda9.2 && \
#     rm -rf /var/lib/apt/lists/*
#
# ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs
#
# # =======================================================
# #   INSTALL NVIDIA cuDNN 7
# # =======================================================
#
# # https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.2/devel/cudnn7/Dockerfile
# ENV CUDNN_VERSION 7.2.1.38
# LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
#
# RUN apt-get update && apt-get install -y --no-install-recommends \
#             libcudnn7=$CUDNN_VERSION-1+cuda9.2 \
#             libcudnn7-dev=$CUDNN_VERSION-1+cuda9.2 && \
#     apt-mark hold libcudnn7 && \
#     rm -rf /var/lib/apt/lists/*
#
# #--------------------------------------
# # Setup for notebook use
# #--------------------------------------
#
# WORKDIR /home/$NB_USER/work/
# USER $NB_USER
