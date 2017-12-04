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
# nvidia-docker build -t grez72/cuda-v8.0-gpu-notebook https://github.com/grez72/notebooks.git#master:cuda-v8.0-gpu-notebook
#
# run it:
# nvidia-docker run -it --rm -p 8888:8888 grez72/cuda-v8.0-gpu-notebook
# nvidia-docker run --user root -it --rm -p 8888:8888 grez72/cuda-v8.0-gpu-notebook

FROM jupyter/scipy-notebook

LABEL com.nvidia.volumes.needed="nvidia_driver"

USER root
