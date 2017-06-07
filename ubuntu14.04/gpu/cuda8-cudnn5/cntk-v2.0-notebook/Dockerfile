# grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-cntk-v2.0-notebook
#
# Microsoft Cognitive Toolkit jupyter notebook with gpu suppport (nvidia CUDA cuDNN).
#
# sudo nvidia-docker build -t grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-cntk-v2.0-notebook https://github.com/grez72/notebooks.git#master:ubuntu14.04/gpu/cuda8-cudnn5/cntk-v2.0-notebook
# nvidia-docker run -it --rm -p 8888:8888 grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-cntk-v2.0-notebook

FROM grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-base-notebook

MAINTAINER George Alvarez <gre72@gmail.com, alvarez@wjh.harvard.edu>

ARG CNTK_VERSION=v0.7.0
ARG OPENMPI_VERSION=1.10.3

# ==============================================================
#  INSTALL CNTK
# ==============================================================

USER root
WORKDIR $HOME

# Install Open Message Passing Interface Library (Open MPI)
RUN wget -q -O - https://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-${OPENMPI_VERSION}.tar.gz | tar -xzf - && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --prefix=/usr/local/mpi && \
    make -j"$(nproc)" install && \
    rm -rf /openmpi-${OPENMPI_VERSION}

ENV PATH /usr/local/mpi/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/mpi/lib:$LD_LIBRARY_PATH

RUN pip install --upgrade pip
RUN pip install https://cntk.ai/PythonWheel/GPU-1bit-SGD/cntk-2.0rc3-cp35-cp35m-linux_x86_64.whl
RUN pip2 install https://cntk.ai/PythonWheel/GPU-1bit-SGD/cntk-2.0rc3-cp27-cp27mu-linux_x86_64.whl

# change to work directory, install samples
WORKDIR /home/$NB_USER/work/
USER $NB_USER
RUN wget https://cntk.ai/Samples/CNTK-Samples-2-0rc3.zip && \
    unzip CNTK-Samples-2-0rc3.zip -d CNTK-Samples-2-0rc3 && \
    pip install -r CNTK-Samples-2-0rc3/requirements.txt && \
    pip2 install -r CNTK-Samples-2-0rc3/requirements.txt && \
    rm CNTK-Samples-2-0rc3.zip
