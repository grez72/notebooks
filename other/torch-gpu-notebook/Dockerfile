# nvidia-cuda-notebook
# Modified nvidia-cuda-devel to work as single-user notebook with jupyterhub
# Starting with the ubuntu14.04 cuda8 runtime and development Dockerfiles
# https://gitlab.com/nvidia/cuda/blob/ubuntu14.04/8.0/runtime/cudnn5/Dockerfile
# https://gitlab.com/nvidia/cuda/blob/ubuntu14.04/8.0/devel/Dockerfile
# sudo nvidia-docker build -t grez72/torch-gpu-notebook https://github.com/grez72/notebooks.git#master:torch-gpu-notebook
# sudo nvidia-docker run -it --rm -p 8888:8888 grez72/torch-gpu-notebook

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

# Install Python 3 Tensorflow
# RUN conda install --quiet --yes 'tensorflow-gpu=1.1*'

# Install Python 2 Tensorflow
# RUN conda install --quiet --yes -n python2 'tensorflow-gpu=1.1*'

# ==============================================================
#  INSTALL TORCH!
# ==============================================================

# Install git, apt-add-repository and dependencies for iTorch
RUN apt-get update && apt-get install -y \
  git \
  software-properties-common \
  libssl-dev \
  libzmq3-dev \
  python-dev \
  python-pip \
  python-zmq \
  libprotobuf-dev \
  protobuf-compiler \
  sudo

# Install OpenBLAS?
RUN git clone https://github.com/xianyi/OpenBLAS.git && \
    cd OpenBLAS && \
    make NO_AFFINITY=1 USE_OPENMP=1 && \
    make install

# Install Jupyter Notebook for iTorch
# RUN pip install --upgrade pip
# RUN pip install jupyter

# take 1
# RUN git clone https://github.com/torch/distro.git /root/torch --recursive;
#  	cd /root/torch && \
#  	bash install-deps && \
#  	yes no | ./install.sh

# take 2 (permission error with itorch kernal)
# RUN git clone https://github.com/torch/distro.git /root/torch --recursive;
# WORKDIR /root/torch
# RUN bash install-deps && \
#   	yes no | ./install.sh

# take 3
USER $NB_USER

RUN git clone --recursive https://github.com/torch/distro.git $HOME/torch

USER root
RUN cd $HOME/torch && bash install-deps

USER $NB_USER

RUN cd $HOME/torch && bash install.sh -b && \
    install/bin/torch-activate && \
    install/bin/luarocks install itorch

# Run Torch7 installation scripts (dependencies only)
# RUN git clone https://github.com/torch/distro.git /root/torch --recursive && cd /root/torch && \
#  bash install-deps

# Run Torch7 installation scripts
# RUN cd /root/torch && \
# Run without nvcc to prevent timeouts
#  sed -i 's/path_to_nvcc=$(which nvcc)/path_to_nvcc=$(which no_nvcc)/g' install.sh && \
#  sed -i 's,path_to_nvcc=/usr/local/cuda/bin/nvcc,path_to_nvcc=,g' install.sh && \
#  ./install.sh

# Export environment variables manually
# ENV LUA_PATH='/root/.luarocks/share/lua/5.1/?.lua;/root/.luarocks/share/lua/5.1/?/init.lua;/root/torch/install/share/lua/5.1/?.lua;/root/torch/install/share/lua/5.1/?/init.lua;./?.lua;/root/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua'
# ENV LUA_CPATH='/root/.luarocks/lib/lua/5.1/?.so;/root/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so'
# ENV PATH=/root/torch/install/bin:$PATH
# ENV LD_LIBRARY_PATH=/root/torch/install/lib:$LD_LIBRARY_PATH
# ENV DYLD_LIBRARY_PATH=/root/torch/install/lib:$DYLD_LIBRARY_PATH
# ENV LUA_CPATH='/root/torch/install/lib/?.so;'$LUA_CPATH

# ENV LUA_PATH='/root/.luarocks/share/lua/5.1/?.lua;/root/.luarocks/share/lua/5.1/?/init.lua;/root/torch/install/share/lua/5.1/?.lua;/root/torch/install/share/lua/5.1/?/init.lua;./?.lua;/root/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua' \
# 	LUA_CPATH='/root/.luarocks/lib/lua/5.1/?.so;/root/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so' \
# 	PATH=/root/torch/install/bin:$PATH \
# 	LD_LIBRARY_PATH=/root/torch/install/lib:$LD_LIBRARY_PATH \
# 	DYLD_LIBRARY_PATH=/root/torch/install/lib:$DYLD_LIBRARY_PATH
# ENV LUA_CPATH='/root/torch/install/lib/?.so;'$LUA_CPATH

ENV LUA_PATH="$HOME/.luarocks/share/lua/5.1/?.lua;/$HOME/.luarocks/share/lua/5.1/?/init.lua;$HOME/torch/install/share/lua/5.1/?.lua;$HOME/torch/install/share/lua/5.1/?/init.lua;./?.lua;$HOME/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua" \
	LUA_CPATH="$HOME/.luarocks/lib/lua/5.1/?.so;$HOME/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so" \
	PATH=$HOME/torch/install/bin:$PATH \
	LD_LIBRARY_PATH=$HOME/torch/install/lib:$LD_LIBRARY_PATH \
	DYLD_LIBRARY_PATH=$HOME/torch/install/lib:$DYLD_LIBRARY_PATH
ENV LUA_CPATH="$HOME/torch/install/lib/?.so;"$LUA_CPATH

# Set ~/torch as working directory
# WORKDIR /root/torch

# Restore Torch7 installation script
# RUN sed -i 's/path_to_nvcc=$(which no_nvcc)/path_to_nvcc=$(which nvcc)/g' install.sh

# Install latest CUDA libraries
# RUN luarocks install torch && \
#  luarocks install cutorch && \
#  luarocks install cunn && \
#  luarocks install cudnn

# Install dependencies for Caffe so we can run luarocks install loadcaffe
USER root
RUN apt-get update && apt-get install -y \
		libboost-all-dev \
		libgflags-dev \
		libgoogle-glog-dev \
		libhdf5-serial-dev \
		libleveldb-dev \
		liblmdb-dev \
		libopencv-dev \
		libprotobuf-dev \
		libsnappy-dev \
		protobuf-compiler \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*

# Install the latest versions of nn, cutorch, cunn, cuDNN bindings and iTorch
RUN luarocks install nn && \
    luarocks install torch && \
  	luarocks install cutorch && \
  	luarocks install cunn && \
    luarocks install cudnn && \
    luarocks install loadcaffe

# USER $NB_USER
# RUN cd $HOME && git clone https://github.com/soumith/cudnn.torch.git && cd cudnn.torch && \
#   	git checkout R4 && \
#   	luarocks make && \
#   	\
#   	cd $HOME && git clone https://github.com/facebook/iTorch.git && \
#   	cd iTorch && \
#   	luarocks make

USER $NB_USER
RUN cd $HOME && git clone https://github.com/soumith/cudnn.torch.git

USER root
WORKDIR $HOME/cudnn.torch
RUN git checkout R4 && \
    luarocks make

USER $NB_USER
RUN cd $HOME && git clone https://github.com/facebook/iTorch.git

USER root
WORKDIR $HOME/iTorch
RUN luarocks make

RUN . $HOME/torch/install/bin/torch-activate

# cleanup
WORKDIR /home/$NB_USER/work/
USER $NB_USER
