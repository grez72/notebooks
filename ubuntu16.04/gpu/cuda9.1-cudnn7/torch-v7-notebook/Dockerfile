# grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-torch-v7-notebook
#
# Torch jupyter notebook with gpu suppport (nvidia CUDA cuDNN).
#
# sudo nvidia-docker build -t grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-torch-v7-notebook https://github.com/grez72/notebooks.git#master:ubuntu16.04/gpu/cuda9.1-cudnn7/torch-v7-notebook
# nvidia-docker run -it --rm --shm-size=2G -p 8888:8888 grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-torch-v7-notebook

FROM grez72/deep-notebooks:ubuntu16.04-gpu-cuda9.1-cudnn7-minimal-notebook

MAINTAINER George Alvarez <gre72@gmail.com, alvarez@wjh.harvard.edu>

# ARG TORCH_VERSION=

USER root
WORKDIR $HOME

# ==============================================================
#  make python2 the default
# ==============================================================

RUN echo 'export PATH="/opt/conda/envs/python2/bin:$PATH"' > ~/.bashrc \
  && source ~/.bashrc
  
# ==============================================================
#  INSTALL TORCH
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

# Install OpenBLAS
RUN git clone https://github.com/xianyi/OpenBLAS.git && \
    cd OpenBLAS && \
    make -j$(nproc) NO_AFFINITY=1 USE_OPENMP=1 && \
    make install

# clone torch as $NB_USER
USER $NB_USER
RUN git clone --recursive https://github.com/torch/distro.git $HOME/torch

# install torch
USER root
RUN cd $HOME/torch && bash install-deps

USER $NB_USER

RUN cd $HOME/torch && bash install.sh -b && \
    install/bin/torch-activate && \
    install/bin/luarocks install itorch

# add / update paths
ENV LUA_PATH="$HOME/.luarocks/share/lua/5.1/?.lua;/$HOME/.luarocks/share/lua/5.1/?/init.lua;$HOME/torch/install/share/lua/5.1/?.lua;$HOME/torch/install/share/lua/5.1/?/init.lua;./?.lua;$HOME/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua" \
	LUA_CPATH="$HOME/.luarocks/lib/lua/5.1/?.so;$HOME/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so" \
	PATH=$HOME/torch/install/bin:$PATH \
	LD_LIBRARY_PATH=$HOME/torch/install/lib:$LD_LIBRARY_PATH \
	DYLD_LIBRARY_PATH=$HOME/torch/install/lib:$DYLD_LIBRARY_PATH
ENV LUA_CPATH="$HOME/torch/install/lib/?.so;"$LUA_CPATH

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

# Install Torch7 FFI bindings for NVIDIA cuDNN (R5) kernels
USER $NB_USER
RUN cd $HOME && git clone https://github.com/soumith/cudnn.torch.git

USER root
WORKDIR $HOME/cudnn.torch
RUN git checkout R4 && \
    luarocks make

# Install iTorch Kernal for use with jupyter notebook
USER $NB_USER
RUN cd $HOME && git clone https://github.com/facebook/iTorch.git

USER root
WORKDIR $HOME/iTorch
RUN luarocks make

# activate torch
RUN . $HOME/torch/install/bin/torch-activate

# ==============================================================
#  set workdir and user for jupyter notebook use
# ==============================================================

WORKDIR /home/$NB_USER/work/
USER $NB_USER
