# nvidia-docker build -t grez72/cuda-caffe-gpu-notebook https://github.com/grez72/notebooks.git#master:caffe-gpu-notebook
# nvidia-docker run -it --rm -p 8888:8888 grez72/cuda-caffe-gpu-notebook
# nvidia-docker run --user root -it --rm -p 8888:8888 grez72/cuda-caffe-gpu-notebook

FROM grez72/cuda-v8.0-gpu-notebook

# setup for notebook use
WORKDIR /home/$NB_USER/work/
USER $NB_USER
