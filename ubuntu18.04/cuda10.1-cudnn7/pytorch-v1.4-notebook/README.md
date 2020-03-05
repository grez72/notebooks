## grez72/deep-notebooks:ubuntu18.04-cuda10.1-cudnn7-pytorch-v1.4-notebook

- docker build -t grez72/deep-notebooks:ubuntu18.04-cuda10.1-cudnn7-pytorch-v1.4-notebook https://github.com/grez72/notebooks.git#master:ubuntu18.04/cuda10.1-cudnn7/pytorch-v1.4-notebook
- docker run -it --rm --user root grez72/deep-notebooks:ubuntu18.04-cuda10.1-cudnn7-pytorch-v1.4-notebook bash
- docker run -it --rm --user root grez72/deep-notebooks:ubuntu18.04-cuda10.1-cudnn7-pytorch-v1.4-notebook nvidia-smi
- docker run -it --rm --shm-size=2G -p 8889:8888 grez72/deep-notebooks:ubuntu18.04-cuda10.1-cudnn7-pytorch-v1.4-notebook