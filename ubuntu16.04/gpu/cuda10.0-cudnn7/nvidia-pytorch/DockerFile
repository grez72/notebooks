# grez72/base-notebook for deep learning
#
# Runs a fully-loaded jupyterlab notebook server with access to the GPU on
# the host machine, running Ubuntu 16.04, CUDA 10.0, CUDNN7, python2, python3,
# and including most of the packages you might need for deep learning,
# experiments except a deep learning framework! (we build those from this image)
#
# The image begins with nvidia/cuda image, then adds in jupyter/docker-stacks/scipy-notebook
# https://github.com/jupyter/docker-stacks/tree/master/scipy-notebook
#
# The Jupyter server uses a password specified in jupyter_notebook_config.py
# To use tokens instead, comment out the password line of the config file:
# c.NotebookApp.password = 'sha1:b43e2d39955f:b09acc36be8f4d051f6922e10f024db6d7dddd4b'
#
# You can then generate your own password by running the following in python:
# from notebook.auth import passwd
# print(passwd('supersecretpassword'))
#
# Then update the config file:
# c.NotebookApp.password = '...'
#
# uncomment the password line, and rebuild the image (fast because the config
# is copied at the end). You should then be able to login with your password.
#
# build it:
# docker build -t grez72/nvidia-pytorch:ubuntu16.04-cuda10.0-cudnn7 https://github.com/grez72/notebooks.git#master:ubuntu16.04/gpu/cuda10.0-cudnn7/nvidia-pytorch
#
# test it:
# docker run -it --rm grez72/nvidia-pytorch:ubuntu16.04-cuda10.0-cudnn7 nvidia-smi
#
# generate a password:
# nvidia-docker run -it --rm grez72/base-notebook:ubuntu16.04-cuda10.0-cudnn7 python genpass.py -p 'supersecretpassword'
#
# mount it:
# docker run -it --rm grez72/nvidia-pytorch:ubuntu16.04-cuda10.0-cudnn7 bash
#
# run it:
# docker run -it --rm -p 8888:8888 grez72/base-notebook:ubuntu16.04-cuda10.0-cudnn7
#
# If you are using jupyterhub, this image can be used with Dockerspawner to
# spawn a single-user notebook.

FROM nvcr.io/nvidia/pytorch:19.01-py3
