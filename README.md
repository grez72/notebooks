## grez72/docker-jupyter-notebooks-gpu
A collection of Dockerfiles for building docker containers used to
spawn single-user notebooks with DockerSpawner in JupyterHub. All notebooks are GPU enabled (CUDA 8.0, cuDNN v5.1), and there are different containers supporting several popular deep-learning frameworks (e.g., Tensorflow, Torch, Caffe, etc.).

The base notebook was created by modifying nvidia-cuda-devel to work as single-user notebook with jupyterhub, and then adding OpenCV.
https://gitlab.com/nvidia/cuda/blob/ubuntu14.04/8.0/runtime/cudnn5/Dockerfile
https://gitlab.com/nvidia/cuda/blob/ubuntu14.04/8.0/devel/Dockerfile

Most of the framework notebooks were created by copying  https://github.com/floydhub/dl-docker, making minor modifications for compatibility with the base-notebook.

## Base Notebook
* Ubuntu 14.04
* [CUDA 8.0](https://developer.nvidia.com/cuda-toolkit) (GPU version only)
* [cuDNN v5.1](https://developer.nvidia.com/cudnn) (GPU version only)
* [iPython/Jupyter Notebook](http://jupyter.org/) (including python2 and python3 kernals)
* [Numpy](http://www.numpy.org/), [SciPy](https://www.scipy.org/), [Pandas](http://pandas.pydata.org/), [Scikit Learn](http://scikit-learn.org/), [Matplotlib](http://matplotlib.org/)
* [OpenCV](http://opencv.org/)

## Frameworks
* [Tensorflow](https://www.tensorflow.org/), which can be used with or without [Keras](http://keras.io/)
* [Torch](http://torch.ch/) (includes nn, cutorch, cunn and cuDNN bindings), and [iPython/Jupyter Notebook](http://jupyter.org/) with the itorch kernal.
* [CNTK v2.0-GPU-1bit-SGD](https://www.microsoft.com/en-us/cognitive-toolkit/)

[ ] - [Caffe](http://caffe.berkeleyvision.org/)

[ ] - [Theano](http://deeplearning.net/software/theano/) which can be used with or without [Keras](http://keras.io/) or [Lasagne](http://lasagne.readthedocs.io/en/latest/)

[ ] - [Chainer](https://chainer.org/)

[ ] - [Digits](https://developer.nvidia.com/digits)

## Build


## Configure Jupyterhub
