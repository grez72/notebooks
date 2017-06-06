# grez72/deep-notebooks-ubuntu14.04-cuda8-cudnn5-base

Dockerfile for building for building the deep-learning framework [Tensorflow](https://www.tensorflow.org/) (with the option to use [Keras](http://keras.io/) or [TFLearn](http://tflearn.org)) for use with jupyter notebook.
This docker image can be used with Jupyterhub + DockerSpawner to spawn single-user notebooks.

## Base Notebook
* Ubuntu 14.04
* [CUDA 8.0](https://developer.nvidia.com/cuda-toolkit) (GPU version only)
* [cuDNN v5.1](https://developer.nvidia.com/cudnn) (GPU version only)
* [iPython/Jupyter Notebook](http://jupyter.org/) (including python2 and python3 kernals)
* [Numpy](http://www.numpy.org/), [SciPy](https://www.scipy.org/), [Pandas](http://pandas.pydata.org/), [Scikit Learn](http://scikit-learn.org/), [Matplotlib](http://matplotlib.org/)
* [OpenCV](http://opencv.org/)

### pull it from dockerhub
```
$ docker pull grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-tensorflow-notebook
```

### build it from github repository
```
sudo nvidia-docker build -t grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-tensorflow-notebook https://github.com/grez72/notebooks.git#master:ubuntu14.04/gpu/cuda8-cudnn5/tensorflow-notebook
```

# run it as standalone container
nvidia-docker run -it --rm -p 8888:8888 grez72/deep-notebooks:ubuntu14.04-gpu-cuda8-cudnn5-tensorflow-notebook

# run it with jupyterhub and Dockerspawner
