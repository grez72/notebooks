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
# nvidia-docker run -it --rm grez72/nvidia-pytorch:ubuntu16.04-cuda10.0-cudnn7 python genpass.py -p 'supersecretpassword'
#
# mount it:
# docker run -it --rm grez72/nvidia-pytorch:ubuntu16.04-cuda10.0-cudnn7 bash
#
# run it:
# docker run -it --rm -p 8888:8888 grez72/nvidia-pytorch:ubuntu16.04-cuda10.0-cudnn7
#
# If you are using jupyterhub, this image can be used with Dockerspawner to
# spawn a single-user notebook.

FROM nvcr.io/nvidia/pytorch:19.01-py3

#--------------------------------------
#   OS dependencies
#--------------------------------------

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get -yq dist-upgrade \
 && apt-get install -yq --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    sudo \
    locales \
    fonts-liberation \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

#--------------------------------------
#   Configure environment
#--------------------------------------

ENV CONDA_DIR=/opt/conda \
    SHELL=/bin/bash \
    NB_USER=jovyan \
    NB_UID=1000 \
    NB_GID=100 \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    JUPYTER_ENABLE_LAB=1
ENV HOME=/home/$NB_USER

ADD fix-permissions /usr/local/bin/fix-permissions
# RUN chmod u+x /usr/local/bin/fix-permissions
# Create jovyan user with UID=1000 and in the 'users' group
# and make sure these dirs are writable by the `users` group.
RUN groupadd wheel -g 11 && \
    echo "auth required pam_wheel.so use_uid" >> /etc/pam.d/su && \
    useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    mkdir -p $CONDA_DIR && \
    chown $NB_USER:$NB_GID $CONDA_DIR && \
    chmod g+w /etc/passwd && \
    fix-permissions $HOME && \
    fix-permissions $CONDA_DIR

# Setup work directory
USER $NB_UID

RUN mkdir /home/$NB_USER/work && \
    fix-permissions /home/$NB_USER

#--------------------------------------
#   Install Jupyter Notebook, Lab, and Hub
#--------------------------------------

USER $NB_USER

# Generate a notebook server config
# Cleanup temporary files
# Correct permissions
# Do all this in a single RUN command to avoid duplicating all of the
# files across image layers when the permissions change
RUN conda install --quiet --yes \
    'notebook=5.7.*' \
    'jupyterhub=0.9.*' \
    'jupyterlab=0.35.*' && \
    conda clean -tipsy && \
    jupyter labextension install @jupyterlab/hub-extension@^0.12.0 && \
    npm cache clean --force && \
    jupyter notebook --generate-config && \
    rm -rf $CONDA_DIR/share/jupyter/lab/staging && \
    rm -rf /home/$NB_USER/.cache/yarn && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

#--------------------------------------
# Install all OS dependencies for fully functional notebook server:
#--------------------------------------

USER root

RUN apt-get update && apt-get install -yq --no-install-recommends \
    build-essential \
    emacs \
    git \
    inkscape \
    jed \
    libsm6 \
    libxext-dev \
    libxrender1 \
    lmodern \
    netcat \
    pandoc \
    python-dev \
    texlive-fonts-extra \
    texlive-fonts-recommended \
    texlive-generic-recommended \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-xetex \
    unzip \
    nano \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#--------------------------------------
# Notebook Setup
#--------------------------------------

USER root

EXPOSE 8888
WORKDIR $HOME/work

# Configure container startup
ENTRYPOINT ["tini", "-g", "--"]
CMD ["start-notebook.sh"]

# Add local files as late as possible to avoid cache busting
COPY start.sh /usr/local/bin/
COPY start-notebook.sh /usr/local/bin/
COPY start-singleuser.sh /usr/local/bin/
COPY jupyter_notebook_config.py /etc/jupyter/
COPY genpass.py .
RUN fix-permissions /etc/jupyter/

WORKDIR $HOME/work

# Switch back to jovyan to avoid accidental container runs as root
USER $NB_UID
