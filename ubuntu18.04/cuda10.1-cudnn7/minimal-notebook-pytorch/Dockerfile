# grez72/minimal-notebook-pytorch for deep learning
#
# Runs a fully-loaded jupyterlab notebook server with access to the GPU on
# the host machine, running Ubuntu 16.04, CUDA 10.1, CUDNN7, python2, python3,
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
# docker build -t grez72/minimal-notebook-pytorch https://github.com/grez72/notebooks.git#master:ubuntu18.04/cuda10.1-cudnn7/minimal-notebook-pytorch
#
# test it:
# docker run -it --rm grez72/minimal-notebook-pytorch nvidia-smi
#
# generate a password:
# nvidia-docker run -it --rm grez72/minimal-notebook-pytorch python genpass.py -p 'supersecretpassword'
#
# mount it:
# docker run -it --rm --user root grez72/minimal-notebook-pytorch bash
#
# run it:
# docker run -it --rm -p 8888:8888 grez72/minimal-notebook-pytorch
#
# If you are using jupyterhub, this image can be used with Dockerspawner to
# spawn a single-user notebook.

FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

LABEL maintainer="George Alvarez <alvarez@wjh.harvard.com>"

USER root

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
  tmux \
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
ENV PATH=$CONDA_DIR/bin:$PATH \
  HOME=/home/$NB_USER

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
#   Install conda / python
#--------------------------------------

# Install conda as NB_USER and check the md5 sum provided on the download site
ENV MINICONDA_VERSION 4.5.4
ENV MINICONDA_CHECKSUM a946ea1d0c4a642ddf0c3a26a18bb16d
# ENV MINICONDA_VERSION 4.7.10
# ENV MINICONDA_CHECKSUM 1c945f2b3335c7b2b15130b1b2dc5cf4
RUN cd /tmp && \
  wget --quiet https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
  echo "${MINICONDA_CHECKSUM} *Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh" | md5sum -c - && \
  /bin/bash Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
  rm Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
  $CONDA_DIR/bin/conda config --system --prepend channels conda-forge && \
  $CONDA_DIR/bin/conda config --system --set auto_update_conda false && \
  $CONDA_DIR/bin/conda config --system --set show_channel_urls true && \
  $CONDA_DIR/bin/conda install --quiet --yes conda="${MINICONDA_VERSION%.*}.*" && \
  conda install -y -c conda-forge libiconv && \  
  conda clean -tipsy && \
  rm -rf /home/$NB_USER/.cache/yarn && \
  fix-permissions $CONDA_DIR && \
  fix-permissions /home/$NB_USER

# Install Tini
RUN conda install --quiet --yes 'tini=0.18.0' && \
  conda list tini | grep tini | tr -s ' ' | cut -d ' ' -f 1,2 >> $CONDA_DIR/conda-meta/pinned && \
  conda clean -tipsy && \
  fix-permissions $CONDA_DIR && \
  fix-permissions /home/$NB_USER

#--------------------------------------
#   Install Jupyter Notebook, Lab, and Hub
#--------------------------------------

USER $NB_UID

# Generate a notebook server config
# Cleanup temporary files
# Correct permissions
# Do all this in a single RUN command to avoid duplicating all of the
# files across image layers when the permissions change
RUN conda install --quiet --yes \
  'notebook=5.7.*' \
  'jupyterhub=1.0.*' \
  'jupyterlab=1.1.*' && \
  conda clean -tipsy && \
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
# Setup Python 3 Environment
#--------------------------------------

USER $NB_UID
WORKDIR $HOME

# Install Python 3 packages
# Remove pyqt and qt pulled in for matplotlib since we're only ever going to
# use notebook-friendly backends in these images
RUN conda install --quiet --yes \
  "conda=${MINICONDA_VERSION}" \
  'av' \
  'conda-forge::blas=*=openblas' \
  'ipywidgets=7.2*' \
  'pandas=0.23*' \
  'numexpr=2.6*' \
  'matplotlib=2.2*' \
  'scipy=1.1*' \
  'seaborn=0.9*' \
  'scikit-learn=0.19*' \
  'scikit-image=0.14*' \
  'sk-video=1.1*' \
  'sympy=1.1*' \
  'cython=0.28*' \
  'patsy=0.5*' \
  'statsmodels=0.9*' \
  'cloudpickle=0.5*' \
  'dill=0.2*' \
  'numba=0.38*' \
  'bokeh=0.12*' \
  'sqlalchemy=1.2*' \
  'hdf5=1.10*' \
  'h5py=2.7*' \
  'vincent=0.4.*' \
  'beautifulsoup4=4.6.*' \
  'protobuf=3.*' \
  'xlrd'  && \
  conda remove --quiet --yes --force qt pyqt && \
  conda clean -tipsy && \
  jupyter nbextension enable --py widgetsnbextension --sys-prefix && \
  jupyter labextension install @jupyter-widgets/jupyterlab-manager@1.0 && \
  npm cache clean --force && \
  rm -rf $CONDA_DIR/share/jupyter/lab/staging && \
  rm -rf /home/$NB_USER/.cache/yarn && \
  rm -rf /home/$NB_USER/.node-gyp && \
  fix-permissions $CONDA_DIR && \
  fix-permissions /home/$NB_USER

# Install facets which does not have a pip or conda package at the moment
RUN cd /tmp && \
  git clone https://github.com/PAIR-code/facets.git && \
  cd facets && \
  jupyter nbextension install facets-dist/ --sys-prefix && \
  cd && \
  rm -rf /tmp/facets && \
  fix-permissions $CONDA_DIR && \
  fix-permissions /home/$NB_USER

# Import matplotlib the first time to build the font cache.
ENV XDG_CACHE_HOME /home/$NB_USER/.cache/
RUN MPLBACKEND=Agg python -c "import matplotlib.pyplot" && \
  fix-permissions /home/$NB_USER

#--------------------------------------
# Install Optimized Fast PIL
#--------------------------------------

RUN conda uninstall -y --force pillow pil jpeg libtiff libjpeg-turbo \
  && pip uninstall -y pillow pil jpeg libtiff libjpeg-turbo \
  && conda install -yc conda-forge libjpeg-turbo \
  && CFLAGS="${CFLAGS} -mavx2" pip install --upgrade --no-cache-dir --force-reinstall --no-binary :all: --compile pillow-simd \
  && conda install -y jpeg libtiff

# ==============================================================
#  INSTALL PYTORCH
# ==============================================================

RUN pip install --upgrade pip \
  && pip install numpy --upgrade \
  && conda update conda \
  && conda install -c conda-forge mock \
  && conda install -y mkl -c intel \
  && conda install pytorch=$PYTORCH_VERSION torchvision cuda100 -c pytorch \
  && conda install -c fastai fastprogress fastai \
  && conda clean -tipsy

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
COPY credential-helper.sh /usr/local/bin/
COPY start.sh /usr/local/bin/
COPY start-notebook.sh /usr/local/bin/
COPY start-singleuser.sh /usr/local/bin/
COPY jupyter_notebook_config.py /etc/jupyter/
COPY genpass.py .
RUN fix-permissions /etc/jupyter/

WORKDIR $HOME/work

# Switch back to jovyan to avoid accidental container runs as root
USER $NB_UID
