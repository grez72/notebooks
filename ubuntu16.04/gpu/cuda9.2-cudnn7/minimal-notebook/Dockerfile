# grez72/minimal-notebook
#
# build it:
# nvidia-docker build -t grez72/minimal-notebook:ubuntu16.04-gpu-cuda9.2-cudnn7 https://github.com/grez72/notebooks.git#master:ubuntu16.04/gpu/cuda9.2-cudnn7/minimal-notebook
#
# test it:
# nvidia-docker run -it --rm grez72/minimal-notebook:ubuntu16.04-gpu-cuda9.2-cudnn7 nvidia-smi
#
# mount it:
# nvidia-docker run -it --rm grez72/minimal-notebook:ubuntu16.04-gpu-cuda9.2-cudnn7 bash
#
# run the notebook:
# nvidia-docker run -it --rm -p 8888:8888 grez72/minimal-notebook:ubuntu16.04-gpu-cuda9.2-cudnn7

FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04

LABEL maintainer="George Alvarez <alvarez@wjh.harvard.com>"

#--------------------------------------
#   OS dependencies
#--------------------------------------

USER root
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

ENV PATH=$CONDA_DIR/bin:$PATH \
    HOME=/home/$NB_USER

ADD fix-permissions /usr/local/bin/fix-permissions

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

USER $NB_UID

# Setup work directory for backward-compatibility
RUN mkdir /home/$NB_USER/work && \
    fix-permissions /home/$NB_USER

#--------------------------------------
#   Install conda / python
#--------------------------------------

USER $NB_UID

# Install conda as jovyan and check the md5 sum provided on the download site
ENV MINICONDA_VERSION 4.5.4
RUN cd /tmp && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    echo "a946ea1d0c4a642ddf0c3a26a18bb16d *Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh" | md5sum -c - && \
    /bin/bash Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    $CONDA_DIR/bin/conda config --system --prepend channels conda-forge && \
    $CONDA_DIR/bin/conda config --system --set auto_update_conda false && \
    $CONDA_DIR/bin/conda config --system --set show_channel_urls true && \
    $CONDA_DIR/bin/conda install --quiet --yes conda="${MINICONDA_VERSION%.*}.*" && \
    $CONDA_DIR/bin/conda update --all --quiet --yes && \
    conda clean -tipsy && \
    rm -rf /home/$NB_USER/.cache/yarn && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER


# python2 environment
RUN conda create -p $CONDA_DIR/envs/python2 python=2.7 \
    'ipython=4.0*' \
    'ipywidgets=4.0*' \
    && conda clean -yt

USER root
RUN /bin/bash -c "source activate python2 && ipython kernel install"

# Install Tini
USER $NB_UID
RUN conda install --quiet --yes 'tini=0.18.0' && \
    conda list tini | grep tini | tr -s ' ' | cut -d ' ' -f 1,2 >> $CONDA_DIR/conda-meta/pinned && \
    conda clean -tipsy && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

#--------------------------------------
#   Jupyterhub/Jupyterlab
#--------------------------------------

USER root
RUN pip install jupyter -U && pip install jupyterlab && ipython kernel install

WORKDIR $HOME/work

EXPOSE 8888

ENTRYPOINT ["tini", "-g", "--"]
CMD ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]
