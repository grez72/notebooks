# level3-notebook for fmri data analysis: Nipype, FSL, AFNI, SPM12 and ANTs
#
# This Dockerfile is based on miykael/nipype_course, which is based on the
# dockerfile 'fmriprep' from the PoldrackLab (https://github.com/poldracklab/fmriprep).
# The jupyter notebook foundation is based on jupyter/docker-stacks's base-notebook.
#
# This means that the same copyrights apply to this Dockerfile, as they do for
# the above mentioned dockerfiles. For more information see:
# https://github.com/miykael/nipype_env
#
# build it:
# docker build -t grez72/nipype-cpu-base-notebook-level3 https://github.com/grez72/notebooks.git#master:nipype/cpu/base-notebook/level3
#
# run it:
# docker run -it --rm -p 8888:8888 grez72/nipype-cpu-base-notebook-level3
# docker run --user root -it --rm -p 8888:8888 grez72/nipype-cpu-base-notebook-level3

FROM grez72/nipype-cpu-base-notebook-level2
MAINTAINER George Alvarez <grez72@gmail.com, alvarez.wjh.harvard.edu>

#---------------------
# Install ANTs and C3D
#---------------------
USER root
RUN mkdir -p /opt/ants && \
    curl -sSL "https://github.com/stnava/ANTs/releases/download/v2.1.0/Linux_Debian_jessie_x64.tar.bz2" \
    | tar -xjC /opt/ants --strip-components 1

ENV ANTSPATH=/opt/ants
ENV PATH=$ANTSPATH:$PATH

# Installing and setting up c3d
RUN mkdir -p /opt/c3d && \
    curl -sSL "http://downloads.sourceforge.net/project/c3d/c3d/1.0.0/c3d-1.0.0-Linux-x86_64.tar.gz" \
    | tar -xzC /opt/c3d --strip-components 1

ENV C3DPATH=/opt/c3d/
ENV PATH=$C3DPATH/bin:$PATH
