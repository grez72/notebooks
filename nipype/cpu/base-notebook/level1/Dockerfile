# level1-notebook for fmri data analysis: Nipype, FSL and AFNI
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
# docker build -t grez72/nipype-cpu-base-notebook-level1 https://github.com/grez72/notebooks.git#master:nipype/cpu/base-notebook/level1
#
# run it:
# docker run -it --rm -p 8888:8888 grez72/nipype-cpu-base-notebook-level1
# docker run --user root -it --rm -p 8888:8888 grez72/nipype-cpu-base-notebook-level1

FROM grez72/nipype-cpu-base-notebook-level0
MAINTAINER George Alvarez <grez72@gmail.com, alvarez.wjh.harvard.edu>

#---------------------
# Install FSL and AFNI
#---------------------
USER root
RUN apt-get update && \
    apt-get install -y -qq --no-install-recommends fsl-core fsl-atlases afni
ENV FSLDIR=/usr/share/fsl/5.0 \
    FSLOUTPUTTYPE=NIFTI_GZ \
    FSLMULTIFILEQUIT=TRUE \
    POSSUMDIR=/usr/share/fsl/5.0 \
    LD_LIBRARY_PATH=/usr/lib/fsl/5.0:$LD_LIBRARY_PATH \
    FSLTCLSH=/usr/bin/tclsh \
    FSLWISH=/usr/bin/wish \
    AFNI_MODELPATH=/usr/lib/afni/models \
    AFNI_IMSAVE_WARNINGS=NO \
    AFNI_TTATLAS_DATASET=/usr/share/afni/atlases \
    AFNI_PLUGINPATH=/usr/lib/afni/plugins \
    PATH=/usr/lib/fsl/5.0:/usr/lib/afni/bin:$PATH

#----------------------------------------
# Clear apt cache and other empty folders
#----------------------------------------
USER root
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /boot /media /mnt /srv
