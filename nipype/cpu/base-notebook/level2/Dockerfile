# level2-notebook for fmri data analysis: Nipype, FSL, AFNI and SPM12
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
# docker build -t grez72/nipype-cpu-base-notebook-level2 https://github.com/grez72/notebooks.git#master:nipype/cpu/base-notebook/level2
#
# run it:
# docker run -it --rm -p 8888:8888 grez72/nipype-cpu-base-notebook-level2
# docker run --user root -it --rm -p 8888:8888 grez72/nipype-cpu-base-notebook-level2

FROM grez72/nipype-cpu-base-notebook-level1
MAINTAINER George Alvarez <grez72@gmail.com, alvarez.wjh.harvard.edu>

#-------------------
# Install MATLAB MCR
#-------------------
USER root
ENV MATLAB_VERSION R2016b
RUN mkdir -p /opt && \
    mkdir /opt/mcr_install && \
    mkdir /opt/mcr && \
    wget -q -P /opt/mcr_install http://www.mathworks.com/supportfiles/downloads/${MATLAB_VERSION}/deployment_files/${MATLAB_VERSION}/installers/glnxa64/MCR_${MATLAB_VERSION}_glnxa64_installer.zip && \
    unzip -q /opt/mcr_install/MCR_${MATLAB_VERSION}_glnxa64_installer.zip -d /opt/mcr_install && \
    /opt/mcr_install/install -destinationFolder /opt/mcr -agreeToLicense yes -mode silent && \
    rm -rf /opt/mcr_install && \
    rm /opt/mcr/v91/sys/os/glnxa64/libstdc++.so.6 && \
    ln /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.20 /opt/mcr/v91/sys/os/glnxa64/libstdc++.so.6

# Configure environment
ENV MCR_VERSION=v91 \
    MCR_INHIBIT_CTF_LOCK=1 \
    PATH=/opt/mcr/v91/bin:$PATH \
    MATLABCMD=/opt/mcr/v91/bin/glnxa64/MATLABWindow
ENV LD_LIBRARY_PATH=/opt/mcr/${MCR_VERSION}/runtime/glnxa64:/opt/mcr/${MCR_VERSION}/bin/glnxa64:/opt/mcr/${MCR_VERSION}/sys/os/glnxa64:/opt/mcr/${MCR_VERSION}/sys/opengl/lib/glnxa64:$LD_LIBRARY_PATH

#-------------------------
# Install SPM12 Standalone
#-------------------------
USER root
ENV SPM_REVISION=r6906
ENV SPM_EXEC=/opt/spm12/spm12 \
    SPMMCRCMD='/opt/spm12/run_spm12.sh /opt/mcr/v91/ script' \
    FORCE_SPMMCR=1
RUN wget -q -P /opt http://www.fil.ion.ucl.ac.uk/spm/download/restricted/bids/spm12_${SPM_REVISION}_Linux_${MATLAB_VERSION}.zip && \
    unzip -q /opt/spm12_${SPM_REVISION}_Linux_${MATLAB_VERSION}.zip -d /opt && \
    rm -f /opt/spm12_${SPM_REVISION}_Linux_${MATLAB_VERSION}.zip && \
    ${SPM_EXEC} function exit

# Make the spm12_run.sh script executable for NB_USER
RUN chown -R $NB_USER:users /opt/spm12
RUN chown -R $NB_USER:users /home/$NB_USER/*

#----------------------------------------
# Clear apt cache and other empty folders
#----------------------------------------
USER root
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /boot /media /mnt /srv
