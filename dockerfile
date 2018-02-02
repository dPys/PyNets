FROM debian:stretch-slim

ARG DEBIAN_FRONTEND="noninteractive"

ENV LANG="C.UTF-8" \
    LC_ALL="C.UTF-8"

RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends software-properties-common \
    # Install system dependencies.
    && apt-get install -y --no-install-recommends \
        bzip2 \
        ca-certificates \
        curl \
        libxtst6 \
        libgtk2.0-bin \
        libxft2 \
        lib32ncurses5 \
        libxmu-dev \
        vim \
        wget \
        libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    # Add new user.
    && useradd --no-user-group --create-home --shell /bin/bash neuro \
    && chmod a+s /opt \
    && chmod 777 -R /opt

USER neuro
WORKDIR /home/neuro

# Install Miniconda.
ARG miniconda_version="4.3.27"
ENV PATH="/opt/conda/bin:$PATH"
RUN curl -sSLO https://repo.continuum.io/miniconda/Miniconda3-${miniconda_version}-Linux-x86_64.sh \
    && bash Miniconda3-${miniconda_version}-Linux-x86_64.sh -b -p /opt/conda \
    && conda config --system --prepend channels conda-forge \
    && conda config --system --set auto_update_conda false \
    && conda config --system --set show_channel_urls true \
    && conda clean -tipsy \
    && rm -rf Miniconda3-${miniconda_version}-Linux-x86_64.sh

# Install pynets.
RUN conda install -yq \
      python=3.6 \
      setuptools>=38.2.4 \
      nipype==0.14.0 \
      traits \
    && conda clean -tipsy \
    && pip install pynets==0.5.2

RUN sed -i '/mpl_patches = _get/,+3 d' /opt/conda/lib/python3.6/site-packages/nilearn/plotting/glass_brain.py \
    && sed -i '/for mpl_patch in mpl_patches:/,+2 d' /opt/conda/lib/python3.6/site-packages/nilearn/plotting/glass_brain.py

# Install skggm
RUN conda install -yq \
        cython \
        libgfortran \
        matplotlib \
        openblas \
    && conda clean -tipsy \
    && pip install --no-cache-dir https://dl.dropbox.com/s/ghgl6lff2fmtldn/skggm-0.2.7-cp36-cp36m-linux_x86_64.whl

USER root
RUN chown -R neuro:users /opt \
    && chmod a+s -R /opt \
    && chmod 777 -R /opt/conda/lib/python3.6/site-packages/pynets \
    && find /opt -type f -iname "*.py" -exec chmod 777 {} \;

USER neuro
ENV PATH="/opt/conda/lib/python3.6/site-packages/pynets:$PATH"
