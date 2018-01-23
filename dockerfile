FROM debian:stretch-slim

ARG DEBIAN_FRONTEND="noninteractive"

ENV LANG="C.UTF-8" \
    LC_ALL="C.UTF-8"

RUN apt-get update -qq \
    # Install system dependencies.
    && apt-get install -y --no-install-recommends \
        bzip2 \
        ca-certificates \
        curl \
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

# Install conda build and deployment tools.
RUN export PATH="/opt/conda/bin:${PATH}" && \
    conda install --yes --quiet conda-build anaconda-client jinja2 setuptools && \
    conda install --yes git && \
    conda clean -tipsy

# Install pynets.
RUN cd /opt \
    && git clone https://github.com/dpisner453/PyNets.git
COPY --chown=neuro:users . /opt/PyNets
RUN conda install -yq \
      python=3.6 \
      matplotlib>=2.0.0 \
      numpy>=1.13.3 \
      scikit-learn>=0.19.1 \
      scipy>=1.0.0 \
      setuptools>=38.2.4 \
      traits \
    && conda clean -tipsy \
    && cd /opt/PyNets \
    && pip install --no-cache-dir -e . \
    && sed -i '/mpl_patches = _get/,+1 d' /opt/conda/lib/python3.6/site-packages/nilearn/plotting/glass_brain.py \
    && sed -i '/for mpl_patch in mpl_patches:/,+1 d' /opt/conda/lib/python3.6/site-packages/nilearn/plotting/glass_brain.py

# Install skggm
RUN conda install -yq \
        cython \
        libgfortran \
        matplotlib \
        openblas \
    && conda clean -tipsy \
    && pip install --no-cache-dir https://dl.dropbox.com/s/ghgl6lff2fmtldn/skggm-0.2.7-cp36-cp36m-linux_x86_64.whl
