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
    && rm -rf /tmp/* /var/cache/apt/* \
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
COPY --chown=neuro:users . /opt/pynets
RUN conda install -y \
      python=3.6 \
      matplotlib>=2.0.0 \
      numpy>=1.13.3 \
      scikit-learn>=0.19.1 \
      scipy>=1.0.0 \
      setuptools>=38.2.4 \
      traits \
    && conda clean -tipsy \
    && pip install --no-cache-dir -e /opt/pynets
