FROM debian:stretch-slim

# Pre-cache neurodebian key
COPY docker/files/neurodebian.gpg /root/.neurodebian.gpg

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
        graphviz \
        libpng-dev \
        gnupg \
        build-essential \
        libgomp1 \
        libmpich-dev \
        mpich \
        git \
        g++ \
        zip \
        unzip \
        libglu1 \
        zlib1g-dev \
        libfreetype6-dev \
        pkg-config \
        libgsl0-dev \
        openssl \
	    openssh-server \
        gsl-bin \
        libglu1-mesa-dev \
        libglib2.0-0 \
        libglw1-mesa \
	liblapack-dev \
	libopenblas-base \
	sqlite3 \
	libsqlite3-dev \
	libquadmath0 \
    # Configure ssh
    && mkdir /var/run/sshd \
    && echo 'root:screencast' | chpasswd \
    && sed -i 's/PermitRootLogin without-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd \
    # Add and configure git-lfs
    && apt-get install -y apt-transport-https debian-archive-keyring \
    && apt-get install -y dirmngr --install-recommends \
    && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get update && \
    apt-get install -y git-lfs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && curl -o /tmp/libxp6.deb -sSL http://mirrors.kernel.org/debian/pool/main/libx/libxp/libxp6_1.0.2-2_amd64.deb \
    && dpkg -i /tmp/libxp6.deb && rm -f /tmp/libxp6.deb \
    # Add new user.
    && groupadd -r neuro && useradd --no-log-init --create-home --shell /bin/bash -r -g neuro neuro \
    && chmod a+s /opt \
    && chmod 777 -R /opt \
    && apt-get clean -y && apt-get autoclean -y && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \

# Add Neurodebian package repositories (i.e. for FSL)
RUN curl -sSL http://neuro.debian.net/lists/stretch.us-tn.full >> /etc/apt/sources.list.d/neurodebian.sources.list && \
    apt-key add /root/.neurodebian.gpg && \
    (apt-key adv --refresh-keys --keyserver hkp://ha.pool.sks-keyservers.net 0xA5D32F012649A5A9 || true) && \
    apt-get update -qq && apt-get update -qq && apt-get install -y fsl-core
USER neuro
WORKDIR /home/neuro

# Install Miniconda, python, and basic packages.
ARG miniconda_version="4.3.27"
ENV PATH="/opt/conda/bin":$PATH
RUN curl -sSLO https://repo.continuum.io/miniconda/Miniconda3-${miniconda_version}-Linux-x86_64.sh \
    && bash Miniconda3-${miniconda_version}-Linux-x86_64.sh -b -p /opt/conda \
    && conda config --system --prepend channels conda-forge \
    && conda config --system --set auto_update_conda false \
    && conda config --system --set show_channel_urls true \
    && conda clean -tipsy \
    && conda install -yq python=3.6 ipython \
    && pip install --upgrade pip \
    && conda clean -tipsy \
    && rm -rf Miniconda3-${miniconda_version}-Linux-x86_64.sh \
    && pip install numpy requests psutil sqlalchemy importlib-metadata>=0.12 \
    # Install pynets
    && git clone -b development https://github.com/dPys/PyNets /home/neuro/PyNets && \
    cd /home/neuro/PyNets && \
    pip install -r requirements.txt && \
    python setup.py install \
    # Install skggm
    && conda install -yq \
        cython \
        libgfortran \
        matplotlib \
        openblas \
    && conda clean -tipsy \
    && pip install skggm python-dateutil==2.8.0 \
    && sed -i '/mpl_patches = _get/,+3 d' /opt/conda/lib/python3.6/site-packages/nilearn/plotting/glass_brain.py \
    && sed -i '/for mpl_patch in mpl_patches:/,+2 d' /opt/conda/lib/python3.6/site-packages/nilearn/plotting/glass_brain.py \
    # Precaching fonts, set 'Agg' as default backend for matplotlib
    && python -c "from matplotlib import font_manager" \
    && sed -i 's/\(backend *: \).*$/\1Agg/g' $( python -c "import matplotlib; print(matplotlib.matplotlib_fname())" ) \
    # Create nipype config for resource monitoring
    && mkdir -p ~/.nipype \
    && echo "[monitoring]" > ~/.nipype/nipype.cfg \
    && echo "enabled = true" >> ~/.nipype/nipype.cfg \
    && pip uninstall -y pandas \
    && pip install pandas -U \
    && rm -rf /home/neuro/PyNets \
    && rm -rf /home/neuro/.cache

# Handle permissions, cleanup, and create mountpoints
USER root
RUN chmod a+s -R /opt \
    && chown -R neuro /opt/conda/lib/python3.6/site-packages \
    && mkdir -p /home/neuro/.pynets \
    && chown -R neuro /home/neuro/.pynets \
    && chmod 777 /opt/conda/bin/pynets \
    && chmod 777 -R /home/neuro/.pynets \
    && chmod 777 /opt/conda/bin/pynets \
    && chmod 777 /opt/conda/bin/pynets_bids \
    && chmod 777 /opt/conda/bin/pynets_collect \
    && chmod 777 /opt/conda/bin/pynets_cloud \
    && find /opt/conda/lib/python3.6/site-packages -type f -iname "*.py" -exec chmod 777 {} \; \
    && find /opt -type f -iname "*.py" -exec chmod 777 {} \; \
    && find /opt -type f -iname "*.yaml" -exec chmod 777 {} \; \
    && apt-get purge -y --auto-remove \
	git \
	gcc \
	wget \
	curl \
	build-essential \
	ca-certificates \
	gnupg \
	g++ \
	openssl \
	git-lfs \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && conda clean -tipsy \
    && mkdir /inputs && \
    chmod -R 777 /inputs \
    && mkdir /outputs && \
    chmod -R 777 /outputs \
    && mkdir /working && \
    chmod -R 777 /working

USER neuro

# ENV Config
ENV LD_LIBRARY_PATH="/opt/conda/lib":$LD_LIBRARY_PATH
ENV PATH="/opt/conda/lib/python3.6/site-packages/pynets":$PATH
ENV FSLDIR=/usr/share/fsl/5.0
ENV PATH=/usr/lib/fsl/5.0/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/lib/fsl/5.0:$LD_LIBRARY_PATH
ENV FSLTCLSH=/usr/bin/tclsh
ENV FSLWISH=/usr/bin/wish
ENV FSLBROWSER=/etc/alternatives/x-www-browser
ENV LD_LIBRARY_PATH="/usr/lib/openblas-base":$LD_LIBRARY_PATH
ENV FSLMULTIFILEQUIT=TRUE
ENV FSLOUTPUTTYPE=NIFTI_GZ
ENV FSLMULTIFILEQUIT=TRUE
ENV PYTHONWARNINGS ignore
ENV OMP_NUM_THREADS=1
ENV USE_SIMPLE_THREADED_LEVEL3=1




EXPOSE 22

# and add it as an entrypoint
#ENTRYPOINT ["pynets_bids"]
