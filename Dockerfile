FROM debian:stretch-slim

# Pre-cache neurodebian key
COPY docker/files/neurodebian.gpg /root/.neurodebian.gpg

ARG DEBIAN_FRONTEND="noninteractive"
ARG miniconda_version="4.3.27"

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
#        vim \
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
#        openssh-server \
#        jq \
        gsl-bin \
        libglu1-mesa-dev \
        libglib2.0-0 \
        libglw1-mesa \
        libxkbcommon-x11-0 \
        liblapack-dev \
        libopenblas-base \
        sqlite3 \
        libsqlite3-dev \
        libquadmath0 \
        gcc-multilib \
    # Configure ssh
#    && mkdir /var/run/sshd \
#    && echo 'root:screencast' | chpasswd \
#    && sed -i 's/PermitRootLogin without-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
#    && sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd \
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
    && curl -sSL http://neuro.debian.net/lists/stretch.us-tn.full >> /etc/apt/sources.list.d/neurodebian.sources.list && \
    apt-key add /root/.neurodebian.gpg && \
    (apt-key adv --refresh-keys --keyserver hkp://ha.pool.sks-keyservers.net 0xA5D32F012649A5A9 || true) && \
    apt-get update -qq && apt-get install --no-install-recommends -y fsl-5.0-core && \
    apt-get clean && cd /tmp \
    && wget https://fsl.fmrib.ox.ac.uk/fsldownloads/patches/fsl-5.0.10-python3.tar.gz \
    && tar -zxvf fsl-5.0.10-python3.tar.gz \
    && cp fsl/bin/* /usr/share/fsl/5.0/bin \
    && rm -r fsl* \
    && chmod 777 -R /usr/share/fsl/5.0/bin \
    && chmod 777 -R /usr/lib/fsl/5.0 \
    && echo "tmpfs   /tmp         tmpfs   rw,nodev,nosuid,size=2G          0  0" >> /etc/fstab \
    && echo "GRUB_CMDLINE_LINUX_DEFAULT="rootflags=uquota,pquota"" >> /etc/default/grub \
    && head -c 2G </dev/urandom > /tmp/3G_heap.txt # Here, we create a tmpfs heap, which gets reflected in /etc/fstab. We will delete it after creating the next run-layer so that the extra tmpfs storage stay available as free disk space.

ENV FSLDIR=/usr/share/fsl/5.0 \
    FSLOUTPUTTYPE=NIFTI_GZ \
    FSLMULTIFILEQUIT=TRUE \
    POSSUMDIR=/usr/share/fsl/5.0 \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/fsl/5.0 \
    FSLTCLSH=/usr/bin/tclsh \
    FSLWISH=/usr/bin/wish \
    PATH=$FSLDIR/bin:$PATH
ENV PATH="/opt/conda/bin":$PATH
ENV PATH="/opt/conda/lib/python3.8/site-packages/pynets/cli":$PATH

WORKDIR /home/neuro

RUN echo "FSLDIR=/usr/share/fsl/5.0" >> /home/neuro/.bashrc && \
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/fsl/5.0" >> /home/neuro/.bashrc && \
    echo ". $FSLDIR/etc/fslconf/fsl.sh" >> /home/neuro/.bashrc && \
    echo "export FSLDIR PATH" >> /home/neuro/.bashrc && \
    echo "export DISPLAY=:0" >> /home/neuro/.bashrc \
    && curl -sSLO https://repo.continuum.io/miniconda/Miniconda3-${miniconda_version}-Linux-x86_64.sh \
    && bash Miniconda3-${miniconda_version}-Linux-x86_64.sh -b -p /opt/conda \
    && conda config --system --prepend channels conda-forge \
    && conda config --system --set auto_update_conda false \
    && conda config --system --set show_channel_urls true \
    && conda install -yq python=3.8 \
    && pip install --upgrade pip \
    && rm -rf Miniconda3-${miniconda_version}-Linux-x86_64.sh \
    && git config --global url."https://".insteadOf git:// \
    && git config --global http.postBuffer 1048576000 \
    && pip install numpy requests psutil sqlalchemy importlib-metadata>=0.12 pytest pingouin>=0.3.7 imbalanced-learn>=0.8.0 \
    && pip install --upgrade pyopenssl \
    && git clone https://github.com/dPys/multinetx.git /home/neuro/multinetx \
    && cd /home/neuro/multinetx && \
    pip install -r requirements.txt && \
    python setup.py install \
    && conda install -yq \
        cython \
        libgfortran \
        matplotlib \
        openblas \
        graph-tool \
    # Install pynets
    && pip install --upgrade --force-reinstall numpy \
    && git clone -b development https://github.com/dPys/PyNets /home/neuro/PyNets && \
    cd /home/neuro/PyNets && \
    pip install -r requirements.txt && \
    python setup.py install \
    && pip install certifi -U --ignore-installed \
    && pip install python-dateutil==2.8.0 \
#    && pip install skggm \
    # Create nipype config for resource monitoring
    && mkdir -p ~/.nipype \
    && echo "[monitoring]" > ~/.nipype/nipype.cfg \
    && echo "enabled = true" >> ~/.nipype/nipype.cfg \
    && pip uninstall -y pandas \
    && pip install pandas -U \
    && conda clean --all -y \
    && pip install --upgrade pyopenssl \
    && cd / \
    && rm -rf /home/neuro/PyNets \
    && rm -rf /home/neuro/.cache \
    && chmod a+s -R /opt \
    && chown -R neuro /opt/conda/lib/python3.8/site-packages \
    && mkdir -p /home/neuro/.pynets \
    && chown -R neuro /home/neuro/.pynets \
    && chmod 777 /opt/conda/bin/pynets \
    && chmod 777 -R /home/neuro/.pynets \
    && chmod 777 /opt/conda/bin/pynets \
#    && chmod 777 /opt/conda/bin/pynets_bids \
#    && chmod 777 /opt/conda/bin/pynets_cloud \
#    && chmod 777 /opt/conda/bin/pynets_benchmark \
#    && chmod 777 /opt/conda/bin/pynets_predict \
    && find /opt/conda/lib/python3.8/site-packages -type f -iname "*.py" -exec chmod 777 {} \; \
    && find /opt -type f -iname "*.py" -exec chmod 777 {} \; \
    && find /opt -type f -iname "*.yaml" -exec chmod 777 {} \; \
    && apt-get clean autoclean \
    && apt-get purge -y --auto-remove \
	git \
	gcc \
	wget \
	curl \
#	openssl \
	build-essential \
	ca-certificates \
	libc6-dev \
	gnupg \
	g++ \
	git-lfs \
    && rm -rf /var/lib/{apt,dpkg,cache,log}/ \
    && conda clean --all -y \
    && rm -rf /tmp/* /var/tmp/* \
    && rm -rf /opt/conda/pkgs \
    && find /opt/conda/ -type f,l -name '*.pyc' -delete \
    && mkdir /inputs && \
    chmod -R 777 /inputs \
    && mkdir /outputs && \
    chmod -R 777 /outputs \
    && mkdir /working && \
    chmod -R 777 /working \
    && rm -f /tmp/2G_heap.txt

# ENV Config
ENV PATH="/opt/conda/lib/python3.8/site-packages/pynets/cli":"$PATH"

ENV FSLDIR=/usr/share/fsl/5.0 \
    FSLOUTPUTTYPE=NIFTI_GZ \
    FSLMULTIFILEQUIT=TRUE \
    POSSUMDIR=/usr/share/fsl/5.0 \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/fsl/5.0 \
    FSLTCLSH=/usr/bin/tclsh \
    FSLWISH=/usr/bin/wish \
    PATH=$FSLDIR/bin:$PATH
ENV PATH="/opt/conda/bin":"$PATH" \
    OPENBLAS_NUM_THREADS=4 \
    GOTO_NUM_THREADS=4 \
    OMP_NUM_THREADS=4 \
    QT_QPA_PLATFORM=offscreen

EXPOSE 22

RUN echo "PATH="/opt/conda/lib/python3.8/site-packages/pynets/cli":$PATH" >> /home/neuro/.bashrc \
    && . /home/neuro/.bashrc

