FROM andrewosh/binder-base

USER root

# Add dependency
RUN apt-get -qq update
RUN conda update libgfortran --force
RUN conda install libgcc --force
RUN apt-get install -y libblas3gf libblas-doc libblas-dev liblapack3gf liblapack-doc liblapack-dev
RUN apt-get install -y software-properties-common python-software-properties
RUN apt-get install -y libxtst6 libgtk2.0-bin libxft2 lib32ncurses5 libXp6 libxpm-dev libxpm4 libxmu-dev
#RUN apt-get install -y libbz2-dev libsqlite3-dev

# Create python3 environment
#RUN wget https://www.python.org/ftp/python/3.6.0/Python-3.6.0.tgz
#RUN tar xvf Python-3.6.0.tgz
#RUN cd Python-3.6.0 && ./configure --enable-optimizations --enable-loadable-sqlite-extensions && make -j8 && make altinstall
#RUN wget https://bootstrap.pypa.io/get-pip.py && python3.6 get-pip.py
#RUN rm -f Python-3.6.0.tgz
RUN apt-get install -y python3-dev python3-pip

RUN python3.4 -m pip install --upgrade pip
RUN python3.4 -m pip install Cython>=0.24 nilearn>=0.2.4 numpy>=1.12.1 scikit-learn>=0.18.2 scipy>=0.19.1 pytest>=2.9.2 seaborn>=0.7.1 nose>=1.3.6 tabulate>=0.7.5 ipython

# Install requirements for Python 3
RUN python3.4 -m pip install skggm --upgrade
RUN python3.4 -m pip install pynets>=0.3.1 --upgrade

RUN chmod 775 -R /usr/local/lib/python3.4
RUN chown root:main -R /usr/local/lib/python3.4
RUN chmod 777 /usr/local/bin/pynets_run.py

USER main
RUN python3.4 -m pip install pandas --upgrade --user

#ENV PYTHONPATH /usr/local/lib/python3.4/dist-packages:$PYTHONPATH

# Environment variable try to fix lapack issue
ENV LD_PRELOAD /usr/lib/x86_64-linux-gnu/libgfortran.so.3
