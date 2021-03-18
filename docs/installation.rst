.. include:: links.rst
.. role:: bash(code)
   :language: bash
.. role:: orange

------------
Installation
------------

There are many ways to use ``PyNets®``: in a `Docker Container`_,
in a `Singularity Container`_, using AWS Batch, or in a Manually
Prepared Environment (Python 3.6+).
Using a local container method is highly recommended.
Once you are ready to run pynets, see Usage_ for details.

---------------------
Hardware Requirements
---------------------
PyNets is designed for maximal scalability-- it can be run on a supercomputer,
but it can also be run on your laptop. Nevertheless, exploring a larger
grid-space of the connectome "multiverse" can be accomplished faster and more
easily on a supercomputer, even if optimization reveals that only one or a
few connectome samples are needed.

With these considerations in mind, the minimal hardware required to run PyNets is 4 vCPUs, at least 8 GB of
free RAM, and at least 15-20 GB of free disk space. However, the recommended hardware for ensemble sampling
is 8+ vCPU's, 16+ GB of RAM, and 20+ GB of disk space (i.e. high-end
desktops and laptops). On AWS and supercomputer clusters, PyNets hypothetically has infinite scalability--
because it relies on a forkserver for multiprocessing,
it will auto-optimize its concurrency based on the cores/ memory
made available to it.

    .. note::
        Another important ceiling to consider is I/O. Be sure that when you
        specify a safe working directory for the heavy metadata disk operations of PyNets.
        This can be set using the `-work` flag, and unless you have a really good reason,
        it should almost always be set to some variation of '/tmp'.

Docker Container
================

In order to run pynets in a Docker container, Docker must be `installed
<https://docs.docker.com/engine/installation/>`_.
Once Docker is installed, you can simply pull a pre-built image from dockerhub as follows: ::

    docker pull dpys/pynets:latest

or you can build a container yourself and test it interactively as follows: ::

    docker build -t pynets .

    docker run -ti --rm --privileged \
        --entrypoint /bin/bash
        -v '/tmp':'/tmp' \
        -v '/var/tmp':'/var/tmp' \
        -v '/input_files_local':'/inputs' \
        -v '/output_files_local':'/outputs' \
        pynets

See `External Dependencies`_ for more information (e.g., specific versions) on what is included in the latest Docker images.


Singularity Container
=====================

For security reasons, many HPCs (e.g., TACC) do not allow Docker containers, but do
allow `Singularity <https://github.com/singularityware/singularity>`_ containers.

Preparing a Singularity image (Singularity version >= 2.5)
----------------------------------------------------------
If the version of Singularity on your HPC is modern enough you can create Singularity
image directly on the HCP.
This is as simple as: ::

    singularity build /my_images/pynets-<version>.simg docker://dpys/pynets:<version>

Where ``<version>`` should be replaced with the desired version of PyNets that you want to download.


Preparing a Singularity image (Singularity version < 2.5)
---------------------------------------------------------
In this case, start with a machine (e.g., your personal computer) with Docker installed.
Use `docker2singularity <https://github.com/singularityware/docker2singularity>`_ to
create a singularity image.
You will need an active internet connection and some time. ::

    docker run --privileged -t --rm \
        -v '/var/run/docker.sock':'/var/run/docker.sock' \
        -v 'D:\host\path\where\to\output\singularity\image:/output' \
        singularityware/docker2singularity \
        dpys/pynets:<version>

Where ``<version>`` should be replaced with the desired version of PyNets that you want
to download.

Beware of the back slashes, expected for Windows systems.
For \*nix users the command translates as follows: ::

    docker run --privileged -t --rm \
        -V '/var/run/docker.sock':'/var/run/docker.sock' \
        -v '/absolute/path/to/output/folder':'/outputs' \
        singularityware/docker2singularity \
        dpys/pynets:<version>

Transfer the resulting Singularity image to the HPC, for example, using ``scp``. ::

    scp pynets*.img user@hcpserver.edu:/my_images


Manually Prepared Environment (Python 3.6+)
===========================================

.. warning::

   This method is not recommended! Make sure you would rather do this than
   use a `Docker Container`_ or a `Singularity Container`_.

Make sure all of pynets's `External Dependencies`_ are installed.
These tools must be installed and their binaries available in the
system's ``$PATH``.
A relatively interpretable description of how your environment can be set-up
is found in the `Dockerfile <https://github.com/dPys/PyNets/blob/master/Dockerfile>`_.

On a functional Python 3.6 (or above) environment with ``pip`` installed,
PyNets can be installed using the habitual command: ::

    [sudo] pip install pynets [--user]

or ::

    # Install git-lfs
    brew install git-lfs (macOS) or [sudo] apt-get install git-lfs (linux)
    git lfs install --skip-repo

    # Clone the repository and install
    git clone https://github.com/dpys/pynets
    cd PyNets
    [sudo] python setup.py install [--user]

External Dependencies
---------------------

PyNets is written using Python 3.6 (or above), and is based on
nipype_.

PyNets requires some other neuroimaging software tools that are
not handled by the Python's packaging system (Pypi) used to deploy
the ``pynets`` package:

- FSL_ (version >=5.0.9). See `<https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation>`_

    .. note::
        If you are using a debian/ubuntu OS, installing FSL can be installed using neurodebian: ::

        [sudo] curl -sSL http://neuro.debian.net/lists/stretch.us-tn.full >> /etc/apt/sources.list.d/neurodebian.sources.list
        [sudo] apt-key add {path to PyNets base directory}/docker/files/neurodebian.gpg
        [sudo] apt-key adv --refresh-keys --keyserver hkp://ha.pool.sks-keyservers.net 0xA5D32F012649A5A9 || true
        [sudo] apt-get update
        [sudo] apt-get install -y fsl-core
