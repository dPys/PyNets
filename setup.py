#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
#from pynets.__about__ import __version__, DOWNLOAD_URL
from pynets.__about__ import __version__
import versioneer
cmdclass = versioneer.get_cmdclass()

with open('README.rst') as readme_file:
    readme = readme_file.read()
    readme_file.close()

with open('requirements.txt') as f:
    requirements = [r for r in f.read().splitlines() if 'git+' not in r]
    f.close()

setup(
    name='pynets',
    version=__version__,
    cmdclass=cmdclass,
    description="A Reproducible Workflow for Structural and Functional Connectome Ensemble Learning",
    author="Derek Pisner",
    author_email='dpisner@utexas.edu',
    url='https://github.com/dPys/pynets',
    packages=['pynets'],
    entry_points={
        'console_scripts': [
            'pynets=pynets.cli.pynets_run:main',
            'pynets_cloud=pynets.cli.pynets_cloud:main',
            'pynets_bids=pynets.cli.pynets_bids:main',
            'pynets_collect=pynets.cli.pynets_collect:main'
            'pynets_benchmark=pynets.cli.pynets_benchmark:main',
            'pynets_predict=pynets.cli.pynets_predict:main'
        ]
    },
    include_package_data=True,
    install_requires=['nilearn==0.6.2', 'deepbrain==0.1.0'] + requirements,
    dependency_links=['git+https://github.com/dPys/nilearn.git@enh/parc_conn#egg=nilearn-0.6.2',
                      'git+https://github.com/dPys/deepbrain.git@master#egg=deepbrain-0.1.0'],
    license="GNU General Public License v3",
    zip_safe=False,
    keywords='pynets',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    test_suite='tests'
)
