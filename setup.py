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
    description="A Fully-Automated Workflow for Reproducible Ensemble Graph Analysis of Functional and Structural Connectomes",
    author="Derek Pisner",
    author_email='dpisner@utexas.edu',
    url='https://github.com/dPys/pynets',
    packages=['pynets'],
    entry_points={
        'console_scripts': [
            'pynets=pynets.pynets_run:main'
        ]
    },
    include_package_data=True,
    install_requires=['nilearn==0.6.0b0'] + requirements,
    dependency_links=['git+https://github.com/dPys/nilearn.git@enh/parc_conn#egg=nilearn-0.6.0b0'],
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
