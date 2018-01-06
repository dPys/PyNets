#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'Click>=6.0',
    'scipy>=0.19.0',
    'nipype>=0.12.1',
    'numpy>=1.12.1',
    'nilearn>=0.2.6',
    'seaborn>=0.7.1',
    'matplotlib>=2.0.0',
    'pandas>=0.19.2',
    'networkx>=2.0',
    'nibabel>=2.1.0',
    'scikit_learn>=0.18.2',
    'pathlib>=1.0.1',
    'setuptools>=36.2.7',
    'configparser>=3.5.0'
]

setup_requirements = [
    'pytest-runner',
    # TODO(dpisner453): put setup requirements (distutils extensions, etc.) here
]

test_requirements = [
    'pytest',
    # TODO: put package test requirements here
]

setup(
    name='pynets',
    version='0.2.5',
    description="A Python-Powered Workflow for Network Analysis of rsfMRI and dMRI",
    long_description=readme + '\n\n' + history,
    author="Derek Pisner",
    author_email='dpisner@utexas.edu',
    url='https://github.com/dpisner453/pynets',
    packages=find_packages(include=['pynets']),
    entry_points={
        'console_scripts': [
            'pynets=pynets.cli:main'
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    license="GNU General Public License v3",
    zip_safe=False,
    keywords='pynets',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
    scripts=["pynets/pynets_run.py"]
)
