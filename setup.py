#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = [
    'Click>=6.0',
    'scipy>=0.19.0',
    'nipype>=1.1.0',
    'numpy>=1.12.1',
    'seaborn>=0.7.1',
    'matplotlib>=2.0.0',
    'nilearn>=0.5.0',
    'pandas>=0.24.2',
    'networkx>=2.0',
    'nibabel>=2.1.0',
    'scikit_learn>=0.18.2',
    'pathlib>=1.0.1',
    'setuptools>=39.0.1',
    'configparser>=3.5.0',
    'PyYAML>=3.12',
    'boto3>=1.9.111',
    'colorama>=0.3.9',
    'pybids>=0.6.4',
    'graspy==0.0.2'
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
    version='0.7.39',
    description="A Fully-Automated Workflow for Reproducible Graph Analysis of Functional and Structural Connectomes",
    #long_description=readme + '\n\n',
    author="Derek Pisner",
    author_email='dpisner@utexas.edu',
    url='https://github.com/dPys/pynets',
    packages=['pynets'],
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
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
    scripts=["pynets/pynets_run.py", "pynets/runconfig.yaml"]
)
