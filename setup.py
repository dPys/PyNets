#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup
import versioneer
from pynets.__about__ import __version__
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
    description="A Reproducible Workflow for Structural and Functional "
                "Connectome Ensemble Learning",
    author="Derek Pisner",
    author_email='dpysalexander@gmail.com',
    url='https://github.com/dPys/pynets',
    packages=['pynets'],
    entry_points={
        'console_scripts': [
            'pynets=pynets.cli.pynets_run:main',
            'pynets_cloud=pynets.cli.pynets_cloud:main',
            'pynets_bids=pynets.cli.pynets_bids:main',
            'pynets_benchmark=pynets.cli.pynets_benchmark:main',
            'pynets_predict=pynets.cli.pynets_predict:main',
            'pynets_config=pynets.cli.pynets_config:main'
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    dependency_links=['git+https://github.com/dPys/nilearn.git@enh/parc_conn#egg=nilearn',
                      'git+https://github.com/dPys/deepbrain.git@master#egg=deepbrain'],
    license="GNU Affero General Public License v3 or later (AGPLv3+)",
    zip_safe=False,
    keywords='pynets',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
    test_suite='tests'
)
