#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=6.0', 'torch==1.0.1.post2', 'torchvision', 'tqdm==4.31.1']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="TorchKGE Developers",
    author_email='armand.boschin@telecom-paristech.fr',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
    ],
    description="TorchKGE : Knowledge Graph embedding in Python and Pytorch.",
    entry_points={
        'console_scripts': [
            'torchkge=torchkge.cli:main',
        ],
    },
    install_requires=requirements,
    license="BSD license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='torchkge',
    name='torchkge',
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/torchkge-team/torchkge',
    version='0.2.0',
    zip_safe=False,
)
