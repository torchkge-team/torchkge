#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = ['torch>=1.2.0', 'tqdm>=4.64', 'pandas>=1.4', 'numpy>=1.22']

setup_requirements = ['pytest-runner']

test_requirements = ['pytest']

setup(
    author="TorchKGE Developers",
    author_email='aboschin@enst.fr',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10'
    ],
    description="Knowledge Graph embedding in Python and PyTorch.",
    license="BSD license",
    long_description=readme,
    include_package_data=True,
    keywords='torchkge',
    name='torchkge',
    url='https://github.com/torchkge-team/torchkge',
    packages=find_packages(),
    install_requires=requirements,
    setup_requires=setup_requirements,
    tests_require=test_requirements,
    test_suite='tests',
    version='0.17.7',
    zip_safe=False,
)
