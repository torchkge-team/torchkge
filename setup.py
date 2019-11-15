#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = ['torch>=1.2.0', 'tqdm', 'pandas']

setup_requirements = []

test_requirements = []

dev_requirements = ['torch==1.3.1', 'pip', 'bumpversion==0.5.3', 'sphinx==1.8.5', 'sphinx_rtd_theme==0.4.3',
                    'numpydoc==0.9.1', 'wheel==0.33.6', 'watchdog==0.9.0', 'flake8==3.7.9', 'tox==3.14.0',
                    'coverage==4.5.4', 'twine==2.0.0']

setup(
    author="TorchKGE Developers",
    author_email='aboschin@enst.fr',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
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
    extra_require={'dev': dev_requirements},
    test_suite='tests',
    version='0.11.2',
    zip_safe=False,
)
