========
TorchKGE
========

.. image:: https://graphs.telecom-paristech.fr/images/logo_torchKGE_small.png
    :align: right
    :width: 100px
    :alt: logo torchkge

.. image:: https://img.shields.io/pypi/v/torchkge.svg
        :target: https://pypi.python.org/pypi/torchkge

.. image:: https://travis-ci.org/torchkge-team/torchkge.svg?branch=master
    :target: https://travis-ci.org/torchkge-team/torchkge

.. image:: https://readthedocs.org/projects/torchkge/badge/?version=latest
    :target: https://torchkge.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://pyup.io/repos/github/torchkge-team/torchkge/shield.svg
     :target: https://pyup.io/repos/github/torchkge-team/torchkge/
     :alt: Updates



TorchKGE: Knowledge Graph embedding in Python and Pytorch.

TorchKGE is a Python module for knowledge graph (KG) embedding relying solely on Pytorch. This package provides
researchers and engineers with a clean and efficient API to design and test new models. It features a KG data structure,
simple model interfaces and modules for negative sampling and model evaluation. Its main strength is a highly efficient
evaluation module for the  link prediction task,  a central application of KG embedding. It has been observed to be up
to five times faster than `Ampligraph <https://docs.ampligraph.org/>`_ and twenty-four times faster than
`OpenKE <https://github.com/thunlp/OpenKE>`_. Various KG embedding models are also already implemented. Special
attention has been paid to code efficiency and simplicity, documentation and API consistency. It is distributed using
PyPI under BSD license.

A paper will be out soon !

* Free software: BSD license
* Documentation: https://torchkge.readthedocs.io.
