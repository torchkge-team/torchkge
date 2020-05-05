# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""

from math import pi
from torch import abs, cos, min


def l1_dissimilarity(a, b):
    """Compute dissimilarity between rows of  `a` and `b` as :math:`||a-b||_1`.

    """
    assert len(a.shape) == len(b.shape)
    return (a-b).norm(p=1, dim=-1)


def l2_dissimilarity(a, b):
    """Compute dissimilarity between rows of  `a` and `b` as
    :math:`||a-b||_2^2`.

    """
    assert len(a.shape) == len(b.shape)
    return (a-b).norm(p=2, dim=-1)**2


def l1_torus_dissimilarity(a, b):
    """See `paper by Ebisu et al. <https://www.semanticscholar.org/paper/TorusE%3A-Knowledge-Graph-Embedding-on-a-Lie-Group-Ebisu-Ichise/990334cf76845e2da64d3baa10b0a671e433d4b6>`_
    for details about the definition of this dissimilarity function.

    """
    assert len(a.shape) == len(b.shape)
    return 2 * min(abs(a - b), 1 - abs(a - b)).sum(dim=-1)


def l2_torus_dissimilarity(a, b):
    """See `paper by Ebisu et al. <https://www.semanticscholar.org/paper/TorusE%3A-Knowledge-Graph-Embedding-on-a-Lie-Group-Ebisu-Ichise/990334cf76845e2da64d3baa10b0a671e433d4b6>`_
    for details about the definition of this dissimilarity function.

    """
    assert len(a.shape) == len(b.shape)
    return 4 * min((a - b) ** 2, 1 - (a - b) ** 2).sum(dim=-1)


def el2_torus_dissimilarity(a, b):
    """See `paper by Ebisu et al. <https://www.semanticscholar.org/paper/TorusE%3A-Knowledge-Graph-Embedding-on-a-Lie-Group-Ebisu-Ichise/990334cf76845e2da64d3baa10b0a671e433d4b6>`_
    for details about the definition of this dissimilarity function.

    """
    assert len(a.shape) == len(b.shape)
    tmp = min(a - b, 1 - (a - b))
    tmp = 2 * (1 - cos(2 * pi * tmp))
    return tmp.sum(dim=-1) / 4
