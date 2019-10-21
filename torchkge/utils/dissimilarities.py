# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
aboschin@enst.fr
"""
from torch import abs, min, sqrt, cos
from math import pi


def l1_dissimilarity(a, b):
    """

    Parameters
    ----------
    a: torch.Tensor, dtype = float, shape = (n_facts, dim)
    b: torch.Tensor, dtype = float, shape = (n_facts, dim)

    Returns
    -------
    dist: torch.Tensor, dtype = float, shape = (n_facts)
        Tensor of the row_wise L1 distance.

    """
    return (a-b).norm(p=1, dim=1)


def l2_dissimilarity(a, b):
    """

    Parameters
    ----------
    a: torch.Tensor, dtype = float, shape = (n_facts, dim)
    b: torch.Tensor, dtype = float, shape = (n_facts, dim)

    Returns
    -------
    dist: torch.Tensor, dtype = float, shape = (n_facts)
        Tensor of the row_wise squared L2 distance.

    """
    return (a-b).norm(p=2, dim=1)**2


def l1_torus_dissimilarity(a, b):
    a, b = a.frac(), b.frac()
    return min(abs(a-b), 1 - abs(a-b)).sum(dim=1)


def l2_torus_dissimilarity(a, b):
    a, b = a.frac(), b.frac()
    return sqrt(min((a - b)**2, 1 - (a - b)**2).sum(dim=1))


def el2_torus_dissimilarity(a, b):
    tmp = min(a - b, 1 - (a-b))
    tmp = 2 * (1 - cos(2 * pi * tmp))
    return sqrt(tmp.sum(dim=1))
