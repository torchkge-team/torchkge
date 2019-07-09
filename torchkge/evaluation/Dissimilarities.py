# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
aboschin@enst.fr
"""


def l1_dissimilarity(a, b):
    """

    Parameters
    ----------
    a: torch.Tensor, dtype = float, shape = (n_sample, dim)
    b: torch.Tensor, dtype = float, shape = (n_sample, dim)

    Returns
    -------
    dist: torch.Tensor, dtype = float, shape = (n_sample)
        Tensor of the row_wise L1 distance.

    """
    return (a-b).norm(p=1, dim=1)


def l2_dissimilarity(a, b):
    """

    Parameters
    ----------
    a: torch.Tensor, dtype = float, shape = (n_sample, dim)
    b: torch.Tensor, dtype = float, shape = (n_sample, dim)

    Returns
    -------
    dist: torch.Tensor, dtype = float, shape = (n_sample)
        Tensor of the row_wise squared L2 distance.

    """
    return (a-b).norm(p=2, dim=1)**2
