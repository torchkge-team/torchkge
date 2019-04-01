# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
armand.boschin@telecom-paristech.fr
"""


def l1_dissimilarity(a, b):
    """
    :param a: tensor of shape (n_sample, dim)
    :param b: tensor of shape (n_sample, dim)
    :return: tensor of shape (n_sample) of the row_wise L1 distance
    """
    return (a-b).norm(p=1, dim=1)


def l2_dissimilarity(a, b):
    """
    :param a: tensor of shape (n_sample, dim)
    :param b: tensor of shape (n_sample, dim)
    :return: tensor of shape (n_sample) of the row_wise squared L2 distance.
    """
    return (a-b).norm(p=2, dim=1)**2
