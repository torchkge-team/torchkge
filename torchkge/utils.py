# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
armand.boschin@telecom-paristech.fr
"""

from torch import bincount


def compute_weight(mask, k):
    """

    Parameters
    ----------
    mask
    k

    Returns
    -------

    """
    weight = bincount(mask)[mask]
    weight[weight > k] = k
    return 1 / weight.float()


def get_rank(data, true):
    """

    Parameters
    ----------
    data : torch tensor, dtype = float, shape = (n_sample, dimensions)
    true : torch tensor, dtype = int, shape = (n_sample)

    Returns
    -------
    ranks : torch tensor, dtype = int, shape = (n_sample)
        data[ranks[i]] = true[i]
    """
    return (data == true.view((-1, 1))).argmax(dim=1)


class Config:
    def __init__(self, ent_emb_dim, rel_emb_dim, n_ent, n_rel, norm_type):
        self.entities_embedding_dimension = ent_emb_dim
        self.relations_embedding_dimension = rel_emb_dim
        self.number_entities = n_ent
        self.number_relations = n_rel
        self.norm_type = norm_type
