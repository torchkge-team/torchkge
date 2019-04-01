# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
armand.boschin@telecom-paristech.fr
"""

from torch import bincount


def compute_weight(mask, k):
    """
    :param mask:
    :param k:
    :return:
    """
    weight = bincount(mask)[mask]
    weight[weight > k] = k
    return 1 / weight.float()


def get_rank(data, true):
    """
    :param data: float tensor of shape (n_sample, dimensions)
    :param true: integer tensor of shape (n_sample,)
    :return: integer tensor of shape (n_sample,) such that data[return[i]] = true[i]
    """
    return (data == true.view((-1, 1))).argmax(dim=1)


class Config:
    def __init__(self, ent_emb_dim, rel_emb_dim, n_ent, n_rel, norm_type):
        self.entities_embedding_dimension = ent_emb_dim
        self.relations_embedding_dimension = rel_emb_dim
        self.number_entities = n_ent
        self.number_relations = n_rel
        self.norm_type = norm_type
