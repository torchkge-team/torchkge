# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
armand.boschin@telecom-paristech.fr
"""

from torch import Tensor, bincount, cat, nonzero, stack


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
    true_data = data.gather(1, true.long().view(-1, 1))

    return (data <= true_data).sum(dim=1)


def get_dictionaries(df, ent=True):
    if ent:
        tmp = list(set(df['from'].unique()).union(set(df['to'].unique())))
        return {ent: i for i, ent in enumerate(tmp)}
    else:
        tmp = list(df['rel'].unique())
        return {rel: i for i, rel in enumerate(tmp)}


def get_max(entities, relations):
    uniques, inverse = cat([entities.view(-1, 1),
                            relations.view(-1, 1)], dim=1).unique(dim=0,
                                                                  return_inverse=True,
                                                                  sorted=False)

    return bincount(inverse).max()


def compute_lists(entities, relations, target):
    """
    TODO
    Parameters
    ----------
    entities :
    relations :
    target :

    Returns
    -------

    """
    # drop duplicates of (entity, relation)
    uniques, inverse = cat([entities.view(-1, 1),
                            relations.view(-1, 1)], dim=1).unique(dim=0,
                                                                  return_inverse=True,
                                                                  sorted=False)

    padded_length = bincount(inverse).max()
    list_ = [target[nonzero(inverse == i).squeeze().view(-1)].long() for i in
             range(len(uniques))]

    # pad each tensor in the list so that they have the same length
    list_ = [cat((list_[i], Tensor([list_[i][-1].item()
                                    for _ in range(padded_length - len(list_[i]))]).long().cuda()))
             for i in range(len(list_))]

    return stack(list_)[inverse]


class Config:
    def __init__(self, ent_emb_dim, rel_emb_dim, n_ent, n_rel, norm_type):
        self.entities_embedding_dimension = ent_emb_dim
        self.relations_embedding_dimension = rel_emb_dim
        self.number_entities = n_ent
        self.number_relations = n_rel
        self.norm_type = norm_type
