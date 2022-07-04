# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""

from collections import defaultdict
from pandas import DataFrame
from torch import zeros, cat
from numpy import unique


def get_mask(length, start, end):
    """Create a mask of length `length` filled with 0s except between indices
    `start` (included) and `end` (excluded).

    Parameters
    ----------
    length: int
        Length of the mask to be created.
    start: int
        First index (included) where the mask will be filled with 0s.
    end: int
        Last index (excluded) where the mask will be filled with 0s.

    Returns
    -------
    mask: `torch.Tensor`, shape: (length), dtype: `torch.bool`
        Mask of length `length` filled with 0s except between indices `start`
        (included) and `end` (excluded).
    """
    mask = zeros(length)
    mask[[i for i in range(start, end)]] = 1
    return mask.bool()


def get_rank(data, true, low_values=False):
    """Computes the rank of entity at index true[i]. If the rank is k then
    there are k-1 entities with better (higher or lower) value in data.

    Parameters
    ----------
    data: `torch.Tensor`, dtype: `torch.float`, shape: (n_facts, dimensions)
        Scores for each entity.
    true: `torch.Tensor`, dtype: `torch.int`, shape: (n_facts)
        true[i] is the index of the true entity for test i of the batch.
    low_values: bool, optional (default=False)
        if True, best rank is the lowest score else it is the highest.

    Returns
    -------
    ranks: `torch.Tensor`, dtype: `torch.int`, shape: (n_facts)
        ranks[i] - 1 is the number of entities which have better (or same)
        scores in data than the one and index true[i]
    """
    true_data = data.gather(1, true.long().view(-1, 1))

    if low_values:
        return (data <= true_data).sum(dim=1)
    else:
        return (data >= true_data).sum(dim=1)


def get_dictionaries(df, ent=True):
    """Build entities or relations dictionaries.

    Parameters
    ----------
    df: `pandas.DataFrame`
        Data frame containing three columns [from, to, rel].
    ent: bool
        if True then ent2ix is returned, if False then rel2ix is returned.

    Returns
    -------
    dict: dictionary
        Either ent2ix or rel2ix.

    """
    if ent:
        tmp = list(set(df['from'].unique()).union(set(df['to'].unique())))
        return {ent: i for i, ent in enumerate(sorted(tmp))}
    else:
        tmp = list(df['rel'].unique())
        return {rel: i for i, rel in enumerate(sorted(tmp))}


def extend_dicts(kg, attributes):
    ent2ix = {k: v for k, v in kg.ent2ix.items()}
    rel2ix = {k: v for k, v in kg.rel2ix.items()}

    assert len(ent2ix) == len(unique(list(ent2ix.values())))
    assert len(rel2ix) == len(unique(list(rel2ix.values())))

    tmp = list(set(attributes['from'].unique()).union(set(attributes['to'].unique())))
    for ent in sorted(tmp):
        if ent in ent2ix.keys():
            continue
        else:
            ent2ix[ent] = len(ent2ix)

    tmp = list(attributes['rel'].unique())
    for rel in sorted(tmp):
        if rel in rel2ix.keys():
            continue
        else:
            rel2ix[rel] = len(rel2ix)
    assert len(ent2ix) == len(unique(list(ent2ix.values())))
    assert len(rel2ix) == len(unique(list(rel2ix.values())))
    assert len(ent2ix) == (max(list(ent2ix.values())) + 1)
    assert len(rel2ix) == (max(list(rel2ix.values())) + 1)

    return ent2ix, rel2ix


def get_tph(t):
    """Get the average number of tail per heads for each relation.

    Parameters
    ----------
    t: `torch.Tensor`, dtype: `torch.long`, shape: (b_size, 3)
        First column contains head indices, second tails and third relations.
    Returns
    -------
    d: dict
        keys: relation indices, values: average number of tail per heads.
    """
    df = DataFrame(t.numpy(), columns=['from', 'to', 'rel'])
    df = df.groupby(['from', 'rel']).count().groupby('rel').mean()
    df.reset_index(inplace=True)
    return {df.loc[i].values[0]: df.loc[i].values[1] for i in df.index}


def get_hpt(t):
    """Get the average number of head per tails for each relation.

    Parameters
    ----------
    t: `torch.Tensor`, dtype: `torch.long`, shape: (b_size, 3)
        First column contains head indices, second tails and third relations.
    Returns
    -------
    d: dict
        keys: relation indices, values: average number of head per tails.
    """
    df = DataFrame(t.numpy(), columns=['from', 'to', 'rel'])
    df = df.groupby(['rel', 'to']).count().groupby('rel').mean()
    df.reset_index(inplace=True)
    return {df.loc[i].values[0]: df.loc[i].values[1] for i in df.index}


def get_bernoulli_probs(kg):
    """Evaluate the Bernoulli probabilities for negative sampling as in the
    TransH original paper by Wang et al. (2014).

    Parameters
    ----------
    kg: `torchkge.data_structures.KnowledgeGraph`

    Returns
    -------
    tph: dict
        keys: relations , values: sampling probabilities as described by
        Wang et al. in their paper.

    """
    t = cat((kg.head_idx.view(-1, 1),
             kg.tail_idx.view(-1, 1),
             kg.relations.view(-1, 1)), dim=1)

    hpt = get_hpt(t)
    tph = get_tph(t)

    assert hpt.keys() == tph.keys()

    for k in tph.keys():
        tph[k] = tph[k] / (tph[k] + hpt[k])

    return tph


def get_fitlering_dictionaries(kg, kg_te=None):
    dict_of_heads = defaultdict(set)
    dict_of_tails = defaultdict(set)
    dict_of_rels = defaultdict(set)
    for i in range(kg.n_facts):
        dict_of_heads[(kg.tail_idx[i].item(), kg.relations[i].item())].add(kg.head_idx[i].item())
        dict_of_tails[(kg.head_idx[i].item(), kg.relations[i].item())].add(kg.tail_idx[i].item())
        dict_of_rels[(kg.head_idx[i].item(), kg.tail_idx[i].item())].add(kg.relations[i].item())
    if kg_te is not None:
        for i in range(kg_te.n_facts):
            dict_of_rels[(kg_te.tail_idx[i].item(), kg_te.relations[i].item())].add(kg_te.head_idx[i].item())
            dict_of_rels[(kg_te.head_idx[i].item(), kg_te.relations[i].item())].add(kg_te.tail_idx[i].item())
            dict_of_rels[(kg_te.head_idx[i].item(), kg_te.tail_idx[i].item())].add(kg_te.relations[i].item())
    return dict_of_heads, dict_of_tails, dict_of_rels