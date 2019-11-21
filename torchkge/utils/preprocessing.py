# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""

from torch import cat, tensor

from torchkge.utils.operations import groupby_mean, groupby_count


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
    tmp = groupby_count(t, by=[0, 2])  # group by ['from', 'rel']
    tmp = tensor([[i[1].item(), tmp[i]] for i in tmp.keys()])
    return groupby_mean(tmp, by=0)


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
    tmp = groupby_count(t, by=[1, 2])  # group by ['to', 'rel']
    tmp = tensor([[i[1].item(), tmp[i]] for i in tmp.keys()])
    return groupby_mean(tmp, by=0)


def get_bernoulli_probs(kg):
    """Evaluate the Bernoulli probabilities for negative sampling as in the TransH original\
    paper by Wang et al. (2014) https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8531.

    Parameters
    ----------
    kg: `torchkge.data.KnowledgeGraph.KnowledgeGraph`

    Returns
    -------
    tph: dict
        keys: relations , values: sampling probabilities as described by Wang et al. in their paper.

    """
    t = cat((kg.head_idx.view(-1, 1), kg.tail_idx.view(-1, 1), kg.relations.view(-1, 1)), dim=1)

    hpt = get_hpt(t)
    tph = get_tph(t)

    assert hpt.keys() == tph.keys()

    for k in tph.keys():
        tph[k] = tph[k] / (tph[k] + hpt[k])

    return tph
