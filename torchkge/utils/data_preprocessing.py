# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
aboschin@enst.fr
"""

from torch import Tensor, tensor

from .operations import concatenate_diff_sizes


def get_dictionaries(df, ent=True):
    """Build entities or relations dictionaries.

    Parameters
    ----------
    df: pandas Dataframe
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
        return {ent: i for i, ent in enumerate(tmp)}
    else:
        tmp = list(df['rel'].unique())
        return {rel: i for i, rel in enumerate(tmp)}


def lists_from_dicts(dictionary, entities, relations, targets, cuda):
    """

    Parameters
    ----------
    dictionary: dict
        keys: (ent, rel), values: list of entities
    entities: torch.Tensor, dtype = long, shape = (batch_size)
        Heads (resp. tails) of facts.
    relations: torch.Tensor, dtype = long, shape = (batch_size)
        Relations of facts
    targets: torch.Tensor, dtype = long, shape = (batch_size)
        Tails (resp. heads) of facts.
    cuda: bool
        If True, result is returned as CUDA tensor.

    Returns
    -------
    result: torch.Tensor, dtype = long, shape = (k)
        k is the largest number of possible alternative to the target in a fact. This tensor\
        contains for each line (fact) the list of possible alternatives to the target. If there\
        are no alternatives, then the line is full of -1.

    """
    result = Tensor().long()

    if entities.is_cuda:
        result = result.cuda()

    for i in range(entities.shape[0]):
        current = dictionary[(entities[i].item(), relations[i].item())]
        current.remove(targets[i])
        if len(current) == 0:
            current.append(-1)
        current = tensor(current).long().view(1, -1)
        if entities.is_cuda:
            current = current.cuda()
        result = concatenate_diff_sizes(result, current)
    if cuda:
        return result.cuda()
    else:
        return result.cpu()


def get_tph(df):
    df = df.groupby(['from', 'rel']).count().groupby('rel').mean()
    df.reset_index(inplace=True)
    return {df.loc[i].values[0]: df.loc[i].values[1] for i in df.index}


def get_hpt(df):
    df = df.groupby(['rel', 'to']).count().groupby('rel').mean()
    df.reset_index(inplace=True)
    return {df.loc[i].values[0]: df.loc[i].values[1] for i in df.index}


def get_bern_probs(df):
    """Evaluate the Bernoulli probabilities for negative sampling as in the TransH original\
    paper by Wang et al. (2014) https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8531.

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe from which the torchkge.data.KnowledgeGraph object is built.

    Returns
    -------
    tph: dict
        keys: relations as they appear in the pandas dataframe df, values: sampling probabilities\
        as described by Wang et al. in their paper.

    """
    hpt = get_hpt(df)
    tph = get_tph(df)

    assert hpt.keys() == tph.keys()

    for k in tph.keys():
        tph[k] = tph[k] / (tph[k] + hpt[k])

    return tph
