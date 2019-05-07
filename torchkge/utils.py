# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
aboschin@enst.fr
"""

from torch import Tensor, bincount, cat, nonzero, topk


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
    """Build entities or relations dictionaries.

    Parameters
    ----------
    df : pandas Dataframe
        Data frame containing three columns [from, to, rel].
    ent : bool
        if True then ent2ix is returned, if False then rel2ix is returned.

    Returns
    -------
    dict : dictionary
        Either ent2ix or rel2ix.
    """
    if ent:
        tmp = list(set(df['from'].unique()).union(set(df['to'].unique())))
        return {ent: i for i, ent in enumerate(tmp)}
    else:
        tmp = list(df['rel'].unique())
        return {rel: i for i, rel in enumerate(tmp)}


def pad_with_last_value(t, k):
    """Pad a tensor with its last value

    Parameters
    ----------
    t : torch tensor, shape = (n,m)
    k : integer

    Returns
    -------
    tensor of shape (n, m+i) where the n added columns are replicates of the last one of t.

    """
    n, m = t.shape
    tmp = t[:, -1].reshape(n, 1).expand(n, k)
    return cat((t, tmp), dim=1)


def concatenate_diff_sizes(a, b):
    """Concatenate 2D tensors of different shape by padding last with last value the one  with
    shortest second dimension.

    Parameters
    ----------
    a : torch tensor, shape = (n, m)
    b : torch tensor, shape = (k, l)

    Returns
    -------
    torch tensor of shape (n+k, max(m, l))
    """
    try:
        _, i = a.shape
        _, j = b.shape
        if i < j:
            return cat((pad_with_last_value(a, j - i), b), dim=0)
        elif j < i:
            return cat((a, pad_with_last_value(b, i - j)), dim=0)
        else:
            return cat((a, b), dim=0)
    except ValueError:
        return cat((a, b), dim=0)


def lists_from_dicts(dictionary, entities, relations, targets):
    """

    Parameters
    ----------
    dictionary : dict
        keys : (ent, rel), values : list of entities
    entities : torch tensor, dtype = long, shape = (batch_size)
        Heads (resp. tails) of facts.
    relations : torch tensor, dtype = long, shape = (batch_size)
        Relations of facts
    targets : torch tensor, dtype = long, shape = (batch_size)
        Tails (resp. heads) of facts.

    Returns
    -------
    result : torch tensor, dtype = long, shape = (k)
        k is the largest number of possible alternative to the target in a fact.
        This tensor contains for each line (fact) the list of possible alternatives to the target.
        If there are no alternatives, then the line is full of -1.
    """
    result = Tensor().long()

    if entities.is_cuda:
        result = result.cuda()

    for i in range(entities.shape[0]):
        current = dictionary[(entities[i].item(), relations[i].item())]
        current.remove(targets[i])
        if len(current) == 0:
            current.append(-1)
        current = Tensor(current).long().view(1, -1)
        if entities.is_cuda:
            current = current.cuda()
        result = concatenate_diff_sizes(result, current)
    return result


def process_dissimilarities(dissimilarities, true, k_max):
    """Compute the rank of the true entities and the best candidates for each fact.

    Parameters
    ----------
    dissimilarities : torch tensor, dtype = float, shape = (batch_size, n_ent)
    true : torch tensor, dtype = long, shape = (batch_size),
        index of the the true entities for current relation
    k_max : integer
        Maximum value of the Hit@K measure.

    Returns
    -------
    rank_true_entities : torch tensor, dtype = long, shape = (batch_size)
        Rank of the true entity among all possible entities (ranked by decreasing dissimilarity)
    sorted_candidates : torch tensor, dtype = long, shape = (batch_size, k_max)
        Top k_max entity candidates in term of smaller dissimilarity(h+r, t).
    """
    # return the rank of the true value along with the sorted top k_max candidates
    _, sorted_candidates = topk(dissimilarities, k_max, dim=1, largest=False, sorted=True)
    rank_true_entities = get_rank(dissimilarities, true)
    return rank_true_entities, sorted_candidates


class Config:
    def __init__(self, ent_emb_dim, rel_emb_dim, n_ent, n_rel, norm_type):
        self.entities_embedding_dimension = ent_emb_dim
        self.relations_embedding_dimension = rel_emb_dim
        self.number_entities = n_ent
        self.number_relations = n_rel
        self.norm_type = norm_type
