# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
aboschin@enst.fr
"""

from torch import empty, bincount, cat, topk, zeros
from torch.nn import Embedding, Parameter
from torch.nn.init import xavier_uniform_


def init_embedding(n_vectors, dim):
    """Create a torch.nn.Embedding object with `n_vectors` samples and `dim` dimensions.
    """
    entity_embeddings = Embedding(n_vectors, dim)
    entity_embeddings.weight = Parameter(xavier_uniform_(empty(size=(n_vectors, dim))))
    return entity_embeddings


def get_mask(length, start, end):
    """Create a mask of length `length` filled with 0s except between indices `start` (included)\
    and `end` (excluded).

    Parameters
    ----------
    length: int
    start: int
    end: int

    Returns
    -------
    mask: torch.Tensor, shape=(length), dtype=byte
        Mask of length `length` filled with 0s except between indices `start` (included)\
        and `end` (excluded).
    """
    mask = zeros(length)
    mask[[i for i in range(start, end)]] = 1
    return mask.byte()


def get_rolling_matrix(x):
    """

    Parameters
    ----------
    x: torch.Tensor, shape=(b_size, dim)

    Returns
    -------
    mat: torch.Tensor, shape=(b_size, dim, dim)
        Rolling matrix sur that mat[i,j] = x[i - j mod(dim)]
    """
    b_size, dim = x.shape
    x = x.view(b_size, 1, dim)
    return cat([x.roll(i, dims=2) for i in range(dim)], dim=1)


def get_rank(data, true, low_values=False):
    """

    Parameters
    ----------
    data: torch.Tensor, dtype = float, shape = (n_sample, dimensions)
    true: torch.Tensor, dtype = int, shape = (n_sample)
    low_values: bool
        if True, best rank is the lowest score else it is the highest

    Returns
    -------
    ranks: torch.Tensor, dtype = int, shape = (n_sample)
        data[ranks[i]] = true[i]
    """
    true_data = data.gather(1, true.long().view(-1, 1))

    if low_values:
        return (data <= true_data).sum(dim=1)
    else:
        return (data >= true_data).sum(dim=1)


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


def pad_with_last_value(t, k):
    """Pad a tensor with its last value

    Parameters
    ----------
    t: torch.Tensor, shape = (n,m)
    k: integer

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
    a: torch.Tensor, shape = (n, m)
    b: torch.Tensor, shape = (k, l)

    Returns
    -------
    torch.Tensor of shape (n+k, max(m, l))
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


def process_dissimilarities(dissimilarities, true, k_max):
    """Compute the rank of the true entities and the best candidates for each fact.

    Parameters
    ----------
    dissimilarities: torch.Tensor, dtype = float, shape = (batch_size, n_ent)
    true: torch.Tensor, dtype = long, shape = (batch_size),
        index of the the true entities for current relation
    k_max: integer
        Maximum value of the Hit@K measure.

    Returns
    -------
    rank_true_entities: torch.Tensor, dtype = long, shape = (batch_size)
        Rank of the true entity among all possible entities (ranked by decreasing dissimilarity)
    sorted_candidates: torch.Tensor, dtype = long, shape = (batch_size, k_max)
        Top k_max entity candidates in term of smaller dissimilarity(h+r, t).
    """
    # return the rank of the true value along with the sorted top k_max candidates
    _, sorted_candidates = topk(dissimilarities, k_max, dim=1, largest=False, sorted=True)
    rank_true_entities = get_rank(dissimilarities, true)
    return rank_true_entities, sorted_candidates
