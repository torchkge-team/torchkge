# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""

from torch import cat, zeros


def get_mask(length, start, end):
    """Create a mask of length `length` filled with 0s except between indices `start` (included)\
    and `end` (excluded).

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
        Mask of length `length` filled with 0s except between indices `start` (included)\
        and `end` (excluded).
    """
    mask = zeros(length)
    mask[[i for i in range(start, end)]] = 1
    return mask.bool()


def get_rank(data, true, low_values=False):
    """Computes the rank of entity at index true[i]. If the rank is k then there are k-1 entities with better (higher \
    or lower) value in data.

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
        ranks[i] - 1 is the number of entities which have better scores in data than the one and index true[i]
    """
    true_data = data.gather(1, true.long().view(-1, 1))

    if low_values:
        return (data < true_data).sum(dim=1) + 1
    else:
        return (data > true_data).sum(dim=1) + 1


def get_rolling_matrix(x):
    """Build a rolling matrix.

    Parameters
    ----------
    x: `torch.Tensor`, shape: (b_size, dim)

    Returns
    -------
    mat: `torch.Tensor`, shape: (b_size, dim, dim)
        Rolling matrix such that mat[i,j] = x[i - j mod(dim)]
    """
    b_size, dim = x.shape
    x = x.view(b_size, 1, dim)
    return cat([x.roll(i, dims=2) for i in range(dim)], dim=1)


def get_n_batch(n, b_size):
    n_batch = n // b_size
    if n % b_size > 0:
        n_batch += 1
    return n_batch


def get_batches(h, t, r, b_size):
    """
    TODO
    Parameters
    ----------
    h
    t
    r
    b_size

    Returns
    -------

    """
    assert len(h) == len(t) == len(r)
    n_batch = get_n_batch(len(h), b_size)
    for i in range(n_batch):
        yield h[i * b_size: (i + 1) * b_size], t[i * b_size: (i + 1) * b_size], r[i * b_size: (i + 1) * b_size]
