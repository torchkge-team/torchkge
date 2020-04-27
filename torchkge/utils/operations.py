# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""

from torch import zeros


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
        ranks[i] - 1 is the number of entities which have better scores in data
        than the one and index true[i]
    """
    true_data = data.gather(1, true.long().view(-1, 1))

    if low_values:
        return (data < true_data).sum(dim=1) + 1
    else:
        return (data > true_data).sum(dim=1) + 1
