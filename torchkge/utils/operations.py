# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""

from torch import cat, zeros, max, Size, mm, diag, eye


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
    """

    Parameters
    ----------
    data: `torch.Tensor`, dtype: `torch.float`, shape: (n_facts, dimensions)
    true: `torch.Tensor`, dtype: `torch.int`, shape: (n_facts)
    low_values: bool
        if True, best rank is the lowest score else it is the highest

    Returns
    -------
    ranks: `torch.Tensor`, dtype: `torch.int`, shape: (n_facts)
        data[ranks[i]] = true[i]
    """
    true_data = data.gather(1, true.long().view(-1, 1))

    if low_values:
        return (data <= true_data).sum(dim=1)
    else:
        return (data >= true_data).sum(dim=1)


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


def one_hot(m):
    """Get one-hot encoding of m.

    Parameters
    ----------
    m: `torch.Tensor`, shape: (i)

    Returns
    -------
    one_hot: `torch.Tensor`, shape: (i, max(m) + 1)
    """
    y = zeros(Size((m.shape[0], max(m) + 1)))
    return y.scatter(1, m.reshape(m.shape[0], 1), 1)


def get_col(t, by):
    """Return the column index not included in `by`. `t` can have shape (n, 2) (resp. (n, 3)) and
    `by` is a integer (resp. list of integers of length 2).

    Parameters
    ----------
    t: `torch.Tensor`, shape: (n, 2) or (n, 3)
    by: int or list of ints


    Returns
    -------
    int

    """
    assert len(t.shape) == 2
    assert (type(by) == int or type(by) == list)
    n_cols = t.shape[1]

    if type(by) == int:
        assert n_cols == 2
        if by == 1:
            return 0
        else:
            return 1
    else:
        n_keys = len(by)
        assert (n_cols - n_keys == 1)
        return (set(range(n_cols)) - set(by)).pop()


def groupby_count(t, by):
    """Groupby operation with count reduction on a 2D tensor of either two or three columns. The by variable contains
    respectively an integer or a list of integers of length 2.

    Parameters
    ----------
    t: `torch.Tensor`, shape: (i, j) or (i, j, k)
    by: int of list on ints

    Returns
    -------
    dict

    """
    uniques, inverse = t[:, by].unique(dim=0, return_inverse=True, sorted=False)
    mask = one_hot(inverse)
    tmp = mm(eye(t.shape[0]), mask).long()
    values = tmp.sum(dim=0)

    if type(by) == int:
        return {uniques[i].item(): values[i].item() for i in range(len(uniques))}
    else:
        return {uniques[i]: values[i].item() for i in range(len(uniques))}


def groupby_mean(t, by):
    """Groupby operation with mean reduction on a 2D tensor of either two or three columns. The by variable contains
    respectively an integer or a list of integers of length 2.

    Parameters
    ----------
    t: `torch.Tensor`, shape: (i, j) or (i, j, k)
    by: int of list on ints

    Returns
    -------
    dict

    """
    uniques, inverse = t[:, by].unique(dim=0, return_inverse=True, sorted=False)
    mask = one_hot(inverse)
    values = mm(diag(t[:, get_col(t, by)]).float(), mask)

    res = dict()
    for i in range(len(uniques)):
        tmp = values[mask[:, i].bool(), i].numpy()
        k = uniques[i]
        if type(by) == int:
            k = k.item()
        if (tmp != 0).sum() == 0:
            res[k] = 0
        else:
            res[k] = tmp[tmp != 0].mean()
        res[k] = tmp.mean()

    return res
