# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""

from torch import tensor, empty
from torch.nn import Embedding, Parameter
from torch.nn.init import xavier_uniform_


def init_embedding(n_vectors, dim):
    """Create a torch.nn.Embedding object with `n_vectors` samples and `dim` dimensions.
    """
    entity_embeddings = Embedding(n_vectors, dim)
    entity_embeddings.weight = Parameter(xavier_uniform_(empty(size=(n_vectors, dim))), requires_grad=True)
    return entity_embeddings


def get_true_targets(dictionary, e_idx, r_idx, true_idx, i):
    """For a current index `i` of the batch, returns a tensor containing the indices of entities for which the triplet
    is an existing one (i.e. a true one under CWA).

    Parameters
    ----------
    dictionary: default dict
        Dictionary of keys (int, int) and values list of ints giving all possible entities for
        the (entity, relation) pair.
    e_idx: `torch.Tensor`, shape: (batch_size), dtype: `torch.long`
        Tensor containing the indices of entities.
    r_idx: `torch.Tensor`, shape: (batch_size), dtype: `torch.long`
        Tensor containing the indices of relations.
    true_idx: `torch.Tensor`, shape: (batch_size), dtype: `torch.long`
        Tensor containing the true entity for each sample.
    i: int
        Indicates which index of the batch is currently treated.

    Returns
    -------
    true_targets: `torch.Tensor`, shape: (batch_size), dtype: `torch.long`
        Tensor containing the indices of entities such that (e_idx[i], r_idx[i], true_target[any]) is a true fact.

    """
    true_targets = dictionary[e_idx[i].item(), r_idx[i].item()].copy()
    if len(true_targets) == 1:
        return None
    true_targets.remove(true_idx[i].item())
    return tensor(list(true_targets)).long()
