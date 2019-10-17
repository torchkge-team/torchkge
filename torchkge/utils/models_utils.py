# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
aboschin@enst.fr
"""

from torch import tensor, empty
from torch.nn import Embedding, Parameter
from torch.nn.init import xavier_uniform_


def init_embedding(n_vectors, dim):
    """Create a torch.nn.Embedding object with `n_vectors` samples and `dim` dimensions.
    """
    entity_embeddings = Embedding(n_vectors, dim)
    entity_embeddings.weight = Parameter(xavier_uniform_(empty(size=(n_vectors, dim))),
                                         requires_grad=True)
    return entity_embeddings


def get_true_targets(dictionary, e_idx, r_idx, true_idx, i):
    true_targets = dictionary[e_idx[i].item(), r_idx[i].item()].copy()
    if len(true_targets) == 1:
        return None
    true_targets.remove(true_idx[i].item())
    return tensor(list(true_targets)).long()



