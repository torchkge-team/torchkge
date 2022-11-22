# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""

from torch import tensor
from torch.nn import Embedding
from torch.nn.init import xavier_uniform_

import pickle
import tarfile

from torchkge.utils import get_data_home, safe_extract

from os import makedirs, remove
from os.path import exists
from urllib.request import urlretrieve


def init_embedding(n_vectors, dim):
    """Create a torch.nn.Embedding object with `n_vectors` samples and `dim`
    dimensions. It is then initialized with Xavier uniform distribution.
    """
    entity_embeddings = Embedding(n_vectors, dim)
    xavier_uniform_(entity_embeddings.weight.data)

    return entity_embeddings


def load_embeddings(model, dim, dataset, data_home=None):

    if data_home is None:
        data_home = get_data_home()
    data_path = data_home + '/models/'
    targz_file = data_path + '{}_{}_{}.tar.gz'.format(model, dataset, dim)
    pkl_file = data_path + '{}_{}_{}.pkl'.format(model, dataset, dim)
    if not exists(pkl_file):
        if not exists(data_path):
            makedirs(data_path, exist_ok=True)
        urlretrieve("https://graphs.telecom-paris.fr/data/torchkge/models/{}_{}_{}.tar.gz".format(model, dataset, dim),
                    targz_file)
        with tarfile.open(targz_file, 'r') as tf:
            safe_extract(tf, data_path)
        remove(targz_file)

    with open(pkl_file, 'rb') as f:
        state_dict = pickle.load(f)

    return state_dict


def get_true_targets(dictionary, key1, key2, true_idx, i):
    """For a current index `i` of the batch, returns a tensor containing the
    indices of entities for which the triplet is an existing one (i.e. a true
    one under CWA).

    Parameters
    ----------
    dictionary: default dict
        Dictionary of keys (int, int) and values list of ints giving all
        possible entities for the (entity, relation) pair.
    key1: torch.Tensor, shape: (batch_size), dtype: torch.long
    key2: torch.Tensor, shape: (batch_size), dtype: torch.long
    true_idx: torch.Tensor, shape: (batch_size), dtype: torch.long
        Tensor containing the true entity for each sample.
    i: int
        Indicates which index of the batch is currently treated.

    Returns
    -------
    true_targets: torch.Tensor, shape: (batch_size), dtype: torch.long
        Tensor containing the indices of entities such that
        (e_idx[i], r_idx[i], true_target[any]) is a true fact.

    """
    try:
        true_targets = dictionary[key1[i].item(), key2[i].item()].copy()
        if true_idx is not None:
            true_targets.remove(true_idx[i].item())
            if len(true_targets) > 0:
                return tensor(list(true_targets)).long()
            else:
                return None
        else:
            return tensor(list(true_targets)).long()
    except KeyError:
        return None


def filter_scores(scores, dictionary, key1, key2, true_idx):
    # filter out the true negative samples by assigning - inf score.
    b_size = scores.shape[0]
    filt_scores = scores.clone()

    for i in range(b_size):
        true_targets = get_true_targets(dictionary, key1, key2, true_idx, i)
        if true_targets is None:
            continue
        filt_scores[i][true_targets] = - float('Inf')

    return filt_scores
