# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""

from ..exceptions import NoPreTrainedVersionError
from ..models import TransEModel, ComplExModel
from ..utils import load_embeddings


def load_pretrained_transe(dataset, emb_dim, data_home=None):
    """Load a pretrained version of TransE model.

    Parameters
    ----------
    dataset: str
    emb_dim: int
        Embedding dimension
    data_home: str (opt, default None)
        Path to the `torchkge_data` directory (containing data folders). Useful
        for pre-trained model loading.

    Returns
    -------
    model: `TorchKGE.model.translation.TransEModel`
        Pretrained version of TransE model.
    """
    try:
        assert (dataset in {'fb15k', 'wn18rr'} and emb_dim == 100) \
            or (dataset == 'fb15k237' and emb_dim == 150) \
            or (dataset == 'wdv5' and emb_dim == 150) \
            or (dataset == 'yago310' and emb_dim == 200)

    except AssertionError:
        raise NoPreTrainedVersionError('No pre-trained version of TransE for '
                                       '{} in dimension {}'.format(dataset,
                                                                   emb_dim))

    state_dict = load_embeddings('transe', emb_dim, dataset, data_home)
    model = TransEModel(emb_dim,
                        n_entities=state_dict['ent_emb.weight'].shape[0],
                        n_relations=state_dict['rel_emb.weight'].shape[0],
                        dissimilarity_type='L2')
    model.load_state_dict(state_dict)

    return model


def load_pretrained_complex(dataset, emb_dim, data_home=None):
    """Load a pretrained version of ComplEx model.

    Parameters
    ----------
    dataset: str
    emb_dim: int
        Embedding dimension
    data_home: str (opt, default None)
        Path to the `torchkge_data` directory (containing data folders). Useful
        for pre-trained model loading.

    Returns
    -------
    model: `TorchKGE.model.translation.ComplExModel`
        Pretrained version of ComplEx model.
    """
    try:
        assert (dataset == 'wn18rr' and emb_dim == 200) \
            or (dataset == 'fb15k237' and emb_dim == 200) \
            or (dataset == 'wdv5' and emb_dim == 200)

    except AssertionError:
        raise NoPreTrainedVersionError('No pre-trained version of ComplEx for '
                                       '{} in dimension {}'.format(dataset,
                                                                   emb_dim))

    state_dict = load_embeddings('complex', emb_dim, dataset, data_home)
    model = ComplExModel(emb_dim,
                         n_entities=state_dict['re_ent_emb.weight'].shape[0],
                         n_relations=state_dict['re_rel_emb.weight'].shape[0])
    model.load_state_dict(state_dict)

    return model
