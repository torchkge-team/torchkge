# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""

from ..exceptions import NoPreTrainedVersionError
from ..models import TransEModel, ComplExModel, RESCALModel
from ..utils import load_embeddings


def load_pretrained_transe(dataset, emb_dim=None, data_home=None):
    """Load a pretrained version of TransE model.

    Parameters
    ----------
    dataset: str
    emb_dim: int (opt, default None)
        Embedding dimension
    data_home: str (opt, default None)
        Path to the `torchkge_data` directory (containing data folders). Useful
        for pre-trained model loading.

    Returns
    -------
    model: `TorchKGE.model.translation.TransEModel`
        Pretrained version of TransE model.
    """
    dims = {'fb15k': 100, 'wn18rr': 100, 'fb15k237': 150, 'wdv5': 150, 'yago310': 200}
    try:
        if emb_dim is None:
            emb_dim = dims[dataset]
        else:
            try:
                assert dims[dataset] == emb_dim
            except AssertionError:
                raise NoPreTrainedVersionError('No pre-trained version of TransE for '
                                               '{} in dimension {}'.format(dataset, emb_dim))
    except KeyError:
        raise NoPreTrainedVersionError('No pre-trained version of TransE for {}'.format(dataset))

    state_dict = load_embeddings('transe', emb_dim, dataset, data_home)
    model = TransEModel(emb_dim,
                        n_entities=state_dict['ent_emb.weight'].shape[0],
                        n_relations=state_dict['rel_emb.weight'].shape[0],
                        dissimilarity_type='L2')
    model.load_state_dict(state_dict)

    return model


def load_pretrained_complex(dataset, emb_dim=None, data_home=None):
    """Load a pretrained version of ComplEx model.

    Parameters
    ----------
    dataset: str
    emb_dim: int (opt, default None)
        Embedding dimension
    data_home: str (opt, default None)
        Path to the `torchkge_data` directory (containing data folders). Useful
        for pre-trained model loading.

    Returns
    -------
    model: `TorchKGE.model.translation.ComplExModel`
        Pretrained version of ComplEx model.
    """
    dims = {'wn18rr': 200, 'fb15k237': 200, 'wdv5': 200, 'yago310': 200}
    try:
        if emb_dim is None:
            emb_dim = dims[dataset]
        else:
            try:
                assert dims[dataset] == emb_dim
            except AssertionError:
                raise NoPreTrainedVersionError('No pre-trained version of ComplEx for '
                                               '{} in dimension {}'.format(dataset, emb_dim))
    except KeyError:
        raise NoPreTrainedVersionError('No pre-trained version of ComplEx for {}'.format(dataset))

    state_dict = load_embeddings('complex', emb_dim, dataset, data_home)
    model = ComplExModel(emb_dim,
                         n_entities=state_dict['re_ent_emb.weight'].shape[0],
                         n_relations=state_dict['re_rel_emb.weight'].shape[0])
    model.load_state_dict(state_dict)

    return model


def load_pretrained_rescal(dataset, emb_dim=None, data_home=None):
    """Load a pretrained version of RESCAL model.

    Parameters
    ----------
    dataset: str
    emb_dim: int (opt, default None)
        Embedding dimension
    data_home: str (opt, default None)
        Path to the `torchkge_data` directory (containing data folders). Useful
        for pre-trained model loading.

    Returns
    -------
    model: `TorchKGE.model.translation.RESCALModel`
        Pretrained version of RESCAL model.
    """
    dims = {'wn18rr': 200, 'fb15k237': 200, 'yago310': 200}
    try:
        if emb_dim is None:
            emb_dim = dims[dataset]
        else:
            try:
                assert dims[dataset] == emb_dim
            except AssertionError:
                raise NoPreTrainedVersionError('No pre-trained version of RESCAL for '
                                               '{} in dimension {}'.format(dataset, emb_dim))
    except KeyError:
        raise NoPreTrainedVersionError('No pre-trained version of RESCAL for {}'.format(dataset))

    state_dict = load_embeddings('rescal', emb_dim, dataset, data_home)
    model = RESCALModel(emb_dim,
                         n_entities=state_dict['ent_emb.weight'].shape[0],
                         n_relations=state_dict['rel_mat.weight'].shape[0])
    model.load_state_dict(state_dict)

    return model
