# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""

from torch import nn, cat

from ..models.interfaces import Model
from ..utils import init_embedding


class ConvKBModel(Model):
    """Implementation of ConvKB model detailed in 2018 paper by Nguyen et al..
    This class inherits from the :class:`torchkge.models.interfaces.Model`
    interface. It then has its attributes as well.


    References
    ----------
    * Nguyen, D. Q., Nguyen, T. D., Nguyen, D. Q., and Phung, D.
      `A Novel Embed- ding Model for Knowledge Base Completion Based on
      Convolutional Neural Network.
      <https://arxiv.org/abs/1712.02121>`_
      In Proceedings of the 2018 Conference of the North American Chapter of
      the Association for Computational Linguistics: Human Language
      Technologies (2018), vol. 2, pp. 327–333.

    Parameters
    ----------
    emb_dim: int
        Dimension of embedding space.
    n_filters: int
        Number of filters used for convolution.
    n_entities: int
        Number of entities in the current data set.
    n_relations: int
        Number of relations in the current data set.

    Attributes
    ----------
    ent_emb: torch.nn.Embedding, shape: (n_ent, emb_dim)
        Embeddings of the entities, initialized with Xavier uniform
        distribution and then normalized.
    rel_emb: torch.nn.Embedding, shape: (n_rel, emb_dim)
        Embeddings of the relations, initialized with Xavier uniform
        distribution.

    """

    def __init__(self, emb_dim, n_filters, n_entities, n_relations):
        super().__init__(n_entities, n_relations)
        self.emb_dim = emb_dim

        self.ent_emb = init_embedding(self.n_ent, self.emb_dim)
        self.rel_emb = init_embedding(self.n_rel, self.emb_dim)

        self.convlayer = nn.Sequential(nn.Conv1d(3, n_filters, 1, stride=1),
                                       nn.ReLU())
        self.output = nn.Sequential(nn.Linear(emb_dim * n_filters, 2),
                                    nn.Softmax(dim=1))

    def scoring_function(self, h_idx, t_idx, r_idx):
        """Compute the scoring function for the triplets given as argument:
        by applying convolutions to the concatenation of the embeddings. See
        referenced paper for more details on the score. See
        torchkge.models.interfaces.Models for more details on the API.

        """
        b_size = h_idx.shape[0]

        h = self.ent_emb(h_idx).view(b_size, 1, -1)
        t = self.ent_emb(t_idx).view(b_size, 1, -1)
        r = self.rel_emb(r_idx).view(b_size, 1, -1)
        concat = cat((h, r, t), dim=1)

        return self.output(self.convlayer(concat).reshape(b_size, -1))[:, 1]

    def normalize_parameters(self):
        """Normalize the entity embeddings, as explained in original paper.
        This methods should be called at the end of each training epoch and at
        the end of training as well.

        """
        pass

    def get_embeddings(self):
        """Return the embeddings of entities and relations.

        Returns
        -------
        ent_emb: torch.Tensor, shape: (n_ent, emb_dim), dtype: torch.float
            Embeddings of entities.
        rel_emb: torch.Tensor, shape: (n_rel, emb_dim), dtype: torch.float
            Embeddings of relations.

        """
        self.normalize_parameters()
        return self.ent_emb.weight.data, self.rel_emb.weight.data

    def inference_scoring_function(self, h, t, r):
        """Link prediction evaluation helper function. See
        torchkge.models.interfaces.Models for more details on the API.

        """
        b_size = h.shape[0]

        if (len(h.shape) == 2) & (len(t.shape) == 4) & (len(r.shape) == 2):
            concat = cat((h.view(b_size, 1, 1, self.emb_dim).expand(b_size, self.n_ent, 1, self.emb_dim),
                          r.view(b_size, 1, 1, self.emb_dim).expand(b_size, self.n_ent, 1, self.emb_dim),
                          t), dim=2)
            concat = concat.reshape(-1, 3, self.emb_dim)

        elif (len(h.shape) == 4) & (len(t.shape) == 2) & (len(r.shape) == 2):
            concat = cat((h,
                          r.view(b_size, 1, 1, self.emb_dim).expand(b_size, self.n_ent, 1, self.emb_dim),
                          t.view(b_size, 1, 1, self.emb_dim).expand(b_size, self.n_ent, 1, self.emb_dim)), dim=2)
            concat = concat.reshape(-1, 3, self.emb_dim)

        else:
            assert (len(h.shape) == 2) & (len(t.shape) == 2) & (len(r.shape) == 4)
            concat = cat((h.view(b_size, 1, 1, self.emb_dim).expand(b_size, self.n_rel, 1, self.emb_dim),
                          r,
                          t.view(b_size, 1, 1, self.emb_dim).expand(b_size, self.n_rel, 1, self.emb_dim)), dim=2)
            concat = concat.reshape(-1, 3, self.emb_dim)

        scores = self.output(self.convlayer(concat).reshape(concat.shape[0], -1))
        scores = scores.reshape(b_size, -1, 2)

        return scores[:, :, 1]

    def inference_prepare_candidates(self, h_idx, t_idx, r_idx, entities=True):
        """Link prediction evaluation helper function. Get entities embeddings
        and relations embeddings. The output will be fed to the
        `inference_scoring_function` method. See torchkge.models.interfaces.Models for
        more details on the API.

        """
        b_size = h_idx.shape[0]

        h = self.ent_emb(h_idx)
        t = self.ent_emb(t_idx)
        r = self.rel_emb(r_idx)

        if entities:
            candidates = self.ent_emb.weight.data.view(1, self.n_ent, self.emb_dim)
            candidates = candidates.expand(b_size, self.n_ent, self.emb_dim)
            candidates = candidates.view(b_size, self.n_ent, 1, self.emb_dim)
        else:
            candidates = self.rel_emb.weight.data.view(1, self.n_rel, self.emb_dim)
            candidates = candidates.expand(b_size, self.n_rel, self.emb_dim)
            candidates = candidates.view(b_size, self.n_rel, 1, self.emb_dim)

        return h, t, r, candidates
