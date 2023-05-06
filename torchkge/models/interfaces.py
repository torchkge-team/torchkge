# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""

from torch.nn import Module

from ..utils.dissimilarities import l1_dissimilarity, l2_dissimilarity, \
    l1_torus_dissimilarity, l2_torus_dissimilarity, el2_torus_dissimilarity


class Model(Module):
    """Model interface to be used by any other class implementing a knowledge
    graph embedding model. It is only
    required to implement the methods `scoring_function`,
    `normalize_parameters`, `inference_prepare_candidates` and `inference_scoring_function`.

    Parameters
    ----------
    n_entities: int
        Number of entities to be embedded.
    n_relations: int
        Number of relations to be embedded.

    Attributes
    ----------
    n_ent: int
        Number of entities to be embedded.
    n_rel: int
        Number of relations to be embedded.

    """
    def __init__(self, n_entities, n_relations):
        super().__init__()
        self.n_ent = n_entities
        self.n_rel = n_relations

    def forward(self, heads, tails, relations, negative_heads, negative_tails, negative_relations=None):
        """

        Parameters
        ----------
        heads: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's heads
        tails: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's tails.
        relations: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's relations.
        negative_heads: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's negatively sampled heads.
        negative_tails: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's negatively sampled tails.ze)
        negative_relations: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's negatively sampled relations.

        Returns
        -------
        positive_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
            Scoring function evaluated on true triples.
        negative_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
            Scoring function evaluated on negatively sampled triples.

        """
        pos = self.scoring_function(heads, tails, relations)

        if negative_relations is None:
            negative_relations = relations

        if negative_heads.shape[0] > negative_relations.shape[0]:
            # in that case, several negative samples are sampled from each fact
            n_neg = int(negative_heads.shape[0] / negative_relations.shape[0])
            pos = pos.repeat(n_neg)
            neg = self.scoring_function(negative_heads,
                                        negative_tails,
                                        negative_relations.repeat(n_neg))
        else:
            neg = self.scoring_function(negative_heads,
                                        negative_tails,
                                        negative_relations)

        return pos, neg

    def scoring_function(self, h_idx, t_idx, r_idx):
        """Compute the scoring function for the triplets given as argument.

        Parameters
        ----------
        h_idx: torch.Tensor, dtype: torch.long, shape: (b_size)
            Integer keys of the current batch's heads
        t_idx: torch.Tensor, dtype: torch.long, shape: (b_size)
            Integer keys of the current batch's tails.
        r_idx: torch.Tensor, dtype: torch.long, shape: (b_size)
            Integer keys of the current batch's relations.

        Returns
        -------
        score: torch.Tensor, dtype: torch.float, shape: (b_size)
            Score of each triplet.

        """
        raise NotImplementedError

    def normalize_parameters(self):
        """Normalize some parameters. This methods should be end at the end of
        each training epoch and at the end of training as well.

        """
        raise NotImplementedError

    def get_embeddings(self):
        """Return the tensors representing entities and relations in current
        model.

        """
        raise NotImplementedError

    def inference_scoring_function(self, h, t, r):
        """ Link prediction evaluation helper function. Compute the scores of
        (h, r, c) or (c, r, t) for any candidate c. The arguments should
        match the ones of `inference_prepare_candidates`.

        Parameters
        ----------
        h: torch.Tensor, shape: (b_size, ent_emb_dim) or (b_size, n_ent,
            ent_emb_dim), dtype: torch.float
        t: torch.Tensor, shape: (b_size, ent_emb_dim) or (b_size, n_ent,
            ent_emb_dim), dtype: torch.float
        r: torch.Tensor, shape: (b_size, ent_emb_dim) or (b_size, n_rel,
            ent_emb_dim), dtype: torch.float

        Returns
        -------
        scores: torch.Tensor, shape: (b_size, n_ent), dtype: torch.float
            Scores of each candidate for each triple.
        """
        raise NotImplementedError

    def inference_prepare_candidates(self, h_idx, t_idx, r_idx, entities=True):
        """Link prediction evaluation helper function. Get entities and
        relations embeddings, along with entity candidates ready for (projected
        if needed). The output will be fed to the `inference_scoring_function`
        method of the model at hand.

        Parameters
        ----------
        h_idx: torch.Tensor, shape: (b_size), dtype: torch.long
            List of heads indices.
        t_idx: torch.Tensor, shape: (b_size), dtype: torch.long
            List of tails indices.
        r_idx: torch.Tensor, shape: (b_size), dtype: torch.long
            List of relations indices.
        entities: bool
            Boolean indicating if candidates are entities or not.

        Returns
        -------
        h: torch.Tensor, shape: (b_size, rel_emb_dim), dtype: torch.float
            Head vectors fed to `inference_scoring_function`. For translation
            models it is the entities embeddings projected in relation space,
            for example.
        t: torch.Tensor, shape: (b_size, rel_emb_dim), dtype: torch.float
            Tail vectors fed to `inference_scoring_function`. For translation
            models it is the entities embeddings projected in relation space,
            for example.
        candidates: torch.Tensor, shape: (b_size, rel_emb_dim, n_ent),
            dtype: torch.float
            All entities embeddings prepared from batch evaluation. Axis 0 is
            simply duplication.
        r: torch.Tensor, shape: (b_size, rel_emb_dim), dtype: torch.float
            Relations embeddings or matrices.

        """
        raise NotImplementedError


class TranslationModel(Model):
    """Model interface to be used by any other class implementing a
    translation knowledge graph embedding model. This interface inherits from
    the interface :class:`torchkge.models.interfaces.Model`. It is only
    required to implement the methods `scoring_function`,
    `normalize_parameters` and `inference_prepare_candidates`.

    Parameters
    ----------
    n_entities: int
        Number of entities to be embedded.
    n_relations: int
        Number of relations to be embedded.
    dissimilarity_type: str
        One of 'L1', 'L2', 'toruse_L1', 'toruse_L2' and 'toruse_eL2'.

    Attributes
    ----------
    dissimilarity: function
        Dissimilarity function.

    """
    def __init__(self, n_entities, n_relations, dissimilarity_type):
        super().__init__(n_entities, n_relations)

        assert dissimilarity_type in ['L1', 'L2', 'torus_L1', 'torus_L2',
                                      'torus_eL2']

        if dissimilarity_type == 'L1':
            self.dissimilarity = l1_dissimilarity
        elif dissimilarity_type == 'L2':
            self.dissimilarity = l2_dissimilarity
        elif dissimilarity_type == 'torus_L1':
            self.dissimilarity = l1_torus_dissimilarity
        elif dissimilarity_type == 'torus_L2':
            self.dissimilarity = l2_torus_dissimilarity
        else:
            self.dissimilarity = el2_torus_dissimilarity

    def scoring_function(self, h_idx, t_idx, r_idx):
        """See torchkge.models.interfaces.Models.

        """
        raise NotImplementedError

    def normalize_parameters(self):
        """See torchkge.models.interfaces.Models.

        """
        raise NotImplementedError

    def get_embeddings(self):
        """See torchkge.models.interfaces.Models.

        """
        raise NotImplementedError

    def inference_prepare_candidates(self, h_idx, t_idx, r_idx, entities=True):
        """See torchkge.models.interfaces.Models.

        """
        raise NotImplementedError

    def inference_scoring_function(self, proj_h, proj_t, r):
        """This overwrites the method declared in
        torchkge.models.interfaces.Models. For translation models, the computed
        score is the dissimilarity of between projected heads + relations and
        projected tails. Projections are done in relation-specific subspaces.

        """
        b_size = proj_h.shape[0]

        if len(r.shape) == 2:
            if len(proj_t.shape) == 3:
                assert (len(proj_h.shape) == 2)
                # this is the tail completion case in link prediction
                hr = (proj_h + r).view(b_size, 1, r.shape[1])
                return - self.dissimilarity(hr, proj_t)
            else:
                assert (len(proj_h.shape) == 3) & (len(proj_t.shape) == 2)
                # this is the head completion case in link prediction
                r_ = r.view(b_size, 1, r.shape[1])
                t_ = proj_t.view(b_size, 1, r.shape[1])
                return - self.dissimilarity(proj_h + r_, t_)
        elif len(r.shape) == 3:
            # this is the relation prediction case
            # Two cases possible:
            # * proj_ent.shape == (b_size, self.n_rel, self.emb_dim) -> projection depending on relations
            # * proj_ent.shape == (b_size, self.emb_dim) -> no projection
            try:
                proj_h = proj_h.view(b_size, -1, self.emb_dim)
                proj_t = proj_t.view(b_size, -1, self.emb_dim)
            except AttributeError:
                proj_h = proj_h.view(b_size, -1, self.rel_emb_dim)
                proj_t = proj_t.view(b_size, -1, self.rel_emb_dim)
            return - self.dissimilarity(proj_h + r, proj_t)


class BilinearModel(Model):
    """Model interface to be used by any other class implementing a
    bilinear knowledge graph embedding model. This interface inherits from
    the interface :class:`torchkge.models.interfaces.Model`. It is only
    required to implement the methods `scoring_function`,
    `normalize_parameters`, `inference_prepare_candidates` and `inference_scoring_function`.

    Parameters
    ----------
    n_entities: int
        Number of entities to be embedded.
    n_relations: int
        Number of relations to be embedded.
    emb_dim: int
        Dimension of the embedding space.

    Attributes
    ----------
    emb_dim: int
        Dimension of the embedding space.

    """

    def __init__(self, emb_dim, n_entities, n_relations):
        super().__init__(n_entities, n_relations)
        self.emb_dim = emb_dim

    def scoring_function(self, h_idx, t_idx, r_idx):
        """See torchkge.models.interfaces.Models.

        """
        raise NotImplementedError

    def normalize_parameters(self):
        """See torchkge.models.interfaces.Models.

        """
        raise NotImplementedError

    def get_embeddings(self):
        """See torchkge.models.interfaces.Models.

        """
        raise NotImplementedError

    def inference_scoring_function(self, h, t, r):
        """See torchkge.models.interfaces.Models.

        """
        raise NotImplementedError

    def inference_prepare_candidates(self, h_idx, t_idx, r_idx, entities=True):
        """See torchkge.models.interfaces.Models.

        """
        raise NotImplementedError
