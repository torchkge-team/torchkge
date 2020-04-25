# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""

from torch import arange, matmul
from torch.nn import Module

from torchkge.utils import init_embedding, l1_dissimilarity, l2_dissimilarity
from torchkge.utils import get_rank, get_true_targets


class Model(Module):
    """Model interface to be used by any other class implementing a knowledge graph embedding model.

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

    def forward(self, heads, tails, negative_heads, negative_tails, relations):
        """Forward pass on the current batch.

        Parameters
        ----------
        heads: `torch.Tensor`, dtype: `torch.long`, shape: (batch_size)
            Integer keys of the current batch's heads
        tails: `torch.Tensor`, dtype: `torch.long`, shape: (batch_size)
            Integer keys of the current batch's tails.
        negative_heads: `torch.Tensor`, dtype: `torch.long`, shape: (batch_size)
            Integer keys of the current batch's negatively sampled heads.
        negative_tails: `torch.Tensor`, dtype: `torch.long`, shape: (batch_size)
            Integer keys of the current batch's negatively sampled tails.
        relations: `torch.Tensor`, dtype: `torch.long`, shape: (batch_size)
            Integer keys of the current batch's relations.

        Returns
        -------
        positive_triplets: `torch.Tensor`, dtype: `torch.float`, shape: (batch_size)
            Scoring function evaluated on true triples.
        negative_triplets: `torch.Tensor`, dtype: `torch.float`, shape: (batch_size)
            Scoring function evaluated on negatively sampled triples.

        """
        return self.scoring_function(heads, tails, relations), \
            self.scoring_function(negative_heads, negative_tails, relations)

    def scoring_function(self, h_idx, t_idx, r_idx):
        """Compute the scoring function for the triplets given as argument.

        Parameters
        ----------
        h_idx: `torch.Tensor`, dtype: `torch.long`, shape: (batch_size)
            Integer keys of the current batch's heads
        t_idx: `torch.Tensor`, dtype: `torch.long`, shape: (batch_size)
            Integer keys of the current batch's tails.
        r_idx: `torch.Tensor`, dtype: `torch.long`, shape: (batch_size)
            Integer keys of the current batch's relations.

        Returns
        -------
        score: `torch.Tensor`, dtype: `torch.float`, shape: (batch_size)
            Score function: opposite of dissimilarities between h+r and t.

        """
        raise NotImplementedError

    def normalize_parameters(self):
        raise NotImplementedError

    def lp_get_emb_cand(self, h_idx, t_idx, r_idx):
        """Get entities and relations, along with candidates, ready for rank computation (projected if needed).

        Parameters
        ----------
        h_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Tensor containing indices of current head entities.
        t_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Tensor containing indices of current tail entities.
        r_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Tensor containing indices of current relations.

        Returns
        -------
        proj_h_emb: `torch.Tensor`, shape: (b_size, rel_emb_dim), dtype: `torch.float`
            Tensor containing embeddings of current head entities projected in relation space.
        proj_t_emb: `torch.Tensor`, shape: (b_size, rel_emb_dim), dtype: `torch.float`
            Tensor containing embeddings of current tail entities projected in relation space.
        proj_candidates: `torch.Tensor`, shape: (b_size, rel_emb_dim, n_entities), dtype: `torch.float`
            Tensor containing all entities projected in each relation spaces (relations
            corresponding to current batch's relations).
        r_emb: `torch.Tensor`, shape: (b_size, rel_emb_dim), dtype: `torch.float`
            Tensor containing current relations embeddings.

        """
        raise NotImplementedError

    def lp_compute_ranks(self, e_emb, candidates, r, e_idx, r_idx, true_idx, dictionary, heads=1):
        """Compute the ranks and the filtered ranks of true entities when doing link prediction. Note that the \
        best rank possible is 1.

        Parameters
        ----------
        e_emb: `torch.Tensor`, shape: (batch_size, rel_emb_dim), dtype: `torch.float`
            Tensor containing current embeddings of entities.
        candidates: torch tensor, shape: (b_size, number_entities, emb_dim), dtype: `torch.float`
            Tensor containing projected embeddings of all entities.
        r: `torch.Tensor`, shape: (b_size, emb_dim, emb_dim) or (b_size, emb_dim) dtype: `torch.float`
            Tensor containing current matrices or embeddings of relations.
        e_idx: torch tensor, shape: (batch_size), dtype: `torch.long`
            Tensor containing the indices of entities.
        r_idx: torch tensor, shape: (batch_size), dtype: `torch.long`
            Tensor containing the indices of relations.
        true_idx: torch tensor, shape: (batch_size), dtype: `torch.long`
            Tensor containing the true entity for each sample.
        dictionary: default dict
            Dictionary of keys (int, int) and values list of ints giving all possible entities for\
            the (entity, relation) pair.
        heads: integer
            1 ou -1 (must be 1 if entities are heads and -1 if entities are tails). We test\
             dissimilarity_type between heads * entities + relations and heads * targets.


        Returns
        -------
        rank_true_entities: torch Tensor, shape: (b_size), dtype: `torch.int`
            Tensor containing the rank of the true entities when ranking any entity based on\
            estimation of 1 or 0.
        filtered_rank_true_entities: torch Tensor, shape: (b_size), dtype: `torch.int`
            Tensor containing the rank of the true entities when ranking only true false entities\
            based on estimation of 1 or 0.

        """
        b_size = r_idx.shape[0]

        if heads == 1:
            scores = self.lp_batch_scoring_function(e_emb, candidates, r)
        else:
            scores = self.lp_batch_scoring_function(candidates, e_emb, r)

        # filter out the true negative samples by assigning negative score
        filt_scores = scores.clone()
        for i in range(b_size):
            true_targets = get_true_targets(dictionary, e_idx, r_idx, true_idx, i)
            if true_targets is None:
                continue
            filt_scores[i][true_targets] = float(-1)

        # from dissimilarities, extract the rank of the true entity.
        rank_true_entities = get_rank(scores, true_idx)
        filtered_rank_true_entities = get_rank(filt_scores, true_idx)

        return rank_true_entities, filtered_rank_true_entities

    def lp_helper(self, h_idx, t_idx, r_idx, kg):
        """Compute the head and tail ranks and filtered ranks of the current batch.

        Parameters
        ----------
        h_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Tensor containing indices of current head entities.
        t_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Tensor containing indices of current tail entities.
        r_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Tensor containing indices of current relations.
        kg: `torchkge.data.KnowledgeGraph.KnowledgeGraph`
            Knowledge graph on which the model was trained.

        Returns
        -------
        rank_true_tails: `torch.Tensor`, shape: (b_size), dtype: `torch.int`
            Tensor containing the rank of the true tails when ranking any entity based on \
            computation of d(hear+relation, tail).
        filt_rank_true_tails: `torch.Tensor`, shape: (b_size), dtype: `torch.int`
            Tensor containing the rank of the true tails when ranking only true false entities \
            based on computation of d(hear+relation, tail).
        rank_true_heads: Tensor containing the rank of the true heads when ranking any entity based on \
            computation of d(hear+relation, tail).
        filt_rank_true_heads: `torch.Tensor`, shape: (b_size), dtype: `torch.int`
            Tensor containing the rank of the true heads when ranking only true false entities \
            based on computation of d(hear+relation, tail).

        """
        h_emb, t_emb, candidates, r = self.lp_get_emb_cand(h_idx, t_idx, r_idx)

        rank_true_tails, filt_rank_true_tails = self.lp_compute_ranks(h_emb, candidates, r, h_idx, r_idx, t_idx,
                                                                      kg.dict_of_tails, heads=1)
        rank_true_heads, filt_rank_true_heads = self.lp_compute_ranks(t_emb, candidates, r, t_idx, r_idx, h_idx,
                                                                      kg.dict_of_heads, heads=-1)

        return rank_true_tails, filt_rank_true_tails, rank_true_heads, filt_rank_true_heads


class TranslationModel(Model):
    """Model interface to be used by any other class implementing a translational knowledge graph embedding model.
    This interface inherits from the interface :class:`torchkge.models.interfaces.Model`.

    Parameters
    ----------
    ent_emb_dim: int
        Embedding dimension of the entities.
    n_entities: int
        Number of entities to be embedded.
    n_relations: int
        Number of relations to be embedded.
    dissimilarity_type: int
        Name of the dissimilarity function to be used.

    Attributes
    ----------
    dissimilarity: function
        Dissimilarity function defined in `torchkge.utils.dissimilarities`.

    """
    def __init__(self, n_entities, n_relations, dissimilarity_type):
        super().__init__(n_entities, n_relations)
        self.dissimilarity_type = dissimilarity_type

    def scoring_function(self, h_idx, t_idx, r_idx):
        raise NotImplementedError

    def normalize_parameters(self):
        raise NotImplementedError

    def lp_batch_scoring_function(self, proj_h, proj_t, r):
        b_size = proj_h.shape[0]

        if len(proj_h.shape) == 2 and len(proj_t.shape) == 3:
            # this is the tail completion case in link prediction
            return - ((proj_h + r).view(b_size, 1, r.shape[1]) - proj_t).norm(p=self.dissimilarity_type, dim=2)
        else:
            # this is the head completion case in link prediction
            return - (proj_h + (r - proj_t).view(b_size, 1, r.shape[1])).norm(p=self.dissimilarity_type, dim=2)

    def lp_get_emb_cand(self, h_idx, t_idx, r_idx):
        raise NotImplementedError


class BilinearModel(Model):

    def __init__(self, emb_dim, n_entities, n_relations):
        super().__init__(n_entities, n_relations)
        self.emb_dim = emb_dim

    def scoring_function(self, h_idx, t_idx, r_idx):
        raise NotImplementedError

    def normalize_parameters(self):
        raise NotImplementedError

    def lp_batch_scoring_function(self, h, t, r):
        raise NotImplementedError

    def lp_get_emb_cand(self, h_idx, t_idx, r_idx):
        raise NotImplementedError
