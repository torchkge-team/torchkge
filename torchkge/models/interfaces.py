# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""

from torch.nn import Module

from ..utils import get_rank, get_true_targets
from ..utils.dissimilarities import l1_dissimilarity, l2_dissimilarity, \
    l1_torus_dissimilarity, l2_torus_dissimilarity, el2_torus_dissimilarity


class Model(Module):
    """Model interface to be used by any other class implementing a knowledge
    graph embedding model. It is only
    required to implement the methods `scoring_function`,
    `normalize_parameters`, `lp_prep_cands` and `lp_scoring_function`.

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
        """

        Parameters
        ----------
        heads: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's heads
        tails: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's tails.
        negative_heads: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's negatively sampled heads.
        negative_tails: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's negatively sampled tails.
        relations: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Integer keys of the current batch's relations.

        Returns
        -------
        positive_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
            Scoring function evaluated on true triples.
        negative_triplets: torch.Tensor, dtype: torch.float, shape: (b_size)
            Scoring function evaluated on negatively sampled triples.

        """
        pos = self.scoring_function(heads, tails, relations)
        if negative_heads.shape[0] > relations.shape[0]:
            # in that case, several negative samples are sampled from each fact
            n_neg = int(negative_heads.shape[0] / relations.shape[0])
            pos = pos.repeat(n_neg)
            neg = self.scoring_function(negative_heads,
                                        negative_tails,
                                        relations.repeat(n_neg))
        else:
            neg = self.scoring_function(negative_heads,
                                        negative_tails,
                                        relations)

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

    def lp_scoring_function(self, h, t, r):
        """ Link prediction evaluation helper function. Compute the scores of
        (h, r, c) or (c, r, t) for any candidate c. The arguments should
        match the ones of `lp_prep_cands`.

        Parameters
        ----------
        h: torch.Tensor, shape: (b_size, ent_emb_dim) or (b_size, n_ent,
            ent_emb_dim), dtype: torch.float
        t: torch.Tensor, shape: (b_size, ent_emb_dim) or (b_size, n_ent,
            ent_emb_dim), dtype: torch.float
        r: torch.Tensor, shape: (b_size, ent_emb_dim), dtype: torch.float

        Returns
        -------
        scores: torch.Tensor, shape: (b_size, n_ent), dtype: torch.float
            Scores of each candidate for each triple.
        """
        raise NotImplementedError

    def lp_prep_cands(self, h_idx, t_idx, r_idx):
        """Link prediction evaluation helper function. Get entities and
        relations embeddings, along with entity candidates ready for (projected
        if needed). The output will be fed to the `lp_scoring_function`
        method of the model at hand.

        Parameters
        ----------
        h_idx: torch.Tensor, shape: (b_size), dtype: torch.long
            List of heads indices.
        t_idx: torch.Tensor, shape: (b_size), dtype: torch.long
            List of tails indices.
        r_idx: torch.Tensor, shape: (b_size), dtype: torch.long
            List of relations indices.

        Returns
        -------
        h: torch.Tensor, shape: (b_size, rel_emb_dim), dtype: torch.float
            Head vectors fed to `lp_scoring_function`. For translation
            models it is the entities embeddings projected in relation space,
            for example.
        t: torch.Tensor, shape: (b_size, rel_emb_dim), dtype: torch.float
            Tail vectors fed to `lp_scoring_function`. For translation
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

    def lp_compute_ranks(self, e_emb, candidates, r, e_idx, r_idx, true_idx,
                         dictionary, heads=1):
        """Link prediction evaluation helper function. Compute the ranks and
        the filtered ranks of true entities when doing link prediction. Note
        that the best rank possible is 1.

        Parameters
        ----------
        e_emb: torch.Tensor, shape: (b_size, rel_emb_dim), dtype: torch.float
            Embeddings of current entities ready for
            `lp_scoring_function`.
        candidates: torch.Tensor, shape: (b_size, n_ent, emb_dim), dtype:
            torch.float
            Embeddings of all entities ready for
            `lp_scoring_function`.
        r: torch.Tensor, shape: (b_size, emb_dim, emb_dim) or (b_size,
            emb_dim), dtype: torch.float
            Embeddings or matrices of current relations ready for
            `lp_scoring_function`.
        e_idx: torch.Tensor, shape: (b_size), dtype: torch.long
            List of entities indices.
        r_idx: torch.Tensor, shape: (b_size), dtype: torch.long
            List of relations indices.
        true_idx: torch.Tensor, shape: (b_size), dtype: torch.long
            List of the indices of the true entity for each sample.
        dictionary: defaultdict
            Dictionary of keys (int, int) and values list of ints giving all
            possible entities for the (entity, relation) pair.
        heads: integer
            1 ou -1 (must be 1 if entities are heads and -1 if entities are
            tails). The computed score is either :math:`f_r(e, candidate)` (if
            `heads` is 1) or :math:`f_r(candidate, e)` (if `heads` is -1).


        Returns
        -------
        rank_true_entities: torch.Tensor, shape: (b_size), dtype: torch.int
            List of the ranks of true entities when all candidates are sorted
            by decreasing order of scoring function.
        filt_rank_true_entities: torch.Tensor, shape: (b_size), dtype:
            torch.int
            List of the ranks of true entities when only candidates which are
            not known to lead to a true fact are sorted by decreasing order
            of scoring function.

        """
        b_size = r_idx.shape[0]

        if heads == 1:
            scores = self.lp_scoring_function(e_emb, candidates, r)
        else:
            scores = self.lp_scoring_function(candidates, e_emb, r)

        # filter out the true negative samples by assigning - inf score.
        filt_scores = scores.clone()
        for i in range(b_size):
            true_targets = get_true_targets(dictionary, e_idx, r_idx,
                                            true_idx, i)
            if true_targets is None:
                continue
            filt_scores[i][true_targets] = - float('Inf')

        # from dissimilarities, extract the rank of the true entity.
        rank_true_entities = get_rank(scores, true_idx)
        filtered_rank_true_entities = get_rank(filt_scores, true_idx)

        return rank_true_entities, filtered_rank_true_entities

    def lp_helper(self, h_idx, t_idx, r_idx, kg):
        """Link prediction evaluation helper function. Compute the head and
        tail ranks and filtered ranks of the current batch.

        Parameters
        ----------
        h_idx: torch.Tensor, shape: (b_size), dtype: torch.long
            List of heads indices.
        t_idx: torch.Tensor, shape: (b_size), dtype: torch.long
            List of tails indices.
        r_idx: torch.Tensor, shape: (b_size), dtype: torch.long
            List of relations indices.
        kg: torchkge.data_structures.KnowledgeGraph
            Knowledge graph on which the model was trained. This is used to
            access the `dict_of_heads` and `dict_of_tails` attributes in order
            to compute the filtered metrics.

        Returns
        -------
        rank_true_tails: torch.Tensor, shape: (b_size), dtype: torch.int
            List of the ranks of true tails when all candidates are sorted
            by decreasing order of scoring function.
        filt_rank_true_tails: torch.Tensor, shape: (b_size), dtype: torch.int
            List of the filtered ranks of true tails when candidates are
            sorted by decreasing order of scoring function.
        rank_true_heads: torch.Tensor, shape: (b_size), dtype: torch.int
            List of the ranks of true heads when all candidates are sorted
            by decreasing order of scoring function.
        filt_rank_true_heads: torch.Tensor, shape: (b_size), dtype: torch.int
            List of the filtered ranks of true heads when candidates are
            sorted by decreasing order of scoring function.

        """
        h_emb, t_emb, candidates, r = self.lp_prep_cands(h_idx, t_idx, r_idx)

        rank_true_tails, filt_rank_true_tails = self.lp_compute_ranks(
            h_emb, candidates, r, h_idx, r_idx, t_idx, kg.dict_of_tails,
            heads=1)
        rank_true_heads, filt_rank_true_heads = self.lp_compute_ranks(
            t_emb, candidates, r, t_idx, r_idx, h_idx, kg.dict_of_heads,
            heads=-1)

        return (rank_true_tails, filt_rank_true_tails,
                rank_true_heads, filt_rank_true_heads)


class TranslationModel(Model):
    """Model interface to be used by any other class implementing a
    translation knowledge graph embedding model. This interface inherits from
    the interface :class:`torchkge.models.interfaces.Model`. It is only
    required to implement the methods `scoring_function`,
    `normalize_parameters` and `lp_prep_cands`.

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

    def lp_prep_cands(self, h_idx, t_idx, r_idx):
        """See torchkge.models.interfaces.Models.

        """
        raise NotImplementedError

    def lp_scoring_function(self, proj_h, proj_t, r):
        """This overwrites the method declared in
        torchkge.models.interfaces.Models. For translation models, the computed
        score is the dissimilarity of between projected heads + relations and
        projected tails. Projections are done in relation-specific subspaces.

        """
        b_size = proj_h.shape[0]

        if len(proj_h.shape) == 2 and len(proj_t.shape) == 3:
            # this is the tail completion case in link prediction
            hr = (proj_h + r).view(b_size, 1, r.shape[1])
            return - self.dissimilarity(hr, proj_t)
        else:
            # this is the head completion case in link prediction
            r_ = r.view(b_size, 1, r.shape[1])
            t_ = proj_t.view(b_size, 1, r.shape[1])
            return - self.dissimilarity(proj_h + r_, t_)


class BilinearModel(Model):
    """Model interface to be used by any other class implementing a
    bilinear knowledge graph embedding model. This interface inherits from
    the interface :class:`torchkge.models.interfaces.Model`. It is only
    required to implement the methods `scoring_function`,
    `normalize_parameters`, `lp_prep_cands` and `lp_scoring_function`.

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

    def lp_scoring_function(self, h, t, r):
        """See torchkge.models.interfaces.Models.

        """
        raise NotImplementedError

    def lp_prep_cands(self, h_idx, t_idx, r_idx):
        """See torchkge.models.interfaces.Models.

        """
        raise NotImplementedError
