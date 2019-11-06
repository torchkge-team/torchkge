# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
aboschin@enst.fr
"""

from torch import arange
from torch.nn import Module

from torchkge.utils import init_embedding, l1_dissimilarity, l2_dissimilarity
from torchkge.utils import get_rank, get_true_targets


class Model(Module):
    def __init__(self, ent_emb_dim, n_entities, n_relations):
        super().__init__()
        self.ent_emb_dim = ent_emb_dim
        self.number_entities = n_entities
        self.number_relations = n_relations

    def forward(self, heads, tails, negative_heads, negative_tails, relations):
        """Forward pass on the current batch.

        Parameters
        ----------
        heads: torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's heads
        tails: torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's tails.
        negative_heads: torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's negatively sampled heads.
        negative_tails: torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's negatively sampled tails.
        relations: torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's relations.

        Returns
        -------
        golden_triplets: torch tensor, dtype = float, shape = (batch_size)
            Scoring function evaluated on true triples.
        negative_triplets: torch tensor, dtype = float, shape = (batch_size)
            Scoring function evaluated on negatively sampled triples.

        """
        return self.scoring_function(heads, tails, relations), \
            self.scoring_function(negative_heads, negative_tails, relations)

    def scoring_function(self, heads_idx, tails_idx, rels_idx):
        pass

    def normalize_parameters(self):
        pass

    def evaluation_helper(self, h_idx, t_idx, r_idx):
        raise NotImplementedError

    def evaluate_candidates(self, h_idx, t_idx, r_idx, kg):
        proj_h_emb, proj_t_emb, candidates, r_emb = self.evaluation_helper(h_idx, t_idx, r_idx)

        # evaluation_helper both ways (head, rel) -> tail and (rel, tail) -> head
        rank_true_tails, filt_rank_true_tails = self.compute_ranks(proj_h_emb,
                                                                   candidates,
                                                                   r_emb, h_idx, r_idx,
                                                                   t_idx,
                                                                   kg.dict_of_tails,
                                                                   heads=1)
        rank_true_heads, filt_rank_true_heads = self.compute_ranks(proj_t_emb,
                                                                   candidates,
                                                                   r_emb, t_idx, r_idx,
                                                                   h_idx,
                                                                   kg.dict_of_heads,
                                                                   heads=-1)

        return rank_true_tails, filt_rank_true_tails, rank_true_heads, filt_rank_true_heads


class TranslationalModel(Model):
    def __init__(self, ent_emb_dim, n_entities, n_relations, dissimilarity):
        super().__init__(ent_emb_dim, n_entities, n_relations)

        self.entity_embeddings = init_embedding(self.number_entities, self.ent_emb_dim)

        assert dissimilarity in ['L1', 'L2', None]
        if dissimilarity == 'L1':
            self.dissimilarity = l1_dissimilarity
        elif dissimilarity == 'L2':
            self.dissimilarity = l2_dissimilarity
        else:
            self.dissimilarity = None

    def recover_project_normalize(self, ent_idx, rel_idx, normalize_):
        pass

    def compute_ranks(self, proj_e_emb, proj_candidates,
                      r_emb, e_idx, r_idx, true_idx, dictionary, heads=1):
        """

        Parameters
        ----------
        proj_e_emb: torch.Tensor, shape = (batch_size, rel_emb_dim), dtype = float
            Tensor containing current projected embeddings of entities.
        proj_candidates: torch.Tensor, shape = (b_size, rel_emb_dim, n_entities), dtype = float
            Tensor containing projected embeddings of all entities.
        r_emb: torch.Tensor, shape = (batch_size, ent_emb_dim), dtype = float
            Tensor containing current embeddings of relations.
        e_idx: torch.Tensor, shape = (batch_size), dtype = long
            Tensor containing the indices of entities.
        r_idx: torch.Tensor, shape = (batch_size), dtype = long
            Tensor containing the indices of relations.
        true_idx: torch.Tensor, shape = (batch_size), dtype = long
            Tensor containing the true entity for each sample.
        dictionary: default dict
            Dictionary of keys (int, int) and values list of ints giving all possible entities for
            the (entity, relation) pair.
        heads: integer
            1 ou -1 (must be 1 if entities are heads and -1 if entities are tails). \
            We test dissimilarity between heads * entities + relations and heads * targets.


        Returns
        -------
        rank_true_entities: torch.Tensor, shape = (b_size), dtype = int
            Tensor containing the rank of the true entities when ranking any entity based on \
            computation of d(hear+relation, tail).
        filtered_rank_true_entities: torch.Tensor, shape = (b_size), dtype = int
            Tensor containing the rank of the true entities when ranking only true false entities \
            based on computation of d(hear+relation, tail).

        """
        current_batch_size, embedding_dimension = proj_e_emb.shape

        # tmp_sum is either heads + r_emb or r_emb - tails (expand does not use extra memory)
        tmp_sum = (heads * proj_e_emb + r_emb).view((current_batch_size, embedding_dimension, 1))
        tmp_sum = tmp_sum.expand((current_batch_size, embedding_dimension, self.number_entities))

        # compute either dissimilarity(heads + relation, proj_candidates) or
        # dissimilarity(-proj_candidates, relation - tails)
        dissimilarities = self.dissimilarity(tmp_sum, heads * proj_candidates)

        # filter out the true negative samples by assigning infinite dissimilarity
        filt_dissimilarities = dissimilarities.clone()
        for i in range(current_batch_size):
            true_targets = get_true_targets(dictionary, e_idx, r_idx, true_idx, i)
            if true_targets is None:
                continue
            filt_dissimilarities[i][true_targets] = float('Inf')

        # from dissimilarities, extract the rank of the true entity.
        rank_true_entities = get_rank(-dissimilarities, true_idx)
        filtered_rank_true_entities = get_rank(-filt_dissimilarities, true_idx)

        return rank_true_entities, filtered_rank_true_entities

    def evaluation_helper(self, h_idx, t_idx, r_idx):
        return None, None, None, None

    def recover_candidates(self, h_idx, b_size):
        all_idx = arange(0, self.number_entities).long()
        if h_idx.is_cuda:
            all_idx = all_idx.cuda()
        candidates = self.entity_embeddings(all_idx).transpose(0, 1)
        candidates = candidates.view((1,
                                      self.ent_emb_dim,
                                      self.number_entities)).expand((b_size,
                                                                     self.ent_emb_dim,
                                                                     self.number_entities))
        return candidates

    def projection_helper(self, h_idx, t_idx, b_size, candidates, rel_emb_dim):
        mask = h_idx.view(b_size, 1, 1).expand(b_size, rel_emb_dim, 1)
        proj_h_emb = candidates.gather(dim=2, index=mask).view(b_size, rel_emb_dim)

        mask = t_idx.view(b_size, 1, 1).expand(b_size, rel_emb_dim, 1)
        proj_t_emb = candidates.gather(dim=2, index=mask).view(b_size, rel_emb_dim)

        return proj_h_emb, proj_t_emb
