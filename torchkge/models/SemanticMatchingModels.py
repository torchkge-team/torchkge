# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
aboschin@enst.fr
"""

from torch import empty, matmul, tensor
from torch.nn import Module, Embedding, Parameter
from torch.nn.functional import normalize
from torch.nn.init import xavier_uniform_
from torchkge.utils import get_rank


class RESCALModel(Module):
    """
    This module should be implemented with ALS. For the time being it is implemented with SGD
    but the loss is still missing sum(true - predicted)**2.
    """

    def __init__(self, config):
        super().__init__()
        self.ent_emb_dim = config.entities_embedding_dimension
        self.number_entities = config.number_entities
        self.number_relations = config.number_relations

        # initialize embedding objects
        self.entity_embeddings = Embedding(self.number_entities, self.ent_emb_dim)
        self.relation_matrices = Parameter(xavier_uniform_(empty(size=(self.number_relations,
                                                                       self.ent_emb_dim,
                                                                       self.ent_emb_dim))))

        # fill the embedding weights with Xavier initialized values
        self.entity_embeddings.weight = Parameter(xavier_uniform_(
            empty(size=(self.number_entities, self.ent_emb_dim))))

        # normalize the embeddings
        self.entity_embeddings.weight.data = normalize(self.entity_embeddings.weight.data,
                                                       p=2, dim=1)

    def forward(self, heads, tails, negative_heads, negative_tails, relations):
        # recover entities embeddings
        heads_embeddings = self.entity_embeddings(heads)
        tails_embeddings = self.entity_embeddings(tails)
        neg_heads_embeddings = self.entity_embeddings(negative_heads)
        neg_tails_embeddings = self.entity_embeddings(negative_tails)

        # recover relation matrices
        relation_matrices = self.relation_matrices[relations]

        return self.compute_product(heads_embeddings, tails_embeddings, relation_matrices), \
            self.compute_product(neg_heads_embeddings, neg_tails_embeddings, relation_matrices)

    def compute_product(self, heads, tails, rel_mat):
        b_size = len(heads)
        if len(tails.shape) == 3:
            tails = tails.transpose(1, 2)
        else:
            tails = tails.view(b_size, self.ent_emb_dim, 1)

        if len(heads.shape) == 2:
            heads = heads.view(b_size, 1, self.ent_emb_dim)

        if (len(heads.shape) == 3) or (len(tails.shape) == 3):
            return matmul(matmul(heads, rel_mat), tails).view(b_size, -1)
        else:
            return matmul(matmul(heads, rel_mat), tails).view(b_size)

    def normalize_parameters(self):
        pass

    def evaluation_helper(self, h_idx, t_idx, r_idx):
        b_size = len(h_idx)

        candidates = self.entity_embeddings.weight.data
        candidates = candidates.view(1, self.number_entities, self.ent_emb_dim)
        candidates = candidates.expand(b_size, self.number_entities, self.ent_emb_dim)

        h_emb = self.entity_embeddings(h_idx)
        t_emb = self.entity_embeddings(t_idx)
        r_mat = self.relation_matrices[r_idx]

        return h_emb, t_emb, r_mat, candidates

    def compute_ranks(self, e_emb, candidates, r_mat, e_idx, r_idx, true_idx, dictionary, heads=1):
        current_batch_size, _ = e_emb.shape

        if heads == 1:
            scores = self.compute_product(e_emb, candidates, r_mat)
        else:
            scores = self.compute_product(candidates, e_emb, r_mat)

        # filter out the true negative samples by assigning negative score
        filt_scores = scores.clone()
        for i in range(current_batch_size):
            true_targets = dictionary[e_idx[i].item(), r_idx[i].item()].copy()
            if len(true_targets) == 1:
                continue
            true_targets.remove(true_idx[i].item())
            true_targets = tensor(true_targets).long()
            filt_scores[i][true_targets] = float(-1)

        # from dissimilarities, extract the rank of the true entity.
        rank_true_entities = get_rank(scores, true_idx, low_values=False)
        filtered_rank_true_entities = get_rank(filt_scores, true_idx, low_values=False)

        return rank_true_entities, filtered_rank_true_entities

    def evaluate_candidates(self, h_idx, t_idx, r_idx, kg):
        h_emb, t_emb, r_mat, candidates = self.evaluation_helper(h_idx, t_idx, r_idx)

        rank_true_tails, filt_rank_true_tails = self.compute_ranks(h_emb,
                                                                   candidates,
                                                                   r_mat, h_idx, r_idx,
                                                                   t_idx,
                                                                   kg.dict_of_tails,
                                                                   heads=1)
        rank_true_heads, filt_rank_true_heads = self.compute_ranks(t_emb,
                                                                   candidates,
                                                                   r_mat, t_idx, r_idx,
                                                                   h_idx,
                                                                   kg.dict_of_heads,
                                                                   heads=-1)

        return rank_true_tails, filt_rank_true_tails, rank_true_heads, filt_rank_true_heads


class DistMulModel(Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass

    def normalize_parameters(self):
        pass


class HolEModel(Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass

    def normalize_parameters(self):
        pass


class ComplexModel(Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass

    def normalize_parameters(self):
        pass


class AnalogyModel(Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass

    def normalize_parameters(self):
        pass
