# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
armand.boschin@telecom-paristech.fr
"""

from torch import Tensor, cat
from torch.utils.data import DataLoader
from torchkge.exceptions import NotYetEvaluated
from torchkge.utils import compute_weight, get_rank


class LinkPredictionEvaluator(object):
    def __init__(self, ent_emb, rel_emb, dissimilarity, knowledge_graph):
        self.ent_embed = ent_emb
        self.rel_embed = rel_emb
        self.dissimilarity = dissimilarity
        self.kg = knowledge_graph

        self.top_head_candidates = Tensor().long()
        self.top_tail_candidates = Tensor().long()
        self.rank_true_heads = Tensor().long()
        self.rank_true_tails = Tensor().long()

        self.evaluated = False
        self.use_cuda = False
        self.k_max = 10

    def cuda(self):
        """
        Move current object to CUDA
        """
        self.use_cuda = True
        self.kg.cuda()
        self.rel_embed.cuda()
        self.ent_embed.cuda()
        self.top_head_candidates = self.top_head_candidates.cuda()
        self.top_tail_candidates = self.top_tail_candidates.cuda()
        self.rank_true_heads = self.rank_true_heads.cuda()
        self.rank_true_tails = self.rank_true_tails.cuda()

    def evaluate_pair(self, entities, relations, true, heads=1):
        """
        :param entities: float tensor of shape (batch_size, ent_emb_dim) containing current
        embeddings of entities
        :param relations: float tensor of shape (batch_size, rel_emb_dim) containing current
        embeddings of relations
        :param true: int tensor of shape (batch_size)
        :param heads: 1 ou -1 (must be 1 if entities are heads and -1 if entities are tails).
        We test dissimilarity between heads * entities + relations and heads * targets.
        :return: rank_true_entities : int tensor of shape (batch_size) containing the rank of the
        true entities when ranking any entities based on computation of d(hear+relation, tail).
        sorted_candidates : int tensor of shape (batch_size, self.k_max) containing the k_max best
        entities ranked by decreasing dissimilarity d(hear+relation, tail).
        """
        current_batch_size, embedding_dimension = entities.shape

        # tmp_sum is either heads + relations or relations - tails
        tmp_sum = (heads * entities + relations).view((current_batch_size, embedding_dimension, 1))
        tmp_sum = tmp_sum.expand((current_batch_size, embedding_dimension, self.kg.n_ent))

        # compute either dissimilarity(heads + relation, candidates) or
        # dissimilarity(-candidates, relation - tails)
        candidates = self.ent_embed.weight.transpose(0, 1)
        dissimilarities = self.dissimilarity(tmp_sum, heads * candidates)

        # sort the candidates and return the rank of the true value along
        # with the top k_max candidates
        sorted_candidates = dissimilarities.argsort(dim=1, descending=False)
        rank_true_entities = get_rank(sorted_candidates, true)

        return rank_true_entities, sorted_candidates[:, :self.k_max]

    def evaluate(self, batch_size, k_max):
        """
        head_dissimilarities (resp. tail_dissimilarities) is the list of the top k_max most
        similar tails (resp. heads) for each (head, rel) pair (resp. (rel, tail) pair).

        Example : self.head_dissimilarities[i] is a list of entities in increasing order of
        dissimilarity such that d(head + rel, self.head_dissimilarities[i]) is the smallest
        possible values for any entity playing the role of tails.

        :param batch_size:
        :param k_max:
        :return:
        """
        dataloader = DataLoader(self.kg, batch_size=batch_size)
        self.k_max = k_max

        for i, batch in enumerate(dataloader):
            h_idx, t_idx, r_idx = batch[0], batch[1], batch[2]

            if self.use_cuda:
                h_idx, t_idx, r_idx = h_idx.cuda(), t_idx.cuda(), r_idx.cuda()

            heads = self.ent_embed.weight[h_idx]
            tails = self.ent_embed.weight[t_idx]
            relations = self.rel_embed.weight[r_idx]

            # evaluate both ways (head, rel) -> tail and (rel, tail) -> head
            rank_true_tails, top_tail_candidates = self.evaluate_pair(heads, relations, t_idx,
                                                                      heads=1)
            rank_true_heads, top_head_candidates = self.evaluate_pair(tails, relations, h_idx,
                                                                      heads=-1)

            self.top_tail_candidates = cat((self.top_tail_candidates, top_tail_candidates), dim=0)
            self.top_head_candidates = cat((self.top_head_candidates, top_head_candidates), dim=0)
            self.rank_true_tails = cat((self.rank_true_tails, rank_true_tails))
            self.rank_true_heads = cat((self.rank_true_heads, rank_true_heads))

        self.evaluated = True

    def mean_rank(self):
        """
        :return: the mean rank of the true entity when replacing alternatively head and tail in
        any fact of the dataset.
        """
        if not self.evaluated:
            raise NotYetEvaluated('Evaluator not evaluated call LinkPredictionEvaluator.evaluate')
        return (self.rank_true_heads.float().mean() + self.rank_true_tails.float().mean()).item() / 2

    def hit_at_k(self, k=10):
        """
        :param k: Hit@k is the number of entities that show up in the top k that give facts present
        in the dataset.
        :return: Average of hit@k for head and tail replacement. Computation is done in a vectorized
        way.
        """
        if not self.evaluated:
            raise NotYetEvaluated('Evaluator not evaluated call LinkPredictionEvaluator.evaluate')

        # drop duplicated (head, rel) and (rel, tail) pairs
        distinct_head_rel, mask_head_rel = cat((self.kg.head_idx.view((-1, 1)),
                                                self.kg.relations.view((-1, 1))), dim=1).unique(
            dim=0, return_inverse=True)
        distinct_rel_tail, mask_rel_tail = cat((self.kg.tail_idx.view((-1, 1)),
                                                self.kg.relations.view((-1, 1))), dim=1).unique(
            dim=0, return_inverse=True)

        # head_ (resp. tail_) refers to computation with (head, rel) pairs (resp. (rel, tail) pairs)
        shape1 = self.top_tail_candidates[:, :k].shape
        shape2 = self.top_head_candidates[:, :k].shape

        # ent_coeff is the number of true entities in the top k candidates when keeping entity
        head_coeff = (self.top_tail_candidates[:, :k] == self.kg.tail_idx.view(-1, 1).expand(
            shape1)).sum(dim=1)
        tail_coeff = (self.top_head_candidates[:, :k] == self.kg.head_idx.view(-1, 1).expand(
            shape2)).sum(dim=1)

        # weights are inverse of the number of times a pair (head, rel) or (rel, tail)
        # appears in the data set or k if greater than k.
        head_weight = compute_weight(mask_head_rel, k)
        tail_weight = compute_weight(mask_rel_tail, k)

        head_score = (head_coeff.float() * head_weight).sum() / len(distinct_head_rel)
        tail_score = (tail_coeff.float() * tail_weight).sum() / len(distinct_rel_tail)

        return (head_score + tail_score).item() / 2
