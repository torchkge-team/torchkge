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
    """Evaluate performance of given embedding using link prediction method. TODO : add reference.

        Parameters
        ----------
        ent_emb : torch tensor, dtype = Float, shape = (n_entities, ent_emb_dim)
            Embeddings of the entities.
        rel_emb : torch tensor, dtype = Float, shape = (n_relations, ent_emb_dim)
            Embeddings of the relations.
        dissimilarity : function
            Function used to compute the dissimilarity between head + relation and tail.
        knowledge_graph : torchkge.data.KnowledgeGraph.KnowledgeGraph
            Knowledge graph in the form of an object implemented in torchkge.data.KnowledgeGraph.KnowledgeGraph

        Attributes
        ----------
        ent_embed : torch tensor, dtype = Float, shape = (n_entities, ent_emb_dim)
            Embeddings of the entities.
        rel_embed : torch tensor, dtype = Float, shape = (n_relations, ent_emb_dim)
            Embeddings of the relations.
        dissimilarity : function
            Function used to compute the dissimilarity between head + relation and tail.
        kg : torchkge.data.KnowledgeGraph.KnowledgeGraph
            Knowledge graph in the form of an object implemented in torchkge.data.KnowledgeGraph.KnowledgeGraph
        top_head_candidates : torch tensor, dtype = long, shape = (batch_size, k_max)
            List of the top k_max most similar tails for each (head, rel) pair.
        top_tail_candidates : torch tensor, dtype = long, shape = (batch_size, k_max)
            List of the top k_max most similar heads for each (rel, tail) pair.
        rank_true_heads : torch tensor, dtype = TODO, shape = TODO
            TODO
        rank_true_tails : torch tensor, dtype = TODO, shape = TODO
            TODO
        evaluated : bool
            Indicates if the method LinkPredictionEvaluator.evaluate() has been called on\
            current object
        use_cuda : bool
            Indicates if the current LinkPredictionEvaluator instance has been moved to cuda.
        k_max : bool, default = 10
            Max value to be used to compute the hit@k score.
    """
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
        """Move current evaluator object to CUDA
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

        Parameters
        ----------
        entities : float tensor
            Tensor of shape (batch_size, ent_emb_dim) containing current embeddings of entities
        relations : float tensor
            Tensor of shape (batch_size, rel_emb_dim) containing current embeddings of relations
        true : integer tensor
            Tensor of shape (batch_size) containing the true entity for each sample.
        heads : integer
            1 ou -1 (must be 1 if entities are heads and -1 if entities are tails). \
            We test dissimilarity between heads * entities + relations and heads * targets.


        Returns
        -------
        rank_true_entities : integer tensor
            Tensor of shape (batch_size) containing the rank of the true entities when ranking any\
            entities based on computation of d(hear+relation, tail).
        sorted_candidates : integer tensor
            Tensor of shape (batch_size, self.k_max) containing the k_max best entities ranked by\
            decreasing dissimilarity d(hear+relation, tail).

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

        Parameters
        ----------
        batch_size : integer
            Size of the current batch.
        k_max : integer
            Maximal k value we plan to use for Hit@k. This is used to truncate tensor so that it \
            fits in memory.
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

        Returns
        -------
        mean_rank : float
            The mean rank of the true entity when replacing alternatively head and tail in\
        any fact of the dataset.

        """
        if not self.evaluated:
            raise NotYetEvaluated('Evaluator not evaluated call LinkPredictionEvaluator.evaluate')
        return (self.rank_true_heads.float().mean() + self.rank_true_tails.float().mean()).item() / 2

    def hit_at_k(self, k=10):
        """

        Parameters
        ----------
        k : integer
            Hit@k is the number of entities that show up in the top k that give facts present\
            in the dataset.

        Returns
        -------
        avg_hitatk : float
            Average of hit@k for head and tail replacement. Computation is done in a \
            vectorized way.
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
