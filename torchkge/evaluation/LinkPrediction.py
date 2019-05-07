# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
aboschin@enst.fr
"""

from torch import Tensor, cat
from torch.utils.data import DataLoader
from torchkge.exceptions import NotYetEvaluated
from torchkge.utils import get_rank

from tqdm import tqdm


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
            Knowledge graph in the form of an object implemented in
            torchkge.data.KnowledgeGraph.KnowledgeGraph

        Attributes
        ----------
        ent_embed : torch tensor, dtype = Float, shape = (n_entities, ent_emb_dim)
            Embeddings of the entities.
        rel_embed : torch tensor, dtype = Float, shape = (n_relations, ent_emb_dim)
            Embeddings of the relations.
        dissimilarity : function
            Function used to compute the dissimilarity between head + relation and tail.
        kg : torchkge.data.KnowledgeGraph.KnowledgeGraph
            Knowledge graph in the form of an object implemented in
            torchkge.data.KnowledgeGraph.KnowledgeGraph
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

        self.filt_top_head_candidates = Tensor().long()
        self.filt_top_tail_candidates = Tensor().long()
        self.filt_rank_true_heads = Tensor().long()
        self.filt_rank_true_tails = Tensor().long()

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

        self.filt_top_head_candidates = self.filt_top_head_candidates.cuda()
        self.filt_top_tail_candidates = self.filt_top_tail_candidates.cuda()
        self.filt_rank_true_heads = self.filt_rank_true_heads.cuda()
        self.filt_rank_true_tails = self.filt_rank_true_tails.cuda()

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

        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

            h_idx, t_idx, r_idx = batch[0], batch[1], batch[2]
            list_of_heads, list_of_tails = batch[3], batch[4]

            if self.use_cuda:
                h_idx, t_idx, r_idx = h_idx.cuda(), t_idx.cuda(), r_idx.cuda()
                list_of_heads, list_of_tails = list_of_heads.cuda(), list_of_tails.cuda()

            heads = self.ent_embed.weight[h_idx]
            tails = self.ent_embed.weight[t_idx]
            relations = self.rel_embed.weight[r_idx]

            # evaluate both ways (head, rel) -> tail and (rel, tail) -> head
            rank_true_tails, filt_rank_true_tails  = self.evaluate_pair(heads, relations, t_idx,
                                                                        list_of_tails, heads=1)
            rank_true_heads, filt_rank_true_heads = self.evaluate_pair(tails, relations, h_idx,
                                                                       list_of_heads, heads=-1)

            # self.top_tail_candidates = cat((self.top_tail_candidates, top_tail_candidates), dim=0)
            # self.top_head_candidates = cat((self.top_head_candidates, top_head_candidates), dim=0)
            self.rank_true_tails = cat((self.rank_true_tails, rank_true_tails))
            self.rank_true_heads = cat((self.rank_true_heads, rank_true_heads))

            # self.filt_top_tail_candidates = cat((self.filt_top_tail_candidates,
            #                                     filt_top_tail_candidates), dim=0)
            # self.filt_top_head_candidates = cat((self.filt_top_head_candidates,
            #                                     filt_top_head_candidates), dim=0)
            self.filt_rank_true_tails = cat((self.filt_rank_true_tails, filt_rank_true_tails))
            self.filt_rank_true_heads = cat((self.filt_rank_true_heads, filt_rank_true_heads))

        self.evaluated = True

    def evaluate_pair(self, entities, relations, true, filter_list, heads=1):
        """

        Parameters
        ----------
        entities : float tensor
            Tensor of shape (batch_size, ent_emb_dim) containing current embeddings of entities
        relations : float tensor
            Tensor of shape (batch_size, rel_emb_dim) containing current embeddings of relations
        true : integer tensor
            Tensor of shape (batch_size) containing the true entity for each sample.
        filter_list : long tensor
            Tensor of shape (batch_size, -1) containing for each line the
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
        filtered_rank_true_entities : integer tensor
            Tensor of shape (batch_size) containing the rank of the true entities when ranking only\
            true false entities based on computation of d(hear+relation, tail).
            filtered_sorted_candidates : integer tensor
            Tensor of shape (batch_size, self.k_max) containing the k_max best entities ranked by\
            decreasing dissimilarity d(hear+relation, tail) with only true corrupted triplets.

        """
        current_batch_size, embedding_dimension = entities.shape

        # tmp_sum is either heads + relations or relations - tails
        tmp_sum = (heads * entities + relations).view((current_batch_size, embedding_dimension, 1))
        tmp_sum = tmp_sum.expand((current_batch_size, embedding_dimension, self.kg.n_ent))

        # compute either dissimilarity(heads + relation, candidates) or
        # dissimilarity(-candidates, relation - tails)
        # filter out the true negative samples by assigning infinite dissimilarity
        candidates = self.ent_embed.weight.transpose(0, 1)
        dissimilarities = self.dissimilarity(tmp_sum, heads * candidates)

        # this masks lines full of -1
        # (facts for which (entitie, relation) has only one target possible)
        filt_dissimilarities = dissimilarities.clone()
        mask = (filter_list.sum(dim=1) != -filter_list.shape[1])
        if mask.sum().item() > 0:
            filt_dissimilarities[mask] = filt_dissimilarities[mask].scatter_(1, filter_list[mask],
                                                                             float('Inf'))

        # from dissimilarities, extract the rank of the true entity and the k_max top entities.
        rank_true_entities = get_rank(dissimilarities, true)
        filtered_rank_true_entities = get_rank(filt_dissimilarities, true)

        return rank_true_entities, filtered_rank_true_entities

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
        sum_ = (self.rank_true_heads.float().mean() +
                self.rank_true_tails.float().mean()).item()
        filt_sum = (self.filt_rank_true_heads.float().mean() +
                    self.filt_rank_true_tails.float().mean()).item()
        return sum_ / 2, filt_sum / 2

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

        head_hit = (self.rank_true_heads < k).float().mean()
        tail_hit = (self.rank_true_tails < k).float().mean()
        filt_head_hit = (self.filt_rank_true_heads < k).float().mean()
        filt_tail_hit = (self.filt_rank_true_tails < k).float().mean()

        return (head_hit + tail_hit).item() / 2, (filt_head_hit + filt_tail_hit).item() / 2
