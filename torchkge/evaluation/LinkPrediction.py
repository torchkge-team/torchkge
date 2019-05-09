# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
aboschin@enst.fr
"""

from torch import Tensor, tensor, cat
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
        rank_true_heads : torch tensor, dtype = TODO, shape = TODO
            TODO
        rank_true_tails : torch tensor, dtype = TODO, shape = TODO
            TODO
        evaluated : bool
            Indicates if the method LinkPredictionEvaluator.evaluate() has been called on\
            current object
        k_max : bool, default = 10
            Max value to be used to compute the hit@k score.

    """

    def __init__(self, ent_emb, rel_emb, dissimilarity, knowledge_graph):
        self.ent_embed = ent_emb
        self.rel_embed = rel_emb
        self.dissimilarity = dissimilarity
        self.kg = knowledge_graph

        self.rank_true_heads = Tensor().long()
        self.rank_true_tails = Tensor().long()
        self.filt_rank_true_heads = Tensor().long()
        self.filt_rank_true_tails = Tensor().long()

        self.evaluated = False
        self.k_max = 10

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
        self.k_max = k_max
        use_cuda = self.ent_embed.weight.is_cuda
        dataloader = DataLoader(self.kg, batch_size=batch_size, pin_memory=use_cuda)

        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

            h_idx, t_idx, r_idx = batch[0], batch[1], batch[2]

            h_emb = self.ent_embed.weight[h_idx]
            t_emb = self.ent_embed.weight[t_idx]
            r_emb = self.rel_embed.weight[r_idx]

            if h_idx.is_pinned():
                h_idx, t_idx, r_idx = h_idx.cuda(), t_idx.cuda(), r_idx.cuda()

            # evaluate both ways (head, rel) -> tail and (rel, tail) -> head
            rank_true_tails, filt_rank_true_tails = self.evaluate_pair(h_emb, r_emb, h_idx, r_idx,
                                                                       t_idx,
                                                                       self.kg.dict_of_tails,
                                                                       heads=1)
            rank_true_heads, filt_rank_true_heads = self.evaluate_pair(t_emb, r_emb, t_idx, r_idx,
                                                                       h_idx,
                                                                       self.kg.dict_of_heads,
                                                                       heads=-1)

            self.rank_true_tails = cat((self.rank_true_tails, rank_true_tails))
            self.rank_true_heads = cat((self.rank_true_heads, rank_true_heads))

            self.filt_rank_true_tails = cat((self.filt_rank_true_tails, filt_rank_true_tails))
            self.filt_rank_true_heads = cat((self.filt_rank_true_heads, filt_rank_true_heads))

        self.evaluated = True

    def evaluate_pair(self, e_emb, r_emb, e_idx, r_idx, true_idx, dictionary, heads=1):
        """

        Parameters
        ----------
        e_emb : torch tensor, shape = (batch_size, ent_emb_dim), dtype = float
            Tensor containing current embeddings of entities.
        r_emb : torch tensor, shape = (batch_size, ent_emb_dim), dtype = float
            Tensor containing current embeddings of relations.
        e_idx : torch tensor, shape = (batch_size), dtype = long
            Tensor containing the indices of entities.
        r_idx : torch tensor, shape = (batch_size), dtype = long
            Tensor containing the indices of relations.
        true_idx : torch tensor, shape = (batch_size), dtype = long
            Tensor containing the true entity for each sample.
        dictionary : default dict
            Dictionary of keys (int, int) and values list of ints giving all possible entities for
            the (entity, relation) pair.
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
        current_batch_size, embedding_dimension = e_emb.shape

        # tmp_sum is either heads + r_emb or r_emb - tails
        tmp_sum = (heads * e_emb + r_emb).view((current_batch_size, embedding_dimension, 1))
        tmp_sum = tmp_sum.expand((current_batch_size, embedding_dimension, self.kg.n_ent))

        # compute either dissimilarity(heads + relation, candidates) or
        # dissimilarity(-candidates, relation - tails)
        candidates = self.ent_embed.weight.transpose(0, 1)
        dissimilarities = self.dissimilarity(tmp_sum, heads * candidates)

        # filter out the true negative samples by assigning infinite dissimilarity
        filt_dissimilarities = dissimilarities.clone()
        for i in range(current_batch_size):
            true_targets = dictionary[e_idx[i].item(), r_idx[i].item()].copy()
            # print(true_targets, true_idx[i].item())
            if len(true_targets) == 1:
                continue
            true_targets.remove(true_idx[i].item())
            true_targets = tensor(true_targets).long()
            filt_dissimilarities[i][true_targets] = float('Inf')

        # from dissimilarities, extract the rank of the true entity and the k_max top e_emb.
        rank_true_entities = get_rank(dissimilarities, true_idx)
        filtered_rank_true_entities = get_rank(filt_dissimilarities, true_idx)

        if e_emb.is_cuda:  # in this case model is cuda so tensors are in cuda
            return rank_true_entities.cpu(), filtered_rank_true_entities.cpu()
        else:
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
