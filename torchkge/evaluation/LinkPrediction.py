# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
aboschin@enst.fr
"""

from torch import empty
from torch.utils.data import DataLoader
from torchkge.exceptions import NotYetEvaluatedError

from tqdm import tqdm


class LinkPredictionEvaluator(object):

    """Evaluate performance of given embedding using link prediction method.

    Parameters
    ----------
    model: torchkge model
    knowledge_graph: torchkge.data.KnowledgeGraph
        Knowledge graph in the form of an object implemented in torchkge.data.KnowledgeGraph.

    Attributes
    ----------
    model: torchkge model
    kg: torchkge.data.KnowledgeGraph.KnowledgeGraph
        Knowledge graph in the form of an object implemented in torchkge.data.KnowledgeGraph.
    rank_true_heads: torch tensor, shape = (n_facts), dtype = int
        Rank of the true head when all possible entities are ranked in term of dissimilarity\
        with tail - relation.
    rank_true_tails: torch tensor, shape = (n_facts), dtype = int
        Rank of the true tail when all possible entities are ranked in term of dissimilarity\
        with head + relation.
    filt_rank_true_heads: torch tensor, shape = (n_facts), dtype = int
        Filtered rank of the true tail when all possible entities are ranked in term of\
        dissimilarity with head + relation.
    filt_rank_true_tails: torch tensor, shape = (n_facts), dtype = int
        Filtered rank of the true tail when all possible entities are ranked in term of\
        dissimilarity with head + relation.
    evaluated: bool
        Indicates if the method LinkPredictionEvaluator.evaluate() has been called on\
        current object
    k_max: bool, default = 10
        Max value to be used to compute the hit@k score.

    References
    ----------
    * Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko.
      Translating Embeddings for Modeling Multi-relational Data.
      In Advances in Neural Information Processing Systems 26, pages 2787–2795, 2013.
      https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data

    """

    def __init__(self, model, knowledge_graph):
        self.model = model
        self.kg = knowledge_graph

        self.rank_true_heads = empty(size=(self.kg.n_sample,)).long()
        self.rank_true_tails = empty(size=(self.kg.n_sample,)).long()
        self.filt_rank_true_heads = empty(size=(self.kg.n_sample,)).long()
        self.filt_rank_true_tails = empty(size=(self.kg.n_sample,)).long()

        self.evaluated = False
        self.k_max = 10

    def evaluate(self, batch_size, k_max):
        """

        Parameters
        ----------
        batch_size: int
            Size of the current batch.
        k_max: int
            Maximal k value we plan to use for Hit@k. This is used to truncate tensor so that it\
            fits in memory.

        """
        self.k_max = k_max
        use_cuda = self.model.entity_embeddings.weight.is_cuda
        dataloader = DataLoader(self.kg, batch_size=batch_size, pin_memory=use_cuda)

        if use_cuda:
            self.rank_true_heads = self.rank_true_heads.cuda()
            self.rank_true_tails = self.rank_true_tails.cuda()
            self.filt_rank_true_heads = self.filt_rank_true_heads.cuda()
            self.filt_rank_true_tails = self.filt_rank_true_tails.cuda()

        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

            h_idx, t_idx, r_idx = batch[0], batch[1], batch[2]

            if h_idx.is_pinned():
                h_idx, t_idx, r_idx = h_idx.cuda(), t_idx.cuda(), r_idx.cuda()

            rank_true_tails, filt_rank_true_tails, rank_true_heads, filt_rank_true_heads \
                = self.model.evaluate_candidates(h_idx, t_idx, r_idx, self.kg)

            self.rank_true_heads[i * batch_size: (i + 1) * batch_size] = rank_true_heads
            self.rank_true_tails[i * batch_size: (i + 1) * batch_size] = rank_true_tails

            self.filt_rank_true_heads[i * batch_size: (i + 1) * batch_size] = filt_rank_true_heads
            self.filt_rank_true_tails[i * batch_size: (i + 1) * batch_size] = filt_rank_true_tails

        self.evaluated = True

        if use_cuda:
            self.rank_true_heads = self.rank_true_heads.cpu()
            self.rank_true_tails = self.rank_true_tails.cpu()
            self.filt_rank_true_heads = self.filt_rank_true_heads.cpu()
            self.filt_rank_true_tails = self.filt_rank_true_tails.cpu()

    def mean_rank(self):
        """

        Returns
        -------
        mean_rank: float
            Mean rank of the true entity when replacing alternatively head and tail in\
            any fact of the dataset.
        filt_mean_rank: float
            Filtered mean rank of the true entity when replacing alternatively head and tail in\
            any fact of the dataset.

        """
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call LinkPredictionEvaluator.evaluate')
        sum_ = (self.rank_true_heads.float().mean() +
                self.rank_true_tails.float().mean()).item()
        filt_sum = (self.filt_rank_true_heads.float().mean() +
                    self.filt_rank_true_tails.float().mean()).item()
        return sum_ / 2, filt_sum / 2

    def hit_at_k(self, k=10):
        """

        Parameters
        ----------
        k: int
            Hit@k is the number of entities that show up in the top k that give facts present\
            in the dataset.

        Returns
        -------
        avg_hitatk: float
            Average of hit@k for head and tail replacement. Computation is done in a\
            vectorized way.
        filt_avg_hitatk: float
            Filtered Average of hit@k for head and tail replacement. Computation is done in a\
            vectorized way.

        """
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call LinkPredictionEvaluator.evaluate')

        head_hit = (self.rank_true_heads < k).float().mean()
        tail_hit = (self.rank_true_tails < k).float().mean()
        filt_head_hit = (self.filt_rank_true_heads < k).float().mean()
        filt_tail_hit = (self.filt_rank_true_tails < k).float().mean()

        return (head_hit + tail_hit).item() / 2, (filt_head_hit + filt_tail_hit).item() / 2

    def mrr(self):
        """

        Returns
        -------
        avg_mrr: float
            Average of mean recovery rank for head and tail replacement. Computation is done in a\
            vectorized way.
        filt_avg_mrr: float
            Filtered Average of mean recovery rank for head and tail replacement. Computation is\
            done in a vectorized way.

        """
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call LinkPredictionEvaluator.evaluate')

        head_mrr = (self.rank_true_heads.float()**(-1)).mean()
        tail_mrr = (self.rank_true_tails.float()**(-1)).mean()
        filt_head_mrr = (self.filt_rank_true_heads.float()**(-1)).mean()
        filt_tail_mrr = (self.filt_rank_true_tails.float()**(-1)).mean()

        return (head_mrr + tail_mrr).item() / 2, (filt_head_mrr + filt_tail_mrr).item() / 2
