# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""

from torch import zeros, cat
from torchkge.sampling import PositionalNegativeSampler
from torchkge.utils import get_batches


class TripletClassificationEvaluator(object):
    """Evaluate performance of given embedding using triplet classification method.

    References
    ----------
    * Richard Socher, Danqi Chen, Christopher D Manning, and Andrew Ng.
      Reasoning With Neural Tensor Networks for Knowledge Base Completion.
      In Advances in Neural Information Processing Systems 26, pages 926–934. 2013.
      https://nlp.stanford.edu/pubs/SocherChenManningNg_NIPS2013.pdf

    Parameters
    ----------
    model: torchkge.models.Model
    kg_val: torchkge.data.KnowledgeGraph.KnowledgeGraph
        Knowledge graph in the form of an object implemented in torchkge.data.KnowledgeGraph.
    kg_test: torchkge.data.KnowledgeGraph.KnowledgeGraph
        Knowledge graph in the form of an object implemented in torchkge.data.KnowledgeGraph.

    Attributes
    ----------
    model: torchkge.models.Model
    kg_val: torchkge.data.KnowledgeGraph.KnowledgeGraph
        Knowledge graph in the form of an object implemented in torchkge.data.KnowledgeGraph.
    kg_test: torchkge.data.KnowledgeGraph.KnowledgeGraph
        Knowledge graph in the form of an object implemented in torchkge.data.KnowledgeGraph.
    use_cuda: bool
        Indicate whether to use cuda or not. It is defined by looking into the device of parameters\
         of the model.
    evaluated: bool
        Indicate whether the `evaluate` function has been called.
    thresholds: float
        Float value of the thresholds for the scoring function to consider a triplet as true. It is\
        defined by calling the `evaluate` function.
    sampler: torchkge.sampling.NegativeSampler
        Negative sampler.

    """

    def __init__(self, model, kg_val, kg_test):
        self.model = model
        self.kg_val = kg_val
        self.kg_test = kg_test
        self.use_cuda = next(self.model.parameters()).is_cuda

        self.evaluated = False
        self.thresholds = None

        self.sampler = PositionalNegativeSampler(self.kg_val, kg_test=self.kg_test)

    def get_scores(self, heads, tails, relations, batch_size):
        """With head, tail and relation indexes, compute the value of the scoring function of the model.

        Parameters
        ----------
        heads: `torch.Tensor`, dtype: `torch.long`, shape: n_facts
            List of heads indices.
        tails: `torch.Tensor`, dtype: `torch.long`, shape: n_facts
            List of tails indices.
        relations: `torch.Tensor`, dtype: `torch.long`, shape: n_facts
            List of relation indices.
        batch_size: int

        Returns
        -------
        scores: `torch.Tensor`, dtype: `torch.float`, shape: n_facts
            List of scores of each triplet.
        """
        scores = []

        if self.use_cuda:
            iterator = enumerate(get_batches(heads.cuda(), tails.cuda(), relations.cuda(), batch_size))
        else:
            iterator = enumerate(get_batches(heads, tails, relations, batch_size))

        for i, batch in iterator:
            h_idx, t_idx, r_idx = batch[0], batch[1], batch[2]

            scores.append(self.model.scoring_function(h_idx, t_idx, r_idx))

        return cat(scores, dim=0)

    def evaluate(self, batch_size):
        """Find relation thresholds using the validation set. As described in the paper by Socher et al., for a
        relation, the threshold is a value t such that if the score of a triplet is larger than t, the fact is true.
        If a relation is not present in any fact of the validation set, then the largest value score of all negative
        samples is used as threshold.

        Parameters
        ----------
        batch_size: int

        """
        r_idx = self.kg_val.relations

        neg_heads, neg_tails = self.sampler.corrupt_kg(batch_size, self.use_cuda, which='main')
        neg_scores = self.get_scores(neg_heads, neg_tails, r_idx, batch_size)

        self.thresholds = zeros(self.kg_val.n_rel)

        for i in range(self.kg_val.n_rel):
            mask = (r_idx == i).bool()
            if mask.sum() > 0:
                self.thresholds[i] = neg_scores[mask].max()
            else:
                self.thresholds[i] = neg_scores.max()

        self.evaluated = True
        self.thresholds.detach_()

    def accuracy(self, batch_size):
        """

        Parameters
        ----------
        batch_size: int

        Returns
        -------
        acc: float
            Share of all triplets (true and negatively sampled ones) that where correctly classified using the
            thresholds learned from the validation set.

        """
        if not self.evaluated:
            self.evaluate(batch_size)

        r_idx = self.kg_test.relations

        neg_heads, neg_tails = self.sampler.corrupt_kg(batch_size, self.use_cuda, which='test')
        scores = self.get_scores(self.kg_test.head_idx, self.kg_test.tail_idx, r_idx, batch_size)
        neg_scores = self.get_scores(neg_heads, neg_tails, r_idx, batch_size)

        if self.use_cuda:
            self.thresholds = self.thresholds.cuda()

        scores = (scores > self.thresholds[r_idx])
        neg_scores = (neg_scores < self.thresholds[r_idx])

        return (scores.sum().item() + neg_scores.sum().item()) / (2 * self.kg_test.n_facts)
