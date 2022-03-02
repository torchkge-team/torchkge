# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""

from torch import empty, zeros, cat
from tqdm.autonotebook import tqdm

from .data_structures import SmallKG
from .exceptions import NotYetEvaluatedError
from .sampling import PositionalNegativeSampler
from .utils import DataLoader, get_rank, filter_scores


class RelationPredictionEvaluator(object):
    """Evaluate performance of given embedding using relation prediction method.

        References
        ----------
        * Armand Boschin, Thomas Bonald.
          Enriching Wikidata with Semantified Wikipedia Hyperlinks
          In proceedings of the Wikidata workshop, ISWC2021, 2021.
          http://ceur-ws.org/Vol-2982/paper-6.pdf

        Parameters
        ----------
        model: torchkge.models.interfaces.Model
            Embedding model inheriting from the right interface.
        knowledge_graph: torchkge.data_structures.KnowledgeGraph
            Knowledge graph on which the evaluation will be done.
        directed: bool, optional (default=True)
            Indicates whether the orientation head to tail is known when
            predicting missing relations. If False, then both tests (h, _, t)
            and (t, _, r) are done to find the best scoring triples.

        Attributes
        ----------
        model: torchkge.models.interfaces.Model
            Embedding model inheriting from the right interface.
        kg: torchkge.data_structures.KnowledgeGraph
            Knowledge graph on which the evaluation will be done.
        rank_true_rels: torch.Tensor, shape: (n_facts), dtype: `torch.int`
            For each fact, this is the rank of the true relation when all relations
            are ranked. They are ranked in decreasing order of scoring function
            :math:`f_r(h,t)`.
        filt_rank_true_rels: torch.Tensor, shape: (n_facts), dtype: `torch.int`
            This is the same as the `rank_true_rels` when in the filtered
            case. See referenced paper by Bordes et al. for more information.
        evaluated: bool
            Indicates if the method LinkPredictionEvaluator.evaluate has already
            been called.
        directed: bool, optional (default=True)
            Indicates whether the orientation head to tail is known when
            predicting missing relations. If False, then both tests (h, _, t)
            and (t, _, r) are done to find the best scoring triples.
        """

    def __init__(self, model, knowledge_graph, directed=True):
        self.model = model
        self.kg = knowledge_graph
        self.directed = directed

        self.rank_true_rels = empty(size=(knowledge_graph.n_facts,)).long()
        self.filt_rank_true_rels = empty(size=(knowledge_graph.n_facts,)).long()

        self.evaluated = False

    def evaluate(self, b_size, verbose=True):
        """

        Parameters
        ----------
        b_size: int
            Size of the current batch.
        verbose: bool
            Indicates whether a progress bar should be displayed during
            evaluation.

        """
        use_cuda = next(self.model.parameters()).is_cuda

        if use_cuda:
            dataloader = DataLoader(self.kg, batch_size=b_size, use_cuda='batch')
            self.rank_true_rels = self.rank_true_rels.cuda()
            self.filt_rank_true_rels = self.filt_rank_true_rels.cuda()
        else:
            dataloader = DataLoader(self.kg, batch_size=b_size)

        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                             unit='batch', disable=(not verbose),
                             desc='Relation prediction evaluation'):
            h_idx, t_idx, r_idx = batch[0], batch[1], batch[2]
            h_emb, t_emb, r_emb, candidates = self.model.inference_prepare_candidates(h_idx, t_idx, r_idx, entities=False)

            scores = self.model.inference_scoring_function(h_emb, t_emb, candidates)
            filt_scores = filter_scores(scores, self.kg.dict_of_rels, h_idx, t_idx, r_idx)

            if not self.directed:
                scores_bis = self.model.inference_scoring_function(t_emb, h_emb, candidates)
                filt_scores_bis = filter_scores(scores_bis, self.kg.dict_of_rels, h_idx, t_idx, r_idx)

                scores = cat((scores, scores_bis), dim=1)
                filt_scores = cat((filt_scores, filt_scores_bis), dim=1)

            self.rank_true_rels[i * b_size: (i + 1) * b_size] = get_rank(scores, r_idx).detach()
            self.filt_rank_true_rels[i * b_size: (i + 1) * b_size] = get_rank(filt_scores, r_idx).detach()

        self.evaluated = True

        if use_cuda:
            self.rank_true_rels = self.rank_true_rels.cpu()
            self.filt_rank_true_rels = self.filt_rank_true_rels.cpu()

    def mean_rank(self):
        """

        Returns
        -------
        mean_rank: float
            Mean rank of the true entity when replacing alternatively head
            and tail in any fact of the dataset.
        filt_mean_rank: float
            Filtered mean rank of the true entity when replacing
            alternatively head and tail in any fact of the dataset.

        """
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')
        sum_ = self.rank_true_rels.float().mean().item()
        filt_sum = self.filt_rank_true_rels.float().mean().item()
        return sum_, filt_sum

    def hit_at_k(self, k=10):
        """

        Parameters
        ----------
        k: int
            Hit@k is the number of entities that show up in the top k that
            give facts present in the dataset.

        Returns
        -------
        avg_hitatk: float
            Average of hit@k for head and tail replacement.
        filt_avg_hitatk: float
            Filtered average of hit@k for head and tail replacement.

        """
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')

        return (self.rank_true_rels <= k).float().mean().item(), (self.filt_rank_true_rels <= k).float().mean().item()

    def mrr(self):
        """

        Returns
        -------
        avg_mrr: float
            Average of mean recovery rank for head and tail replacement.
        filt_avg_mrr: float
            Filtered average of mean recovery rank for head and tail
            replacement.

        """
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')
        mrr = (self.rank_true_rels.float()**(-1)).mean()
        filt_mrr = (self.filt_rank_true_rels.float()**(-1)).mean()

        return mrr.item(), filt_mrr.item()

    def print_results(self, k=None, n_digits=3):
        """

        Parameters
        ----------
        k: int or list
            k (or list of k) such that hit@k will be printed.
        n_digits: int
            Number of digits to be printed for hit@k and MRR.
        """
        if k is None:
            k = 10

        if k is not None and type(k) == int:
            print('Hit@{} : {} \t\t Filt. Hit@{} : {}'.format(
                k, round(self.hit_at_k(k=k)[0], n_digits),
                k, round(self.hit_at_k(k=k)[1], n_digits)))
        if k is not None and type(k) == list:
            for i in k:
                print('Hit@{} : {} \t\t Filt. Hit@{} : {}'.format(
                    i, round(self.hit_at_k(k=i)[0], n_digits),
                    i, round(self.hit_at_k(k=i)[1], n_digits)))

        print('Mean Rank : {} \t Filt. Mean Rank : {}'.format(
            int(self.mean_rank()[0]), int(self.mean_rank()[1])))
        print('MRR : {} \t\t Filt. MRR : {}'.format(
            round(self.mrr()[0], n_digits), round(self.mrr()[1], n_digits)))


class LinkPredictionEvaluator(object):
    """Evaluate performance of given embedding using link prediction method.

    References
    ----------
    * Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston,
      and Oksana Yakhnenko.
      Translating Embeddings for Modeling Multi-relational Data.
      In Advances in Neural Information Processing Systems 26, pages 2787–2795,
      2013.
      https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data

    Parameters
    ----------
    model: torchkge.models.interfaces.Model
        Embedding model inheriting from the right interface.
    knowledge_graph: torchkge.data_structures.KnowledgeGraph
        Knowledge graph on which the evaluation will be done.

    Attributes
    ----------
    model: torchkge.models.interfaces.Model
        Embedding model inheriting from the right interface.
    kg: torchkge.data_structures.KnowledgeGraph
        Knowledge graph on which the evaluation will be done.
    rank_true_heads: torch.Tensor, shape: (n_facts), dtype: `torch.int`
        For each fact, this is the rank of the true head when all entities
        are ranked as possible replacement of the head entity. They are
        ranked in decreasing order of scoring function :math:`f_r(h,t)`.
    rank_true_tails: torch.Tensor, shape: (n_facts), dtype: `torch.int`
        For each fact, this is the rank of the true tail when all entities
        are ranked as possible replacement of the tail entity. They are
        ranked in decreasing order of scoring function :math:`f_r(h,t)`.
    filt_rank_true_heads: torch.Tensor, shape: (n_facts), dtype: `torch.int`
        This is the same as the `rank_of_true_heads` when in the filtered
        case. See referenced paper by Bordes et al. for more information.
    filt_rank_true_tails: torch.Tensor, shape: (n_facts), dtype: `torch.int`
        This is the same as the `rank_of_true_tails` when in the filtered
        case. See referenced paper by Bordes et al. for more information.
    evaluated: bool
        Indicates if the method LinkPredictionEvaluator.evaluate has already
        been called.

    """

    def __init__(self, model, knowledge_graph):
        self.model = model
        self.kg = knowledge_graph

        self.rank_true_heads = empty(size=(knowledge_graph.n_facts,)).long()
        self.rank_true_tails = empty(size=(knowledge_graph.n_facts,)).long()
        self.filt_rank_true_heads = empty(size=(knowledge_graph.n_facts,)).long()
        self.filt_rank_true_tails = empty(size=(knowledge_graph.n_facts,)).long()

        self.evaluated = False

    def evaluate(self, b_size, verbose=True):
        """

        Parameters
        ----------
        b_size: int
            Size of the current batch.
        verbose: bool
            Indicates whether a progress bar should be displayed during
            evaluation.

        """
        use_cuda = next(self.model.parameters()).is_cuda

        if use_cuda:
            dataloader = DataLoader(self.kg, batch_size=b_size, use_cuda='batch')
            self.rank_true_heads = self.rank_true_heads.cuda()
            self.rank_true_tails = self.rank_true_tails.cuda()
            self.filt_rank_true_heads = self.filt_rank_true_heads.cuda()
            self.filt_rank_true_tails = self.filt_rank_true_tails.cuda()
        else:
            dataloader = DataLoader(self.kg, batch_size=b_size)

        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                             unit='batch', disable=(not verbose),
                             desc='Link prediction evaluation'):
            h_idx, t_idx, r_idx = batch[0], batch[1], batch[2]
            h_emb, t_emb, r_emb, candidates = self.model.inference_prepare_candidates(h_idx, t_idx, r_idx, entities=True)

            scores = self.model.inference_scoring_function(h_emb, candidates, r_emb)
            filt_scores = filter_scores(scores, self.kg.dict_of_tails, h_idx, r_idx, t_idx)
            self.rank_true_tails[i * b_size: (i + 1) * b_size] = get_rank(scores, t_idx).detach()
            self.filt_rank_true_tails[i * b_size: (i + 1) * b_size] = get_rank(filt_scores, t_idx).detach()

            scores = self.model.inference_scoring_function(candidates, t_emb, r_emb)
            filt_scores = filter_scores(scores, self.kg.dict_of_heads, t_idx, r_idx, h_idx)
            self.rank_true_heads[i * b_size: (i + 1) * b_size] = get_rank(scores, h_idx).detach()
            self.filt_rank_true_heads[i * b_size: (i + 1) * b_size] = get_rank(filt_scores, h_idx).detach()

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
            Mean rank of the true entity when replacing alternatively head
            and tail in any fact of the dataset.
        filt_mean_rank: float
            Filtered mean rank of the true entity when replacing
            alternatively head and tail in any fact of the dataset.

        """
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')
        sum_ = (self.rank_true_heads.float().mean() +
                self.rank_true_tails.float().mean()).item()
        filt_sum = (self.filt_rank_true_heads.float().mean() +
                    self.filt_rank_true_tails.float().mean()).item()
        return sum_ / 2, filt_sum / 2

    def hit_at_k_heads(self, k=10):
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')
        head_hit = (self.rank_true_heads <= k).float().mean()
        filt_head_hit = (self.filt_rank_true_heads <= k).float().mean()

        return head_hit.item(), filt_head_hit.item()

    def hit_at_k_tails(self, k=10):
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')
        tail_hit = (self.rank_true_tails <= k).float().mean()
        filt_tail_hit = (self.filt_rank_true_tails <= k).float().mean()

        return tail_hit.item(), filt_tail_hit.item()

    def hit_at_k(self, k=10):
        """

        Parameters
        ----------
        k: int
            Hit@k is the number of entities that show up in the top k that
            give facts present in the dataset.

        Returns
        -------
        avg_hitatk: float
            Average of hit@k for head and tail replacement.
        filt_avg_hitatk: float
            Filtered average of hit@k for head and tail replacement.

        """
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')

        head_hit, filt_head_hit = self.hit_at_k_heads(k=k)
        tail_hit, filt_tail_hit = self.hit_at_k_tails(k=k)

        return (head_hit + tail_hit) / 2, (filt_head_hit + filt_tail_hit) / 2

    def mrr(self):
        """

        Returns
        -------
        avg_mrr: float
            Average of mean recovery rank for head and tail replacement.
        filt_avg_mrr: float
            Filtered average of mean recovery rank for head and tail
            replacement.

        """
        if not self.evaluated:
            raise NotYetEvaluatedError('Evaluator not evaluated call '
                                       'LinkPredictionEvaluator.evaluate')
        head_mrr = (self.rank_true_heads.float()**(-1)).mean()
        tail_mrr = (self.rank_true_tails.float()**(-1)).mean()
        filt_head_mrr = (self.filt_rank_true_heads.float()**(-1)).mean()
        filt_tail_mrr = (self.filt_rank_true_tails.float()**(-1)).mean()

        return ((head_mrr + tail_mrr).item() / 2,
                (filt_head_mrr + filt_tail_mrr).item() / 2)

    def print_results(self, k=None, n_digits=3):
        """

        Parameters
        ----------
        k: int or list
            k (or list of k) such that hit@k will be printed.
        n_digits: int
            Number of digits to be printed for hit@k and MRR.
        """
        if k is None:
            k = 10

        if k is not None and type(k) == int:
            print('Hit@{} : {} \t\t Filt. Hit@{} : {}'.format(
                k, round(self.hit_at_k(k=k)[0], n_digits),
                k, round(self.hit_at_k(k=k)[1], n_digits)))
        if k is not None and type(k) == list:
            for i in k:
                print('Hit@{} : {} \t\t Filt. Hit@{} : {}'.format(
                    i, round(self.hit_at_k(k=i)[0], n_digits),
                    i, round(self.hit_at_k(k=i)[1], n_digits)))

        print('Mean Rank : {} \t Filt. Mean Rank : {}'.format(
            int(self.mean_rank()[0]), int(self.mean_rank()[1])))
        print('MRR : {} \t\t Filt. MRR : {}'.format(
            round(self.mrr()[0], n_digits), round(self.mrr()[1], n_digits)))


class TripletClassificationEvaluator(object):
    """Evaluate performance of given embedding using triplet classification
    method.

    References
    ----------
    * Richard Socher, Danqi Chen, Christopher D Manning, and Andrew Ng.
      Reasoning With Neural Tensor Networks for Knowledge Base Completion.
      In Advances in Neural Information Processing Systems 26, pages 926–934.
      2013.
      https://nlp.stanford.edu/pubs/SocherChenManningNg_NIPS2013.pdf

    Parameters
    ----------
    model: torchkge.models.interfaces.Model
        Embedding model inheriting from the right interface.
    kg_val: torchkge.data_structures.KnowledgeGraph
        Knowledge graph on which the validation thresholds will be computed.
    kg_test: torchkge.data_structures.KnowledgeGraph
        Knowledge graph on which the testing evaluation will be done.

    Attributes
    ----------
    model: torchkge.models.interfaces.Model
        Embedding model inheriting from the right interface.
    kg_val: torchkge.data_structures.KnowledgeGraph
        Knowledge graph on which the validation thresholds will be computed.
    kg_test: torchkge.data_structures.KnowledgeGraph
        Knowledge graph on which the evaluation will be done.
    evaluated: bool
        Indicate whether the `evaluate` function has been called.
    thresholds: float
        Value of the thresholds for the scoring function to consider a
        triplet as true. It is defined by calling the `evaluate` method.
    sampler: torchkge.sampling.NegativeSampler
        Negative sampler.

    """

    def __init__(self, model, kg_val, kg_test):
        self.model = model
        self.kg_val = kg_val
        self.kg_test = kg_test
        self.is_cuda = next(self.model.parameters()).is_cuda

        self.evaluated = False
        self.thresholds = None

        self.sampler = PositionalNegativeSampler(self.kg_val,
                                                 kg_test=self.kg_test)

    def get_scores(self, heads, tails, relations, batch_size):
        """With head, tail and relation indexes, compute the value of the
        scoring function of the model.

        Parameters
        ----------
        heads: torch.Tensor, dtype: torch.long, shape: n_facts
            List of heads indices.
        tails: torch.Tensor, dtype: torch.long, shape: n_facts
            List of tails indices.
        relations: torch.Tensor, dtype: torch.long, shape: n_facts
            List of relation indices.
        batch_size: int

        Returns
        -------
        scores: torch.Tensor, dtype: torch.float, shape: n_facts
            List of scores of each triplet.
        """
        scores = []

        small_kg = SmallKG(heads, tails, relations)
        if self.is_cuda:
            dataloader = DataLoader(small_kg, batch_size=batch_size,
                                    use_cuda='batch')
        else:
            dataloader = DataLoader(small_kg, batch_size=batch_size)

        for i, batch in enumerate(dataloader):
            h_idx, t_idx, r_idx = batch[0], batch[1], batch[2]
            scores.append(self.model.scoring_function(h_idx, t_idx, r_idx))

        return cat(scores, dim=0)

    def evaluate(self, b_size):
        """Find relation thresholds using the validation set. As described in
        the paper by Socher et al., for a relation, the threshold is a value t
        such that if the score of a triplet is larger than t, the fact is true.
        If a relation is not present in any fact of the validation set, then
        the largest value score of all negative samples is used as threshold.

        Parameters
        ----------
        b_size: int
            Batch size.
        """
        r_idx = self.kg_val.relations

        neg_heads, neg_tails = self.sampler.corrupt_kg(b_size, self.is_cuda,
                                                       which='main')
        neg_scores = self.get_scores(neg_heads, neg_tails, r_idx, b_size)

        self.thresholds = zeros(self.kg_val.n_rel)

        for i in range(self.kg_val.n_rel):
            mask = (r_idx == i).bool()
            if mask.sum() > 0:
                self.thresholds[i] = neg_scores[mask].max()
            else:
                self.thresholds[i] = neg_scores.max()

        self.evaluated = True
        self.thresholds.detach_()

    def accuracy(self, b_size):
        """

        Parameters
        ----------
        b_size: int
            Batch size.

        Returns
        -------
        acc: float
            Share of all triplets (true and negatively sampled ones) that where
            correctly classified using the thresholds learned from the
            validation set.

        """
        if not self.evaluated:
            self.evaluate(b_size)

        r_idx = self.kg_test.relations

        neg_heads, neg_tails = self.sampler.corrupt_kg(b_size,
                                                       self.is_cuda,
                                                       which='test')
        scores = self.get_scores(self.kg_test.head_idx,
                                 self.kg_test.tail_idx,
                                 r_idx,
                                 b_size)
        neg_scores = self.get_scores(neg_heads, neg_tails, r_idx, b_size)

        if self.is_cuda:
            self.thresholds = self.thresholds.cuda()

        scores = (scores > self.thresholds[r_idx])
        neg_scores = (neg_scores < self.thresholds[r_idx])

        return (scores.sum().item() +
                neg_scores.sum().item()) / (2 * self.kg_test.n_facts)
