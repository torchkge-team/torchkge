# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""

from torch import empty, arange
from tqdm.autonotebook import tqdm

from ..exceptions import NotYetEvaluatedError
from ..utils import DataLoader, get_true_targets, get_rank


class LinkPredictionEvaluator(object):
    """Evaluate performance of given embedding using link prediction method.

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
        This is the same as the `rank_of_true_heads` when is the filtered
        case. See referenced paper by Bordes et al. for more information.
    filt_rank_true_tails: torch.Tensor, shape: (n_facts), dtype: `torch.int`
        This is the same as the `rank_of_true_tails` when is the filtered
        case. See referenced paper by Bordes et al. for more information.
    evaluated: bool
        Indicates if the method LinkPredictionEvaluator.evaluate has already
        been called.
    k_max: bool, default = 10
        Max value to be used to compute the hit@k score.

    References
    ----------
    * Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston,
      and Oksana Yakhnenko.
      Translating Embeddings for Modeling Multi-relational Data.
      In Advances in Neural Information Processing Systems 26, pages 2787–2795,
      2013.
      https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data

    """

    def __init__(self, model, knowledge_graph):
        self.model = model
        self.kg = knowledge_graph
        self.n_ent = model.n_ent
        self.n_rel = model.n_rel

        self.rank_true_heads = empty(size=(knowledge_graph.n_facts,)).long()
        self.rank_true_tails = empty(size=(knowledge_graph.n_facts,)).long()
        self.filt_rank_true_heads = empty(size=(knowledge_graph.n_facts,)
                                          ).long()
        self.filt_rank_true_tails = empty(size=(knowledge_graph.n_facts,)
                                          ).long()

        self.model.eval()
        self.evaluated = False
        self.k_max = 10

    def compute_ranks(self, e_idx, r_idx, true_idx, dictionary, heads=1):
        """Link prediction evaluation helper function. Compute the ranks and
        the filtered ranks of true entities when doing link prediction. Note
        that the best rank possible is 1.

        Parameters
        ----------
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
        device = next(self.model.parameters()).device
        mask = arange(0, self.n_ent, device=device).long()
        mask = mask.view(-1, 1).expand(self.n_ent, b_size).reshape(-1)

        if heads == 1:
            # e_idx are heads of the batch
            scores = self.model.scoring_function(e_idx.repeat(self.n_ent),
                                                 mask,
                                                 r_idx.repeat(self.n_ent))
        else:
            # e_idx are tails of the batch
            scores = self.model.scoring_function(mask,
                                                 e_idx.repeat(self.n_ent),
                                                 r_idx.repeat(self.n_ent))
        scores = scores.view(self.n_ent, -1).transpose(0, 1)

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

    def evaluate(self, b_size, k_max, verbose=True):
        """

        Parameters
        ----------
        b_size: int
            Size of the current batch.
        k_max: int
            Maximal k value we plan to use for Hit@k. This is used to
            truncate tensor so that it fits in memory.
        verbose: bool
            Indicates whether a progress bar should be displayed during
            evaluation.

        """
        self.k_max = k_max
        use_cuda = next(self.model.parameters()).is_cuda

        if use_cuda:
            dataloader = DataLoader(self.kg, batch_size=b_size,
                                    use_cuda='batch')
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
            rk_true_t, f_rk_true_t = self.compute_ranks(h_idx, r_idx, t_idx,
                                                        self.kg.dict_of_tails,
                                                        heads=1)
            rk_true_h, f_rk_true_h = self.compute_ranks(t_idx, r_idx, h_idx,
                                                        self.kg.dict_of_heads,
                                                        heads=-1)

            self.rank_true_heads[i * b_size: (i + 1) * b_size] = rk_true_h
            self.rank_true_tails[i * b_size: (i + 1) * b_size] = rk_true_t

            self.filt_rank_true_heads[i * b_size:
                                      (i + 1) * b_size] = f_rk_true_h
            self.filt_rank_true_tails[i * b_size:
                                      (i + 1) * b_size] = f_rk_true_t

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
