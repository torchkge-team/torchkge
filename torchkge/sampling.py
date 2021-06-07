# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""
import torch
import torch.nn.functional as F

import numpy as np

from torch.nn import Module

from collections import defaultdict

from torch import tensor, bernoulli, randint, ones, rand, cat, cuda

from torchkge.exceptions import NotYetImplementedError, ModelBindFailError
from torchkge.utils.data import DataLoader
from torchkge.utils.operations import get_bernoulli_probs

from torch.distributions.beta import Beta


class NegativeSampler:
    """This is an interface for negative samplers in general.

    Parameters
    ----------
    kg: torchkge.data_structures.KnowledgeGraph
        Main knowledge graph (usually training one).
    kg_val: torchkge.data_structures.KnowledgeGraph (optional)
        Validation knowledge graph.
    kg_test: torchkge.data_structures.KnowledgeGraph (optional)
        Test knowledge graph.
    n_neg: int
        Number of negative sample to create from each fact.

    Attributes
    ----------
    kg: torchkge.data_structures.KnowledgeGraph
        Main knowledge graph (usually training one).
    kg_val: torchkge.data_structures.KnowledgeGraph (optional)
        Validation knowledge graph.
    kg_test: torchkge.data_structures.KnowledgeGraph (optional)
        Test knowledge graph.
    n_ent: int
        Number of entities in the entire knowledge graph. This is the same in
        `kg`, `kg_val` and `kg_test`.
    n_facts: int
        Number of triplets in `kg`.
    n_facts_val: in
        Number of triplets in `kg_val`.
    n_facts_test: int
        Number of triples in `kg_test`.
    n_neg: int
        Number of negative sample to create from each fact.
    """

    def __init__(self, kg, kg_val=None, kg_test=None, n_neg=1):
        self.kg = kg
        self.n_ent = kg.n_ent
        self.n_facts = kg.n_facts

        self.kg_val = kg_val
        self.kg_test = kg_test

        self.n_neg = n_neg

        if kg_val is None:
            self.n_facts_val = 0
        else:
            self.n_facts_val = kg_val.n_facts

        if kg_test is None:
            self.n_facts_test = 0
        else:
            self.n_facts_test = kg_test.n_facts

    def corrupt_batch(self, heads, tails, relations, n_neg):
        raise NotYetImplementedError('NegativeSampler is just an interface, '
                                     'please consider using a child class '
                                     'where this is implemented.')

    def corrupt_kg(self, batch_size, use_cuda, which='main'):
        """Corrupt an entire knowledge graph using a dataloader and by calling
        `corrupt_batch` method.

        Parameters
        ----------
        batch_size: int
            Size of the batches used in the dataloader.
        use_cuda: bool
            Indicate whether to use cuda or not
        which: str
            Indicate which graph should be corrupted. Possible values are :
            * 'main': attribute self.kg is corrupted
            * 'train': attribute self.kg is corrupted
            * 'val': attribute self.kg_val is corrupted. In this case this
            attribute should have been initialized.
            * 'test': attribute self.kg_test is corrupted. In this case this
            attribute should have been initialized.

        Returns
        -------
        neg_heads: torch.Tensor, dtype: torch.long, shape: (n_facts)
            Tensor containing the integer key of negatively sampled heads of
            the relations in the graph designated by `which`.
        neg_tails: torch.Tensor, dtype: torch.long, shape: (n_facts)
            Tensor containing the integer key of negatively sampled tails of
            the relations in the graph designated by `which`.
        """
        assert which in ['main', 'train', 'test', 'val']
        if which == 'val':
            assert self.n_facts_val > 0
        if which == 'test':
            assert self.n_facts_test > 0

        if use_cuda:
            tmp_cuda = 'batch'
        else:
            tmp_cuda = None

        if which == 'val':
            dataloader = DataLoader(self.kg_val, batch_size=batch_size,
                                    use_cuda=tmp_cuda)
        elif which == 'test':
            dataloader = DataLoader(self.kg_test, batch_size=batch_size,
                                    use_cuda=tmp_cuda)
        else:
            dataloader = DataLoader(self.kg, batch_size=batch_size,
                                    use_cuda=tmp_cuda)

        corr_heads, corr_tails = [], []

        for i, batch in enumerate(dataloader):
            heads, tails, rels = batch[0], batch[1], batch[2]
            neg_heads, neg_tails = self.corrupt_batch(heads, tails, rels,
                                                      n_neg=1)

            corr_heads.append(neg_heads)
            corr_tails.append(neg_tails)

        if use_cuda:
            return cat(corr_heads).long().cpu(), cat(corr_tails).long().cpu()
        else:
            return cat(corr_heads).long(), cat(corr_tails).long()


class UniformNegativeSampler(NegativeSampler):
    """Uniform negative sampler as presented in 2013 paper by Bordes et al..
    Either the head or the tail of a triplet is replaced by another entity at
    random. The choice of head/tail is uniform. This class inherits from the
    :class:`torchkge.sampling.NegativeSampler` interface. It
    then has its attributes as well.

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
    kg: torchkge.data_structures.KnowledgeGraph
        Main knowledge graph (usually training one).
    kg_val: torchkge.data_structures.KnowledgeGraph (optional)
        Validation knowledge graph.
    kg_test: torchkge.data_structures.KnowledgeGraph (optional)
        Test knowledge graph.
    n_neg: int
        Number of negative sample to create from each fact.
    """

    def __init__(self, kg, kg_val=None, kg_test=None, n_neg=1):
        super().__init__(kg, kg_val, kg_test, n_neg)

    def corrupt_batch(self, heads, tails, relations=None, n_neg=None):
        """For each true triplet, produce a corrupted one not different from
        any other true triplet. If `heads` and `tails` are cuda objects ,
        then the returned tensors are on the GPU.

        Parameters
        ----------
        heads: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of heads of the relations in the
            current batch.
        tails: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of tails of the relations in the
            current batch.
        relations: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of relations in the current
            batch. This is optional here and mainly present because of the
            interface with other NegativeSampler objects.
        n_neg: int (opt)
            Number of negative sample to create from each fact. It overwrites
            the value set at the construction of the sampler.
        Returns
        -------
        neg_heads: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of negatively sampled heads of
            the relations in the current batch.
        neg_tails: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of negatively sampled tails of
            the relations in the current batch.
        """
        if n_neg is None:
            n_neg = self.n_neg

        device = heads.device
        assert (device == tails.device)

        batch_size = heads.shape[0]
        neg_heads = heads.repeat(n_neg)
        neg_tails = tails.repeat(n_neg)

        # Randomly choose which samples will have head/tail corrupted
        mask = bernoulli(ones(size=(batch_size * n_neg,),
                              device=device) / 2).double()

        n_h_cor = int(mask.sum().item())
        neg_heads[mask == 1] = randint(1, self.n_ent,
                                       (n_h_cor,),
                                       device=device)
        neg_tails[mask == 0] = randint(1, self.n_ent,
                                       (batch_size * n_neg - n_h_cor,),
                                       device=device)

        return neg_heads.long(), neg_tails.long()


class BernoulliNegativeSampler(NegativeSampler):
    """Bernoulli negative sampler as presented in 2014 paper by Wang et al..
    Either the head or the tail of a triplet is replaced by another entity at
    random. The choice of head/tail is done using probabilities taking into
    account profiles of the relations. See the paper for more details. This
    class inherits from the
    :class:`torchkge.sampling.NegativeSampler` interface.
    It then has its attributes as well.

    References
    ----------
    * Zhen Wang, Jianwen Zhang, Jianlin Feng, and Zheng Chen.
      Knowledge Graph Embedding by Translating on Hyperplanes.
      In Twenty-Eighth AAAI Conference on Artificial Intelligence, June 2014.
      https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8531

    Parameters
    ----------
    kg: torchkge.data_structures.KnowledgeGraph
        Main knowledge graph (usually training one).
    kg_val: torchkge.data_structures.KnowledgeGraph (optional)
        Validation knowledge graph.
    kg_test: torchkge.data_structures.KnowledgeGraph (optional)
        Test knowledge graph.
    n_neg: int
        Number of negative sample to create from each fact.
    Attributes
    ----------
    bern_probs: torch.Tensor, dtype: torch.float, shape: (kg.n_rel)
        Bernoulli sampling probabilities. See paper for more details.

    """

    def __init__(self, kg, kg_val=None, kg_test=None, n_neg=1):
        super().__init__(kg, kg_val, kg_test, n_neg)
        self.bern_probs = self.evaluate_probabilities()

    def evaluate_probabilities(self):
        """Evaluate the Bernoulli probabilities for negative sampling as in the
        TransH original paper by Wang et al. (2014).
        """
        bern_probs = get_bernoulli_probs(self.kg)

        tmp = []
        for i in range(self.kg.n_rel):
            if i in bern_probs.keys():
                tmp.append(bern_probs[i])
            else:
                tmp.append(0.5)

        return tensor(tmp).float()

    def corrupt_batch(self, heads, tails, relations, n_neg=None):
        """For each true triplet, produce a corrupted one different from any
        other true triplet. If `heads` and `tails` are cuda objects , then the
        returned tensors are on the GPU.

        Parameters
        ----------
        heads: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of heads of the relations in the
            current batch.
        tails: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of tails of the relations in the
            current batch.
        relations: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of relations in the current
            batch.
        n_neg: int (opt)
            Number of negative sample to create from each fact. It overwrites
            the value set at the construction of the sampler.
        Returns
        -------
        neg_heads: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of negatively sampled heads of
            the relations in the current batch.
        neg_tails: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of negatively sampled tails of
            the relations in the current batch.
        """
        if n_neg is None:
            n_neg = self.n_neg

        device = heads.device
        assert (device == tails.device)

        batch_size = heads.shape[0]
        neg_heads = heads.repeat(n_neg)
        neg_tails = tails.repeat(n_neg)

        # Randomly choose which samples will have head/tail corrupted
        mask = bernoulli(self.bern_probs[relations].repeat(n_neg)).double()
        n_h_cor = int(mask.sum().item())
        neg_heads[mask == 1] = randint(1, self.n_ent,
                                       (n_h_cor,),
                                       device=device)
        neg_tails[mask == 0] = randint(1, self.n_ent,
                                       (batch_size * n_neg - n_h_cor,),
                                       device=device)

        return neg_heads.long(), neg_tails.long()


class PositionalNegativeSampler(BernoulliNegativeSampler):
    """Positional negative sampler as presented in 2011 paper by Socher et al..
    Either the head or the tail of a triplet is replaced by another entity
    chosen among entities that have already appeared at the same place in a
    triplet (involving the same relation). It is not clear in the paper how the
    choice of head/tail is done. We chose to use Bernoulli sampling as in 2014
    paper by Wang et al. as we believe it serves the same purpose as the
    original paper. This class inherits from the
    :class:`torchkge.sampling.BernouilliNegativeSampler` class
    seen as an interface. It then has its attributes as well.

    References
    ----------
    * Richard Socher, Danqi Chen, Christopher D Manning, and Andrew Ng.
      Reasoning With Neural Tensor Networks for Knowledge Base Completion.
      In Advances in Neural Information Processing Systems 26, pages 926–934.,
      2013.
      https://nlp.stanford.edu/pubs/SocherChenManningNg_NIPS2013.pdf
    * Zhen Wang, Jianwen Zhang, Jianlin Feng, and Zheng Chen.
      Knowledge Graph Embedding by Translating on Hyperplanes.
      In Twenty-Eighth AAAI Conference on Artificial Intelligence, June 2014.
      https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8531

    Parameters
    ----------
    kg: torchkge.data_structures.KnowledgeGraph
        Main knowledge graph (usually training one).
    kg_val: torchkge.data_structures.KnowledgeGraph (optional)
        Validation knowledge graph.
    kg_test: torchkge.data_structures.KnowledgeGraph (optional)
        Test knowledge graph.

    Attributes
    ----------
    possible_heads: dict
        keys : relations, values : list of possible heads for each relation.
    possible_tails: dict
        keys : relations, values : list of possible tails for each relation.
    n_poss_heads: list
        List of number of possible heads for each relation.
    n_poss_tails: list
        List of number of possible tails for each relation.

    """

    def __init__(self, kg, kg_val=None, kg_test=None):
        super().__init__(kg, kg_val, kg_test, 1)
        self.possible_heads, self.possible_tails, \
        self.n_poss_heads, self.n_poss_tails = self.find_possibilities()

    def find_possibilities(self):
        """For each relation of the knowledge graph (and possibly the
        validation graph but not the test graph) find all the possible heads
        and tails in the sense of Wang et al., e.g. all entities that occupy
        once this position in another triplet.

        Returns
        -------
        possible_heads: dict
            keys : relation index, values : list of possible heads
        possible tails: dict
            keys : relation index, values : list of possible tails
        n_poss_heads: torch.Tensor, dtype: torch.long, shape: (n_relations)
            Number of possible heads for each relation.
        n_poss_tails: torch.Tensor, dtype: torch.long, shape: (n_relations)
            Number of possible tails for each relation.

        """
        possible_heads, possible_tails = get_possible_heads_tails(self.kg)

        if self.n_facts_val > 0:
            possible_heads, \
                possible_tails = get_possible_heads_tails(self.kg_val,
                                                          possible_heads,
                                                          possible_tails)

        n_poss_heads = []
        n_poss_tails = []

        assert possible_heads.keys() == possible_tails.keys()

        for r in range(self.kg.n_rel):
            if r in possible_heads.keys():
                n_poss_heads.append(len(possible_heads[r]))
                n_poss_tails.append(len(possible_tails[r]))
                possible_heads[r] = list(possible_heads[r])
                possible_tails[r] = list(possible_tails[r])
            else:
                n_poss_heads.append(0)
                n_poss_tails.append(0)
                possible_heads[r] = list()
                possible_tails[r] = list()

        n_poss_heads = tensor(n_poss_heads)
        n_poss_tails = tensor(n_poss_tails)

        return possible_heads, possible_tails, n_poss_heads, n_poss_tails

    def corrupt_batch(self, heads, tails, relations, n_neg=None):
        """For each true triplet, produce a corrupted one not different from
        any other golden triplet. If `heads` and `tails` are cuda objects,
        then the returned tensors are on the GPU.

        Parameters
        ----------
        heads: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of heads of the relations in the
            current batch.
        tails: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of tails of the relations in the
            current batch.
        relations: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of relations in the current
            batch. This is optional here and mainly present because of the
            interface with other NegativeSampler objects.

        Returns
        -------
        neg_heads: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of negatively sampled heads of
            the relations in the current batch.
        neg_tails: torch.Tensor, dtype: torch.long, shape: (batch_size)
            Tensor containing the integer key of negatively sampled tails of
            the relations in the current batch.
        """
        device = heads.device
        assert (device == tails.device)

        batch_size = heads.shape[0]
        neg_heads, neg_tails = heads.clone(), tails.clone()

        # Randomly choose which samples will have head/tail corrupted
        mask = bernoulli(self.bern_probs[relations]).double()
        n_heads_corrupted = int(mask.sum().item())

        # Get the number of possible entities for head and tail
        n_poss_heads = self.n_poss_heads[relations[mask == 1]]
        n_poss_tails = self.n_poss_tails[relations[mask == 0]]

        assert n_poss_heads.shape[0] == n_heads_corrupted
        assert n_poss_tails.shape[0] == batch_size - n_heads_corrupted

        # Choose a rank of an entity in the list of possible entities
        choice_heads = (n_poss_heads.float() *
                        rand((n_heads_corrupted,))).floor().long()
        choice_tails = (n_poss_tails.float() *
                        rand((batch_size - n_heads_corrupted,))).floor().long()

        corr = []
        rels = relations[mask == 1]
        for i in range(n_heads_corrupted):
            r = rels[i].item()
            choices = self.possible_heads[r]
            if len(choices) == 0:
                # in this case the relation r has never been used with any head
                # choose one entity at random
                corr.append(randint(low=0, high=self.n_ent, size=(1,)).item())
            else:
                corr.append(choices[choice_heads[i].item()])
        neg_heads[mask == 1] = tensor(corr, device=device).long()

        corr = []
        rels = relations[mask == 0]
        for i in range(batch_size - n_heads_corrupted):
            r = rels[i].item()
            choices = self.possible_tails[r]
            if len(choices) == 0:
                # in this case the relation r has never been used with any tail
                # choose one entity at random
                corr.append(randint(low=0, high=self.n_ent, size=(1,)).item())
            else:
                corr.append(choices[choice_tails[i].item()])
        neg_tails[mask == 0] = tensor(corr, device=device).long()

        return neg_heads.long(), neg_tails.long()


def get_possible_heads_tails(kg, possible_heads=None, possible_tails=None):
    """Gets for each relation of the knowledge graph the possible heads and
    possible tails.

    Parameters
    ----------
    kg: `torchkge.data_structures.KnowledgeGraph`
    possible_heads: dict, optional (default=None)
    possible_tails: dict, optional (default=None)

    Returns
    -------
    possible_heads: dict, optional (default=None)
        keys: relation indices, values: set of possible heads for each
        relations.
    possible_tails: dict, optional (default=None)
        keys: relation indices, values: set of possible tails for each
        relations.

    """

    if possible_heads is None:
        possible_heads = defaultdict(set)
    else:
        assert type(possible_heads) == dict
        possible_heads = defaultdict(set, possible_heads)
    if possible_tails is None:
        possible_tails = defaultdict(set)
    else:
        assert type(possible_tails) == dict
        possible_tails = defaultdict(set, possible_tails)

    for i in range(kg.n_facts):
        possible_heads[kg.relations[i].item()].add(kg.head_idx[i].item())
        possible_tails[kg.relations[i].item()].add(kg.tail_idx[i].item())

    return dict(possible_heads), dict(possible_tails)


def stage_results(datas, file_pref, epoch):
    import csv

    file_name = 'stage/complex_wn18rr' + file_pref + "_" + str(epoch) +'.csv'
    heads, tails, relations = datas
    for h, t, r in zip(heads, tails, relations):
        row = [h.item(), t.item(),  r.item()]
        with open(file_name, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(row)

#
# class MABSampling(NegativeSampler):
#     def __init__(self, kg, kg_val=None, kg_test=None, model=None, cache_dim=50, epsilon=0.5, n_neg=1):
#         super(MABSampling, self).__init__(kg, kg_val, kg_test, n_neg)
#         self.head_cache, self.tail_cache = defaultdict(list), defaultdict(list)
#         self.head_cache_idx, self.tail_cache_idx = defaultdict(list), defaultdict(list)
#         self.head_rewards_record_actions, self.tail_rewards_record_actions = None, None
#         # self.head_delta, self.tail_delta = None, None
#         self.head_rewards_record_reward, self.tail_rewards_record_reward = None, None
#         self.cache_dim = cache_dim
#         self.model = model
#         # self.embedding_device = model.ent_emb.weight.device
#         self.embedding_device = model.re_ent_emb.weight.device
#         self.epsilon = epsilon
#         self.bern_prob = self.evaluate_probabilities()
#         self.init_cache(kg)
#
#     def evaluate_probabilities(self):
#         """
#         Evaluate the Bernoulli probabilities for negative sampling as in the
#         TransH original paper by Wang et al. (2014).
#         """
#         bern_probs = get_bernoulli_probs(self.kg)
#         tmp = []
#         for i in range(self.kg.n_rel):
#             if i in bern_probs.keys():
#                 tmp.append(bern_probs[i])
#             else:
#                 tmp.append(0.5)
#         return tensor(tmp).float()
#
#     def reward(self, heads, tails, relations):
#         if self.model is None:
#             raise ModelBindFailError('MABSampling is required the referenced model to be bind')
#         score = torch.abs(self.model.scoring_function(heads, tails, relations).data)
#         return F.softmax(score, dim=-1)
#
#     def init_cache(self, kg):
#         device = kg.head_idx.device
#         count_h, count_t = 0, 0
#
#         h_init_reward = []
#         t_init_reward = []
#
#         for h, t, r in zip(kg.head_idx, kg.tail_idx, kg.relations):
#             if not (t.item(), r.item()) in self.head_cache:
#                 h_bandits = torch.randint(low=0, high=self.n_ent, size=(self.cache_dim, ))
#                 tails = torch.ones(self.cache_dim, dtype=torch.int64)*t.item()
#                 relations = torch.ones(self.cache_dim, dtype=torch.int64)*r.item()
#                 if cuda.is_available():
#                     if self.embedding_device == torch.device('cuda:' + str(torch.cuda.current_device())):
#                         h_bandits = h_bandits.cuda()
#                         tails = tails.cuda()
#                         relations = relations.cuda()
#                 h_init_reward.append(self.reward(h_bandits, tails, relations).data)
#                 self.head_cache[(t.item(), r.item())] = h_bandits
#                 self.head_cache_idx[(t.item(), r.item())] = torch.tensor([count_h])
#                 count_h += 1
#
#             if not (h.item(), r.item()) in self.tail_cache:
#                 t_bandits = torch.randint(low=0, high=self.n_ent, size=(self.cache_dim, ))
#                 heads = torch.ones(self.cache_dim, dtype=torch.int64)*h.item()
#                 relations = torch.ones(self.cache_dim, dtype=torch.int64)*r.item()
#                 if cuda.is_available():
#                     if self.embedding_device == torch.device('cuda:' + str(torch.cuda.current_device())):
#                         t_bandits = t_bandits.cuda()
#                         heads = heads.cuda()
#                         relations = relations.cuda()
#                 t_init_reward.append(self.reward(heads, t_bandits, relations).data)
#                 self.tail_cache[(h.item(), r.item())] = t_bandits
#                 self.tail_cache_idx[(h.item(), r.item())] = torch.tensor([count_t])
#                 count_t += 1
#
#         self.head_rewards_record_reward = torch.stack(h_init_reward).data.cpu()
#         self.tail_rewards_record_reward = torch.stack(t_init_reward).data.cpu()
#         self.head_rewards_record_actions = torch.ones([count_h, self.cache_dim], dtype=torch.int32, device=device)
#         self.tail_rewards_record_actions = torch.ones([count_t, self.cache_dim], dtype=torch.int32, device=device)
#         # self.head_delta = torch.zeros([count_h, self.cache_dim], dtype=torch.float32, device=device)
#         # self.tail_delta = torch.zeros([count_t, self.cache_dim], dtype=torch.float32, device=device)
#
#     def update_cache(self):
#         h_update_reward, t_update_reward = [], []
#         h_candidates, t_candidates = [], []
#         h_caches_cand, t_caches_cand = [], []
#
#         for h_cache_key in self.head_cache.keys():
#             (t, r) = h_cache_key
#             h_bandits = torch.randint(low=0, high=self.n_ent, size=(self.cache_dim,))
#             tails = torch.ones(self.cache_dim, dtype=torch.int64) * t
#             relations = torch.ones(self.cache_dim, dtype=torch.int64) * r
#             if cuda.is_available():
#                 if self.embedding_device == torch.device('cuda:' + str(torch.cuda.current_device())):
#                     h_bandits = h_bandits.cuda()
#                     tails = tails.cuda()
#                     relations = relations.cuda()
#             h_update_reward.append(self.reward(h_bandits, tails, relations).data)
#             h_caches_cand.append(self.head_cache[(t, r)])
#             h_candidates.append(h_bandits)
#
#         for t_cache_key in self.tail_cache.keys():
#             (h, r) = t_cache_key
#             t_bandits = torch.randint(low=0, high=self.n_ent, size=(self.cache_dim,))
#             heads = torch.ones(self.cache_dim, dtype=torch.int64) * h
#             relations = torch.ones(self.cache_dim, dtype=torch.int64) * r
#             if cuda.is_available():
#                 if self.embedding_device == torch.device('cuda:' + str(torch.cuda.current_device())):
#                     heads = heads.cuda()
#                     t_bandits = t_bandits.cuda()
#                     relations = relations.cuda()
#             t_update_reward.append(self.reward(heads, t_bandits, relations).data)
#             t_caches_cand.append(self.tail_cache[(h, r)])
#             t_candidates.append(t_bandits)
#
#         head_rewards_record_reward = torch.stack(h_update_reward).data.cpu()
#         tail_rewards_record_reward = torch.stack(t_update_reward).data.cpu()
#         head_rewards_record_actions = torch.ones([len(self.head_cache), self.cache_dim], dtype=torch.int32)
#         tail_rewards_record_actions = torch.ones([len(self.tail_cache), self.cache_dim], dtype=torch.int32)
#         head_avg_reward = torch.div(self.head_rewards_record_reward, self.head_rewards_record_actions)
#         tail_avg_reward = torch.div(self.tail_rewards_record_reward, self.tail_rewards_record_actions)
#
#         head_updated_cache = torch.where(head_avg_reward < head_rewards_record_reward, torch.stack(h_caches_cand), torch.stack(h_candidates))
#         head_updated_reward = torch.where(head_avg_reward < head_rewards_record_reward, self.head_rewards_record_reward, head_rewards_record_reward)
#         head_updated_action = torch.where(head_avg_reward < head_rewards_record_reward, self.head_rewards_record_actions, head_rewards_record_actions)
#
#         tail_updated_cache = torch.where(tail_avg_reward < tail_rewards_record_reward, torch.stack(t_caches_cand), torch.stack(t_candidates))
#         tail_updated_reward = torch.where(tail_avg_reward < tail_rewards_record_reward, self.tail_rewards_record_reward, tail_rewards_record_reward)
#         tail_updated_action = torch.where(tail_avg_reward < tail_rewards_record_reward, self.tail_rewards_record_actions, tail_rewards_record_actions)
#
#         for cache_key, items in zip(self.head_cache, head_updated_cache):
#             self.head_cache[cache_key] = items
#         self.head_rewards_record_reward = head_updated_reward
#         self.head_rewards_record_actions = head_updated_action
#
#         for cache_key, items in zip(self.tail_cache, tail_updated_cache):
#             self.tail_cache[cache_key] = items
#         self.tail_rewards_record_reward = tail_updated_reward
#         self.tail_rewards_record_actions = tail_updated_action
#
#     def corrupt_batch(self, heads, tails, relations, epoch):
#         device = heads.device
#         assert (device == tails.device)
#
#         h_candidates, t_candidates = [], []
#         h_cache_idx, t_cache_idx = [], []
#         for h, t, r in zip(heads, tails, relations):
#             h_candidates.append(self.head_cache[(t.item(), r.item())])
#             t_candidates.append(self.tail_cache[(h.item(), r.item())])
#             h_cache_idx.append(self.head_cache_idx[(t.item(), r.item())])
#             t_cache_idx.append(self.tail_cache_idx[(h.item(), r.item())])
#
#         n_h, h_bandit_idx = self.choose_bandit(torch.stack(h_candidates).cpu(), torch.stack(h_cache_idx), head=True)
#         n_t, t_bandit_idx = self.choose_bandit(torch.stack(t_candidates).cpu(), torch.stack(t_cache_idx), head=False)
#
#         h_rewards = self.reward(n_h, tails, relations).data
#         t_rewards = self.reward(heads, n_t, relations).data
#         self.record_action(
#             torch.reshape(torch.stack(h_cache_idx), (-1,)),
#             h_bandit_idx,
#             h_rewards.cpu(), head=True
#         )
#         self.record_action(
#             torch.reshape(torch.stack(t_cache_idx), (-1,)),
#             t_bandit_idx,
#             t_rewards.cpu(), head=False
#         )
#
#         # Bernoulli sampling to select (h', r, t) and (h, r, t')
#         selection = bernoulli(self.bern_prob[relations]).double()
#         if cuda.is_available():
#             if self.embedding_device == torch.device('cuda:' + str(torch.cuda.current_device())):
#                 selection = selection.cuda()
#                 heads = heads.cuda()
#                 tails = tails.cuda()
#                 n_h = n_h.cuda()
#                 n_t = n_t.cuda()
#         ones = torch.ones([len(selection)], dtype=torch.int32, device=device)
#         neg_heads = torch.where(selection == ones, heads, n_h)
#         neg_tails = torch.where(selection == ones, n_t, tails)
#
#         return neg_heads, neg_tails
#
#     def record_action(self, idx, bandit_idx, reward, head):
#         if head:
#             self.head_rewards_record_actions[idx, bandit_idx] = self.head_rewards_record_actions[idx, bandit_idx] + 1
#             self.head_rewards_record_reward[idx, bandit_idx] = self.head_rewards_record_reward[idx, bandit_idx] + reward
#             # self.head_delta[idx, bandit_idx] = reward - self.head_rewards_record_reward[idx, bandit_idx]
#         else:
#             self.tail_rewards_record_actions[idx, bandit_idx] = self.tail_rewards_record_actions[idx, bandit_idx] + 1
#             self.tail_rewards_record_reward[idx, bandit_idx] = self.tail_rewards_record_reward[idx, bandit_idx] + reward
#             # self.tail_delta[idx, bandit_idx] = reward - self.tail_rewards_record_reward[idx, bandit_idx]
#
#     def choose_bandit(self, bandits, idx, head):
#         device = self.head_rewards_record_actions.device
#
#         epsilons = torch.ones([len(idx)], dtype=torch.float32, device=device) * self.epsilon
#         if head:
#             rewards_record_actions = torch.reshape(self.head_rewards_record_actions[idx], (len(idx), self.cache_dim))
#             # delta = torch.reshape(self.head_delta[idx], (len(idx), self.cache_dim))
#             rewards_record_reward = torch.reshape(self.head_rewards_record_reward[idx], (len(idx), self.cache_dim))
#         else:
#             rewards_record_actions = torch.reshape(self.tail_rewards_record_actions[idx], (len(idx), self.cache_dim))
#             # delta = torch.reshape(self.tail_delta[idx], (len(idx), self.cache_dim))
#             rewards_record_reward = torch.reshape(self.tail_rewards_record_reward[idx], (len(idx), self.cache_dim))
#
#         random_bandit_idx = torch.reshape(torch.randint(low=0, high=self.cache_dim, size=(len(idx), 1), device=device), (-1,))
#         reward_bandit_idx = torch.argmax(torch.div(rewards_record_reward, rewards_record_actions), dim=1)
#         # reward_bandit_idx = torch.argmax(delta, dim=1)
#
#         p = torch.rand(len(idx), device=device)
#         bandit_idx = torch.where(p < epsilons, random_bandit_idx, reward_bandit_idx)
#         bandit = bandits.gather(1, bandit_idx.view(-1, 1))
#         if cuda.is_available():
#             if self.embedding_device == torch.device('cuda:' + str(torch.cuda.current_device())):
#                 bandit = bandit.cuda()
#         return torch.reshape(bandit, (-1,)), bandit_idx


class MFSampler(Module):
    def __init__(self, n_rel, n_ent, n_factors=20):
        super().__init__()
        self.emb_heads = torch.nn.Embedding(n_ent, n_factors)
        self.emb_tails = torch.nn.Embedding(n_ent, n_factors)
        self.emb_heads.weight.data.uniform_(0, 0.0005)
        self.emb_tails.weight.data.uniform_(0, 0.0005)

    def forward(self, heads, tails):
        h = self.emb_heads(heads)
        t = self.emb_tails(tails)
        return (h*t).sum(1)


class MFSampling(NegativeSampler):
    def __init__(self, kg, kg_val=None, kg_test=None, cache_dim=50, n_itter=1000, n_factors=20, n_neg=1):
        super(MFSampling, self).__init__(kg, kg_val, kg_test, n_neg)
        self.model = MFSampler(kg.n_rel, kg.n_ent, n_factors)
        self.cache_dim = cache_dim
        self.bern_prob = self.evaluate_probabilities()
        self.setup_itterations = n_itter
        self.head_cache, self.tail_cache = defaultdict(list), defaultdict(list)
        self.set_up()
        self.create_cache(kg)

    def evaluate_probabilities(self):
        """
        Evaluate the Bernoulli probabilities for negative sampling as in the
        TransH original paper by Wang et al. (2014).
        """
        bern_probs = get_bernoulli_probs(self.kg)
        tmp = []
        for i in range(self.kg.n_rel):
            if i in bern_probs.keys():
                tmp.append(bern_probs[i])
            else:
                tmp.append(0.5)
        return tensor(tmp).float()

    def set_up(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=0.0)
        self.model.train()
        use_cuda = None
        if cuda.is_available():
            cuda.empty_cache()
            self.model.cuda()
            use_cuda = 'all'
        dataloader = DataLoader(self.kg, batch_size=10000, use_cuda=use_cuda)

        for i in range(self.setup_itterations):
            for j, batch in enumerate(dataloader):
                h, t, r = batch[0], batch[1], batch[2]
                hr_hat = self.model(h, t)
                loss = F.mse_loss(hr_hat.type(torch.float64), r.type(torch.float64))
                optimizer.zero_grad()  # reset gradient
                loss.backward()
                optimizer.step()

    def create_cache(self, kg):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        np_entities = np.arange(0, self.kg.n_ent)
        for h, t, r in zip(kg.head_idx, kg.tail_idx, kg.relations):
            if not (h.item(), r.item()) in self.tail_cache:
                p_t = list(kg.dict_of_pos_tails[h.item()])
                t_entities = torch.from_numpy(np.delete(np_entities, p_t, None)).to(device=device)
                head_relation_predictions = torch.round(self.model(h.to(device=device), t_entities)).type(torch.int64)
                np_head_relation_predictions = head_relation_predictions.cpu().numpy()
                # get the tails indices
                np_tail_candidates = np.where(np_head_relation_predictions != r.cpu().numpy())
                np_tail_candidates = np_tail_candidates[0]
                if len(np_tail_candidates) < self.cache_dim:
                    self.tail_cache[(h.item(), r.item())] = torch.randint(0, self.kg.n_ent, (self.cache_dim,))
                else:
                    np_tail_select_indices = np.random.randint(0, len(np_tail_candidates), size=self.cache_dim)
                    n_tails = np.take(np_tail_candidates, np_tail_select_indices)
                    self.tail_cache[(h.item(), r.item())] = torch.from_numpy(n_tails)

            if not (t.item(), r.item()) in self.head_cache:
                p_h = list(kg.dict_of_pos_heads[t.item()])
                h_entities = torch.from_numpy(np.delete(np_entities, p_h, None)).to(device=device)
                tail_relation_predictions = torch.round(self.model(t.to(device=device), h_entities)).type(torch.int64)
                np_tail_relation_predictions = tail_relation_predictions.cpu().numpy()
                # get the tails indices
                np_head_candidates = np.where(np_tail_relation_predictions != r.cpu().numpy())
                np_head_candidates = np_head_candidates[0]
                if len(np_head_candidates) < self.cache_dim:
                    self.head_cache[(t.item(), r.item())] = torch.randint(0, self.kg.n_ent, (self.cache_dim,))
                else:
                    np_head_select_indices = np.random.randint(0, len(np_head_candidates), size=self.cache_dim)
                    n_head = np.take(np_head_candidates, np_head_select_indices)
                    self.head_cache[(t.item(), r.item())] = torch.from_numpy(n_head)
        if cuda.is_available():
            cuda.empty_cache()

    def corrupt_batch(self, heads, tails, relations, n_neg=None):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if n_neg is None:
            n_neg = self.n_neg

        h_candidates, t_candidates = [], []
        for h, t, r in zip(heads, tails, relations):
            head_idx = torch.randint(0, self.cache_dim, (n_neg,))
            tail_idx = torch.randint(0, self.cache_dim, (n_neg,))
            n_heads = self.head_cache[(t.item(), r.item())]
            n_tails = self.tail_cache[(h.item(), r.item())]
            h_candidates.append(n_heads[head_idx])
            t_candidates.append(n_tails[tail_idx])

        n_h = torch.reshape(torch.transpose(torch.stack(h_candidates).data.cpu(), 0, 1), (-1,))
        n_t = torch.reshape(torch.transpose(torch.stack(t_candidates).data.cpu(), 0, 1), (-1,))
        selection = bernoulli(self.bern_prob[relations].repeat(n_neg)).double()
        if cuda.is_available():
            selection = selection.cuda()
            n_h = n_h.cuda()
            n_t = n_t.cuda()
        ones = torch.ones([len(selection)], dtype=torch.int32, device=device)

        n_heads = heads.repeat(n_neg)
        n_tails = tails.repeat(n_neg)

        neg_heads = torch.where(selection == ones, n_heads, n_h)
        neg_tails = torch.where(selection == ones, n_t, n_tails)

        return neg_heads, neg_tails

