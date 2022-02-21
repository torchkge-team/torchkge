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
from sklearn.metrics import roc_auc_score
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


class MDModel(Module):
    """Referenced with  MDNCaching Negative Sampling. Model latent relations
    in MDNCaching.

    Parameters
    ----------
    n_ent: int
        Entity count in the knowledge graph.
    n_factors: int, optional (default=20)
        Matrix Factorization model feature space size.
    Attributes
    ----------
    emb_heads: `torch.nn.Embedding`, shape: (n_ent, n_factors)
        Embeddings of the head entities, initialized with uniform
        distribution.
    emb_tails: `torch.nn.Embedding`, shape: (n_ent, n_factors)
        Embeddings of the tail entities, initialized with uniform
        distribution.
    """

    def __init__(self, n_ent, n_factors=20):
        super().__init__()
        self.emb_heads = torch.nn.Embedding(n_ent, n_factors)
        self.emb_tails = torch.nn.Embedding(n_ent, n_factors)
        self.emb_heads.weight.data.uniform_(0, 0.0005)
        self.emb_tails.weight.data.uniform_(0, 0.0005)

    def forward(self, heads, tails):
        h = self.emb_heads(heads)
        t = self.emb_tails(tails)
        return (h*t).sum(1)

    def latent_relation_prediction(self, heads, tails, relations):
        r = torch.round(self.forward(heads, tails)).type(torch.int64)
        correct = torch.where(r==relations, 1, 0)
        csum = torch.sum(correct, dtype=torch.int64)
        return csum


class MDNCaching(NegativeSampler):
    def __init__(self, kg, model, kg_val=None, kg_test=None, cache_dim=50,
                 n_itter=1000, k=1, growth=0.1, n_factors=20, n_neg=1):

        super(MDNCaching, self).__init__(kg, kg_val, kg_test, n_neg)

        self.md_model = MDModel(kg.n_ent, n_factors)
        self.cache_dim = cache_dim
        self.kg_model = model
        self.update_cache_ittr = 0
        self.temp_batch = 0
        self.batch_count = 0
        self.itteration_track = 0
        self.init_cand_heads = {}
        self.init_cand_tails = {}
        self.n_facts = kg.n_facts
        self.k = k
        self.growth = growth
        self.bern_prob = self.evaluate_probabilities()
        self.setup_itterations = n_itter
        self.head_cache, self.tail_cache = defaultdict(list), defaultdict(list)
        self.train_md_model()
        self.test_md_model()
        self.create_cache()

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

    def reward(self, heads, tails, relations):
        if self.kg_model is None:
            raise ModelBindFailError('MDNSampling is required the referenced model to be bind')
        score = torch.abs(self.kg_model.scoring_function(heads, tails, relations).data)
        return score

    def train_md_model(self):
        optimizer = torch.optim.Adam(self.md_model.parameters(), lr=0.01, weight_decay=0.0)
        self.md_model.train()
        use_cuda = None
        if cuda.is_available():
            cuda.empty_cache()
            self.md_model.cuda()
            use_cuda = 'all'
        dataloader = DataLoader(self.kg, batch_size=100000, use_cuda=use_cuda)

        for i in range(self.setup_itterations):
            for j, batch in enumerate(dataloader):
                h, t, r = batch[0], batch[1], batch[2]
                optimizer.zero_grad()  # reset gradient
                hr_hat = self.md_model(h, t)
                loss = F.mse_loss(hr_hat.type(torch.float64), r.type(torch.float64))
                loss.backward()
                optimizer.step()

    def test_md_model(self):
        use_cuda = None
        correct = 0
        if cuda.is_available():
            cuda.empty_cache()
            self.md_model.cuda()
            use_cuda = 'all'
        total = len(self.kg_test)
        dataloader = DataLoader(self.kg_test, batch_size=10000, use_cuda=use_cuda)
        for j, batch in enumerate(dataloader):
            h, t, r = batch[0], batch[1], batch[2]
            correct += self.md_model.latent_relation_prediction(h, t, r)
        accuracy = (correct/total)*100
        print("md relation prediction accuracy:" + str(accuracy.data))

    def create_cache(self):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        head_cache, tail_cache = {}, {}
        head_pos, tail_pos = [], []
        head_idx, tail_idx = [], []
        count_h, count_t = 0, 0

        for h, t, r in zip(self.kg.head_idx, self.kg.tail_idx, self.kg.relations):
            if not (t.item(), r.item()) in head_cache:
                head_cache[(t.item(), r.item())] = count_h
                head_pos.append([h.item()])
                count_h += 1
            else:
                head_pos[head_cache[(t.item(), r.item())]].append(h.item())

            if not (h.item(), r.item()) in tail_cache:
                tail_cache[(h.item(), r.item())] = count_t
                tail_pos.append([t.item()])
                count_t += 1
            else:
                tail_pos[tail_cache[(h.item(), r.item())]].append(t.item())

            head_idx.append(head_cache[(t.item(), r.item())])
            tail_idx.append(tail_cache[(h.item(), r.item())])
        head_idx = np.array(head_idx, dtype=int)
        tail_idx = np.array(tail_idx, dtype=int)
        self.head_cache = np.random.randint(low=0, high=self.n_ent, size=(count_h, self.cache_dim))
        self.tail_cache = np.random.randint(low=0, high=self.n_ent, size=(count_t, self.cache_dim))

        np_entities = np.arange(self.kg.n_ent)
        head_positives = np.take(head_pos, head_idx)
        tail_positives = np.take(tail_pos, tail_idx)
        for h, t, r, h_pos, t_pos, h_idx, t_idx in zip(self.kg.head_idx, self.kg.tail_idx, self.kg.relations,
                                                       head_positives, tail_positives, head_idx, tail_idx):
            filt_heads = self.filter_true_positives(h_pos, np_entities, device=device)
            filt_tails = self.filter_true_positives(t_pos, np_entities, device=device)
            head_rel_pred = torch.round(self.md_model(filt_heads, t.to(device=device)))
            tail_rel_pred = torch.round(self.md_model(h.to(device=device), filt_tails))
            self.init_cand_heads[h_idx] = self.filter_false_positives(head_rel_pred, r, filt_heads).detach().cpu().numpy()
            self.init_cand_tails[t_idx] = self.filter_false_positives(tail_rel_pred, r, filt_tails).detach().cpu().numpy()

        self.head_idx, self.tail_idx = head_idx, tail_idx
        self.head_pos, self.tail_pos = head_pos, tail_pos
        self.update_cache()

    def update_cache(self):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        temp_batch = 0
        use_cuda = None

        if cuda.is_available():
            cuda.empty_cache()
            self.md_model.cuda()
            use_cuda = 'all'

        dataloader = DataLoader(self.kg, batch_size=10000, use_cuda=use_cuda)
        h_track = np.zeros(len(self.head_cache), dtype=int)
        t_track = np.zeros(len(self.tail_cache), dtype=int)

        for j, batch in enumerate(dataloader):
            h, t, r = batch[0], batch[1], batch[2]
            rewards = self.reward(h, t, r)
            k_fac_rew = torch.mul(rewards, self.k)

            batch_size = len(h)
            start = temp_batch
            end = start + batch_size
            head_idx = self.head_idx[start: end]
            tail_idx = self.tail_idx[start: end]
            head_pos = np.take(self.head_pos, head_idx)
            tail_pos = np.take(self.tail_pos, tail_idx)
            temp_batch = end

            for h, t, r, rew, h_pos, t_pos, h_idx, t_idx in zip(h, t, r, k_fac_rew, head_pos, tail_pos, head_idx, tail_idx):
                if h_track[h_idx] == 0:
                    filt_heads = torch.tensor(self.init_cand_heads[h_idx], device=device)
                    h_cache = self.score_filter(filt_heads, h, t, r, rew, device, head=True)
                    self.head_cache[h_idx] = h_cache.detach().cpu().numpy()
                    h_track[h_idx] = 1

                if t_track[t_idx] == 0:
                    filt_tails = torch.tensor(self.init_cand_tails[t_idx], device=device)
                    t_cache = self.score_filter(filt_tails, h, t, r, rew, device, head=False)
                    self.tail_cache[t_idx] = t_cache.detach().cpu().numpy()
                    t_track[t_idx] = 1

        self.k = self.k + self.growth
        if cuda.is_available():
            cuda.empty_cache()
        self.itteration_track = 0

    def filter_true_positives(self, true_positives, entities, device):
        np_filt = np.delete(entities, true_positives, None)
        return torch.from_numpy(np_filt).to(device=device)

    def filter_false_positives(self, pred_rel, org_rel, candidates):
        cand = (pred_rel != org_rel).nonzero(as_tuple=True)[0]
        return torch.index_select(candidates, 0, cand)

    def score_filter(self, candidates, heads, tails, relations, rewards, device, head=True):
        if len(candidates) == 0:
            return torch.randint(0, self.kg.n_ent, (self.cache_dim,), device=device)
        else:
            rels = torch.ones(len(candidates), dtype=torch.int64, device=device) * relations.item()
            if head:
                hs = candidates
                ts = torch.ones(len(candidates), dtype=torch.int64, device=device) * tails.item()
            else:
                hs = torch.ones(len(candidates), dtype=torch.int64, device=device) * heads.item()
                ts = candidates
            rew = self.reward(hs, ts, rels)
            sel = (rew <= rewards).nonzero(as_tuple=True)[0]
            select = torch.index_select(candidates, 0, sel)

            if len(select) < self.cache_dim:
                rand_ent = torch.randint(0, self.kg.n_ent, (self.cache_dim - len(select),), device=device)
                cands = torch.cat((select, rand_ent), 0)
            else:
                sel_idx = torch.randint(0, len(select), (self.cache_dim,), device=device)
                cands = torch.index_select(select, 0, sel_idx)
            return cands

    def corrupt_batch(self, heads, tails, relations, n_neg=None):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if n_neg is None:
            n_neg = self.n_neg

        batch_size = len(heads)
        h_candidates, t_candidates = [], []
        start = self.temp_batch
        end = start + batch_size
        idx = np.arange(start, end, 1, dtype=int)
        self.temp_batch = end

        self.update_cache_ittr = int((self.n_facts / batch_size) * self.cache_dim * self.growth)

        if self.itteration_track == self.update_cache_ittr:
            self.update_cache()
        self.itteration_track += 1

        if self.temp_batch == self.kg.n_facts:
            self.temp_batch = 0

        for h_idx, t_idx in zip(self.head_idx[idx], self.tail_idx[idx]):
            randint = np.random.randint(low=0, high=self.cache_dim, size=(n_neg,))
            h_cand = self.head_cache[h_idx, randint]
            t_cand = self.tail_cache[t_idx, randint]
            h_candidates.append(torch.from_numpy(h_cand))
            t_candidates.append(torch.from_numpy(t_cand))

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

        neg_heads = torch.where(selection == ones, n_h, n_heads)
        neg_tails = torch.where(selection == ones, n_tails, n_t)

        return neg_heads, neg_tails
