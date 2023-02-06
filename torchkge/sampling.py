# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""

from collections import defaultdict

from torch import tensor, bernoulli, randint, ones, rand, cat

from torchkge.exceptions import NotYetImplementedError
from torchkge.utils.data import DataLoader
from torchkge.utils.operations import get_bernoulli_probs


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
        """For each true triplet, produce a corrupted one assumed to be different
        from any other true triplet. If `heads` and `tails` are cuda objects,
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
        self.bern_probs = self.bern_probs.to(device)
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


class BernoulliRelationNegativeSampler(NegativeSampler):

    def __init__(self, kg, kg_val=None, kg_test=None, n_neg=1, rel_share=.33):
        super().__init__(kg, kg_val, kg_test, n_neg)
        self.bern_probs = self.evaluate_probabilities()
        self.rel_share = rel_share

    def evaluate_probabilities(self):
        bern_probs = get_bernoulli_probs(self.kg)

        tmp = []
        for i in range(self.kg.n_rel):
            if i in bern_probs.keys():
                tmp.append(bern_probs[i])
            else:
                tmp.append(0.5)

        return tensor(tmp).float()

    def corrupt_batch(self, heads, tails, relations, n_neg=None):

        device = heads.device
        assert (device == tails.device)

        batch_size = heads.shape[0]
        neg_heads = heads.clone().detach()
        neg_tails = tails.clone().detach()
        neg_rels = relations.clone().detach()

        mask1 = bernoulli(self.rel_share * ones(batch_size)).double()  # if 1 then entities are corrupted

        neg_rels[mask1 == 0] = randint(1, self.kg.n_rel, (int(batch_size - (mask1.sum().item())),), device=device)

        # Randomly choose which samples will have head/tail corrupted
        mask2 = ones(len(mask1))
        mask2[mask1 == 0] = 0.

        mask2[mask2 == 1] = bernoulli(self.bern_probs[relations[mask1 == 1]])

        mask2 = mask2.double()

        n_h_cor = int(mask2.sum().item())
        neg_heads[(mask1 == 1) & (mask2 == 1)] = randint(1, self.n_ent, (n_h_cor,), device=device)
        neg_tails[(mask1 == 1) & (mask2 == 0)] = randint(1, self.n_ent, (int(mask1.sum().item()) - n_h_cor,),
                                                         device=device)

        return neg_heads.long(), neg_tails.long(), neg_rels.long()


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
