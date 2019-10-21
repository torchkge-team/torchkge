# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
aboschin@enst.fr
"""
from torch import tensor, bernoulli, randint, ones, rand, cat
from torch.utils.data import DataLoader

from torchkge.utils import get_bern_probs, fill_in_dicts
from torchkge.exceptions import NotYetImplementedError


class NegativeSampler:
    """This is an interface for negative samplers in general.

    Parameters
    ----------
    kg: torchkge.data.KnowledgeGraph.KnowledgeGraph
        Main knowledge graph (usually training one).
    kg_val: torchkge.data.KnowledgeGraph.KnowledgeGraph (optional)
        Validation knowledge graph.
    kg_test: torchkge.data.KnowledgeGraph.KnowledgeGraph (optional)
        Test knowledge graph.

    Attributes
    ----------
    kg: torchkge.data.KnowledgeGraph.KnowledgeGraph
        Main knowledge graph (usually training one).
    kg_val: torchkge.data.KnowledgeGraph.KnowledgeGraph (optional)
        Validation knowledge graph.
    kg_test: torchkge.data.KnowledgeGraph.KnowledgeGraph (optional)
        Test knowledge graph.
    n_ent: int
        Number of entities in the entire knowledge graph. This is the same in `kg`, `kg_val`\
         and `kg_test`.
    n_facts: int
        Number of triplets in `kg`.
    n_facts_val: in
        Number of triplets in `kg_val`.
    n_facts_test: int
        Number of triples in `kg_test`.

    """
    def __init__(self, kg, kg_val=None, kg_test=None):
        self.kg = kg
        self.n_ent = kg.n_ent
        self.n_facts = kg.n_facts

        self.kg_val = kg_val
        self.kg_test = kg_test

        if kg_val is None:
            self.n_facts_val = 0
        else:
            self.n_facts_val = kg_val.n_facts

        if kg_test is None:
            self.n_facts_test = 0
        else:
            self.n_facts_test = kg_test.n_facts

    def corrupt_batch(self, heads, tails, relations):
        raise NotYetImplementedError('NegativeSampler is just an interface, please consider using '
                                     'a child class where this is implemented.')

    def corrupt_kg(self, batch_size, use_cuda, which='main'):
        """Corrupt an entire knowledge graph using a dataloader and by calling `corrupt_batch`
        methods.

        Parameters
        ----------
        batch_size: int
            Size of the batches used in the dataloader.
        use_cuda: bool
            Indicate whether to use cuda or not
        which: str
            Indicate which graph should be corrupted. Possible values are :\
            * 'main': attribute self.kg is corrupted
            * 'train': attribute self.kg is corrupted
            * 'val': attribute self.kg_val is corrupted. In this case this attribute should have\
            been initialized.
            * 'test': attribute self.kg_test is corrupted. In this case this attribute should have\
            been initialized.

        Returns
        -------
        neg_heads: torch.Tensor, dtype = long, shape = (n_facts)
            Tensor containing the integer key of negatively sampled heads of the relations\
            in the graph designated by `which`.
        neg_tails: torch.Tensor, dtype = long, shape = (n_facts)
            Tensor containing the integer key of negatively sampled tails of the relations\
            in the graph designated by `which`.
        """
        assert which in ['main', 'train', 'test', 'val']
        if which == 'val':
            assert self.n_facts_val > 0
        if which == 'test':
            assert self.n_facts_test > 0

        if which == 'val':
            dataloader = DataLoader(self.kg_val, batch_size=batch_size, shuffle=False,
                                    pin_memory=use_cuda)
        elif which == 'test':
            dataloader = DataLoader(self.kg_test, batch_size=batch_size, shuffle=False,
                                    pin_memory=use_cuda)
        else:
            dataloader = DataLoader(self.kg, batch_size=batch_size, shuffle=False,
                                    pin_memory=use_cuda)

        corr_heads, corr_tails = [], []

        for i, batch in enumerate(dataloader):

            heads, tails, rels = batch[0], batch[1], batch[2]
            if heads.is_pinned():
                heads, tails, rels = heads.cuda(), tails.cuda(), rels.cuda()

            neg_heads, neg_tails = self.corrupt_batch(heads, tails, rels)

            corr_heads.append(neg_heads)
            corr_tails.append(neg_tails)

        if use_cuda:
            return cat(corr_heads).long().cpu(), cat(corr_tails).long().cpu()
        else:
            return cat(corr_heads).long(), cat(corr_tails).long()


class UniformNegativeSampler(NegativeSampler):
    """Uniform negative sampler as presented in 2013 paper by Bordes et al.. Either the head or\
    the tail of a triplet is replaced by another entity at random. The choice of head/tail is\
    uniform.

    References
    ----------
    * Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko.
      Translating Embeddings for Modeling Multi-relational Data.
      In Advances in Neural Information Processing Systems 26, pages 2787–2795, 2013.
      https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data

    Parameters
    ----------
    kg: torchkge.data.KnowledgeGraph.KnowledgeGraph
        Main knowledge graph (usually training one).
    kg_val: torchkge.data.KnowledgeGraph.KnowledgeGraph (optional)
        Validation knowledge graph.
    kg_test: torchkge.data.KnowledgeGraph.KnowledgeGraph (optional)
        Test knowledge graph.

    Attributes
    ----------
    kg: torchkge.data.KnowledgeGraph.KnowledgeGraph
        Main knowledge graph (usually training one).
    kg_val: torchkge.data.KnowledgeGraph.KnowledgeGraph (optional)
        Validation knowledge graph.
    kg_test: torchkge.data.KnowledgeGraph.KnowledgeGraph (optional)
        Test knowledge graph.
    n_ent: int
        Number of entities in the entire knowledge graph. This is the same in `kg`, `kg_val`\
         and `kg_test`.
    n_facts: int
        Number of triplets in `kg`.
    n_facts_val: in
        Number of triplets in `kg_val`.
    n_facts_test: int
        Number of triples in `kg_test`.

    """
    def __init__(self, kg, kg_val=None, kg_test=None):
        super().__init__(kg, kg_val, kg_test)

    def corrupt_batch(self, heads, tails, relations=None):
        """For each golden triplet, produce a corrupted one not different from any other golden\
        triplet. If `heads` and `tails` are cuda objects , then the returned tensors are on the GPU.

        Parameters
        ----------
        heads: torch.Tensor, dtype = long, shape = (batch_size)
            Tensor containing the integer key of heads of the relations in the current batch.
        tails: torch.Tensor, dtype = long, shape = (batch_size)
            Tensor containing the integer key of tails of the relations in the current batch.
        relations: torch.Tensor, dtype = long, shape = (batch_size)
            Tensor containing the integer key of relations in the current batch. This is optional\
            here and mainly present because of the interface with other NegativeSampler objects.

        Returns
        -------
        neg_heads: torch.Tensor, dtype = long, shape = (batch_size)
            Tensor containing the integer key of negatively sampled heads of the relations\
            in the current batch.
        neg_tails: torch.Tensor, dtype = long, shape = (batch_size)
            Tensor containing the integer key of negatively sampled tails of the relations\
            in the current batch.
        """
        use_cuda = heads.is_cuda
        assert (use_cuda == tails.is_cuda)
        if use_cuda:
            device = 'cuda'
        else:
            device = 'cpu'

        batch_size = heads.shape[0]
        neg_heads, neg_tails = heads.clone(), tails.clone()

        # Randomly choose which samples will have head/tail corrupted
        mask = bernoulli(ones(size=(batch_size,), device=device) / 2).double()
        n_heads_corrupted = int(mask.sum().item())
        neg_heads[mask == 1] = randint(1, self.n_ent, (n_heads_corrupted,), device=device)
        neg_tails[mask == 0] = randint(1, self.n_ent, (batch_size - n_heads_corrupted,),
                                       device=device)

        return neg_heads.long(), neg_tails.long()


class BernoulliNegativeSampler(NegativeSampler):
    """Bernoulli negative sampler as presented in 2014 paper by Wang et al.. Either the head or\
    the tail of a triplet is replaced by another entity at random. The choice of head/tail is done\
    using probabilities taking into account profiles of the relations. See the paper for more\
    details.

    References
    ----------
    * Zhen Wang, Jianwen Zhang, Jianlin Feng, and Zheng Chen.
      Knowledge Graph Embedding by Translating on Hyperplanes.
      In Twenty-Eighth AAAI Conference on Artificial Intelligence, June 2014.
      https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8531

    Parameters
    ----------
    kg: torchkge.data.KnowledgeGraph.KnowledgeGraph
        Main knowledge graph (usually training one).
    kg_val: torchkge.data.KnowledgeGraph.KnowledgeGraph (optional)
        Validation knowledge graph.
    kg_test: torchkge.data.KnowledgeGraph.KnowledgeGraph (optional)
        Test knowledge graph.

    Attributes
    ----------
    kg: torchkge.data.KnowledgeGraph.KnowledgeGraph
        Main knowledge graph (usually training one).
    kg_val: torchkge.data.KnowledgeGraph.KnowledgeGraph (optional)
        Validation knowledge graph.
    kg_test: torchkge.data.KnowledgeGraph.KnowledgeGraph (optional)
        Test knowledge graph.
    n_ent: int
        Number of entities in the entire knowledge graph. This is the same in `kg`, `kg_val`\
         and `kg_test`.
    n_facts: int
        Number of triplets in `kg`.
    n_facts_val: in
        Number of triplets in `kg_val`.
    n_facts_test: int
        Number of triples in `kg_test`.
    bern_probs: torch.Tensor, dtype = float, shape = (kg.n_rel)
        Bernoulli sampling probabilities. See paper for more details.

    """
    def __init__(self, kg, kg_val=None, kg_test=None):
        super().__init__(kg, kg_val, kg_test)
        self.bern_probs = self.evaluate_probabilities()

    def evaluate_probabilities(self):
        """Evaluate the Bernoulli probabilities for negative sampling as in the TransH original\
        paper by Wang et al. (2014) https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8531.\
        Currently it is done using a pandas DataFrame. This should change as soon as the authors\
        find an efficient way to group-by in torch. TODO

        """
        bern_probs = get_bern_probs(self.kg.df)
        assert len(bern_probs) == self.kg.n_rel
        bern_probs = {self.kg.rel2ix[rel]: bern_probs[rel] for rel in bern_probs.keys()}
        return tensor([bern_probs[k] for k in sorted(bern_probs.keys())]).float()

    def corrupt_batch(self, heads, tails, relations):
        """For each golden triplet, produce a corrupted one different from any other golden\
        triplet. If `heads` and `tails` are cuda objects , then the returned tensors are on the GPU.

        Parameters
        ----------
        heads: torch.Tensor, dtype = long, shape = (batch_size)
            Tensor containing the integer key of heads of the relations in the current batch.
        tails: torch.Tensor, dtype = long, shape = (batch_size)
            Tensor containing the integer key of tails of the relations in the current batch.
        relations: torch.Tensor, dtype = long, shape = (batch_size)
            Tensor containing the integer key of relations in the current batch.

        Returns
        -------
        neg_heads: torch.Tensor, dtype = long, shape = (batch_size)
            Tensor containing the integer key of negatively sampled heads of the relations\
            in the current batch.
        neg_tails: torch.Tensor, dtype = long, shape = (batch_size)
            Tensor containing the integer key of negatively sampled tails of the relations\
            in the current batch.
        """
        use_cuda = heads.is_cuda
        assert (use_cuda == tails.is_cuda)
        if use_cuda:
            device = 'cuda'
        else:
            device = 'cpu'

        batch_size = heads.shape[0]
        neg_heads, neg_tails = heads.clone(), tails.clone()

        # Randomly choose which samples will have head/tail corrupted
        mask = bernoulli(self.bern_probs[relations]).double()
        n_heads_corrupted = int(mask.sum().item())
        neg_heads[mask == 1] = randint(1, self.n_ent, (n_heads_corrupted,), device=device)
        neg_tails[mask == 0] = randint(1, self.n_ent, (batch_size - n_heads_corrupted,),
                                       device=device)

        return neg_heads.long(), neg_tails.long()


class PositionalNegativeSampler(BernoulliNegativeSampler):
    """Positional negative sampler as presented in 2011 paper by Socher et al.. Either the head or\
    the tail of a triplet is replaced by another entity chosen among entities that have already\
    appeared at the same place in a triplet (involving the same relation). It is not clear in the\
    paper how the choice of head/tail is done. We chose to use Bernoulli sampling as in 2014 paper\
    by Wang et al. as we believe it serves the same purpose as the original paper.

    References
    ----------
    * Richard Socher, Danqi Chen, Christopher D Manning, and Andrew Ng.
      Reasoning With Neural Tensor Networks for Knowledge Base Completion.
      In Advances in Neural Information Processing Systems 26, pages 926–934., 2013.
      https://nlp.stanford.edu/pubs/SocherChenManningNg_NIPS2013.pdf
    * Zhen Wang, Jianwen Zhang, Jianlin Feng, and Zheng Chen.
      Knowledge Graph Embedding by Translating on Hyperplanes.
      In Twenty-Eighth AAAI Conference on Artificial Intelligence, June 2014.
      https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8531

    Parameters
    ----------
    kg: torchkge.data.KnowledgeGraph.KnowledgeGraph
        Main knowledge graph (usually training one).
    kg_val: torchkge.data.KnowledgeGraph.KnowledgeGraph (optional)
        Validation knowledge graph.
    kg_test: torchkge.data.KnowledgeGraph.KnowledgeGraph (optional)
        Test knowledge graph.

    Attributes
    ----------
    kg: torchkge.data.KnowledgeGraph.KnowledgeGraph
        Main knowledge graph (usually training one).
    kg_val: torchkge.data.KnowledgeGraph.KnowledgeGraph (optional)
        Validation knowledge graph.
    kg_test: torchkge.data.KnowledgeGraph.KnowledgeGraph (optional)
        Test knowledge graph.
    n_ent: int
        Number of entities in the entire knowledge graph. This is the same in `kg`, `kg_val`\
         and `kg_test`.
    n_facts: int
        Number of triplets in `kg`.
    n_facts_val: in
        Number of triplets in `kg_val`.
    n_facts_test: int
        Number of triples in `kg_test`.
    bern_probs: torch.Tensor, dtype = float, shape = (kg.n_rel)
        Bernoulli sampling probabilities. See paper for more details.
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
        super().__init__(kg, kg_val, kg_test)
        self.possible_heads, self.possible_tails, self.n_poss_heads, self.n_poss_tails = self.find_possibilities()

    def find_possibilities(self):
        """For each relation of the knowledge graph (and possibly the validation graph but not the\
        test graph) find all the possible heads and tails in the sense of Wang et al., e.g. all\
        entities that occupy once this position in another triplet.

        Returns
        -------
        possible_heads: dict
            keys : relation index, values : list of possible heads
        possible tails: dict
            keys : relation index, values : list of possible tails
        n_poss_heads: torch.Tensor, dtype = long, shape = (n_relations)
            Number of possible heads for each relation.
        n_poss_tails: torch.Tensor, dtype = long, shape = (n_relations)
            Number of possible tails for each relation.

        """
        possible_heads, possible_tails = fill_in_dicts(self.kg)

        if self.n_facts_val > 0:
            possible_heads, possible_tails = fill_in_dicts(self.kg_val,
                                                           possible_heads, possible_tails)

        n_poss_heads = []
        n_poss_tails = []

        for i in range(self.kg.n_rel):
            n_poss_heads.append(len(possible_heads[i]))
            n_poss_tails.append(len(possible_tails[i]))
            possible_heads[i] = list(possible_heads[i])
            possible_tails[i] = list(possible_tails[i])

        n_poss_heads = tensor(n_poss_heads)
        n_poss_tails = tensor(n_poss_tails)

        return dict(possible_heads), dict(possible_tails), n_poss_heads, n_poss_tails

    def corrupt_batch(self, heads, tails, relations):
        """For each golden triplet, produce a corrupted one not different from any other golden\
        triplet. If `heads` and `tails` are cuda objects , then the returned tensors are on the GPU.

        Parameters
        ----------
        heads: torch.Tensor, dtype = long, shape = (batch_size)
            Tensor containing the integer key of heads of the relations in the current batch.
        tails: torch.Tensor, dtype = long, shape = (batch_size)
            Tensor containing the integer key of tails of the relations in the current batch.
        relations: torch.Tensor, dtype = long, shape = (batch_size)
            Tensor containing the integer key of relations in the current batch. This is optional\
            here and mainly present because of the interface with other NegativeSampler objects.

        Returns
        -------
        neg_heads: torch.Tensor, dtype = long, shape = (batch_size)
            Tensor containing the integer key of negatively sampled heads of the relations\
            in the current batch.
        neg_tails: torch.Tensor, dtype = long, shape = (batch_size)
            Tensor containing the integer key of negatively sampled tails of the relations\
            in the current batch.
        """
        use_cuda = heads.is_cuda
        assert (use_cuda == tails.is_cuda)
        if use_cuda:
            device = 'cuda'
        else:
            device = 'cpu'

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
        choice_heads = (n_poss_heads.float() * rand((n_heads_corrupted,))).floor().long()
        choice_tails = (
                n_poss_tails.float() * rand((batch_size - n_heads_corrupted,))).floor().long()

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
