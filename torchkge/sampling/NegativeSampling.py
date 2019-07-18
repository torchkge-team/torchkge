# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
aboschin@enst.fr
"""
from collections import defaultdict
from tqdm import tqdm

from torch import tensor, bernoulli, randint, ones, rand, cat
from torch.utils.data import DataLoader

from torchkge.utils import get_bern_probs
from torchkge.exceptions import NotYetImplementedError


class NegativeSampler(object):
    def __init__(self, kg, kg_val=None, kg_test=None):
        self.kg = kg
        self.n_ent = kg.n_ent
        self.n_sample = kg.n_sample

        self.kg_val = kg_val
        self.kg_test = kg_test

        if kg_val is None:
            self.n_sample_val = 0
        else:
            self.n_sample_val = kg_val.n_sample

        if kg_test is None:
            self.n_sample_test = 0
        else:
            self.n_sample_test = kg_test.n_sample

    def corrupt_batch(self, heads, tails, relations):
        raise NotYetImplementedError('This should be implemented...')

    def corrupt_kg(self, batch_size, use_cuda, val=False, test=False):
        assert not (val and test)
        if val:
            assert self.n_sample_val > 0
        if test:
            assert self.n_sample_test > 0

        if val:
            dataloader = DataLoader(self.kg_val, batch_size=batch_size, shuffle=False,
                                    pin_memory=use_cuda)
        elif test:
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

        return cat(corr_heads).long(), cat(corr_tails).long()


class UniformNegativeSampler(NegativeSampler):
    def __init__(self, kg, kg_val=None, kg_test=None):
        super().__init__(kg, kg_val, kg_test)

    def corrupt_batch(self, heads, tails, relations=None):
        """For each golden triplet, produce a corrupted one not different from any other golden\
        triplet.

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
    def __init__(self, kg, kg_val=None, kg_test=None):
        super().__init__(kg, kg_val, kg_test)
        self.bern_probs = self.evaluate_probabilities()

    def evaluate_probabilities(self):
        """Evaluate the Bernoulli probabilities for negative sampling as in the TransH original\
        paper by Wang et al. (2014) https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8531.\
        Currently it is done using a pandas DataFrame. This should change as soon as the authors\
        find an efficient way to grouby in torch. TODO

        """
        bern_probs = get_bern_probs(self.kg.df)
        assert len(bern_probs) == self.kg.n_rel
        bern_probs = {self.kg.rel2ix[rel]: bern_probs[rel] for rel in bern_probs.keys()}
        return tensor([bern_probs[k] for k in sorted(bern_probs.keys())]).float()

    def corrupt_batch(self, heads, tails, relations):
        """For each golden triplet, produce a corrupted one different from any other golden\
        triplet.

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

    def __init__(self, kg, kg_val=None, kg_test=None):
        super().__init__(kg, kg_val, kg_test)
        self.possible_heads, self.possible_tails, self.n_poss_heads, self.n_poss_tails = self.find_possibilities()

    def fill_in_dicts(self, kg, possible_heads=None, possible_tails=None):
        if possible_heads is None:
            possible_heads = defaultdict(set)
        if possible_tails is None:
            possible_tails = defaultdict(set)

        for i in tqdm(range(kg.n_sample)):
            possible_heads[kg.relations[i].item()].add(kg.head_idx[i].item())
            possible_tails[kg.relations[i].item()].add(kg.tail_idx[i].item())

        return possible_heads, possible_tails

    def find_possibilities(self):
        possible_heads, possible_tails = self.fill_in_dicts(self.kg)

        if self.n_sample_val > 0:
            possible_heads, possible_tails = self.fill_in_dicts(self.kg_val,
                                                                possible_heads, possible_tails)

        if self.n_sample_test > 0:
            possible_heads, possible_tails = self.fill_in_dicts(self.kg_test,
                                                                possible_heads, possible_tails)

        possible_heads = dict(possible_heads)
        possible_tails = dict(possible_tails)

        n_poss_heads = []
        n_poss_tails = []

        for i in range(self.kg.n_rel):
            n_poss_heads.append(len(possible_heads[i]))
            n_poss_tails.append(len(possible_tails[i]))
            possible_heads[i] = list(possible_heads[i])
            possible_tails[i] = list(possible_tails[i])

        n_poss_heads = tensor(n_poss_heads)
        n_poss_tails = tensor(n_poss_tails)

        return possible_heads, possible_tails, n_poss_heads, n_poss_tails

    def corrupt_batch(self, heads, tails, relations):
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
            corr.append(self.possible_heads[r][choice_heads[i].item()])
        neg_heads[mask == 1] = tensor(corr, device=device)

        corr = []
        rels = relations[mask == 0]
        for i in range(batch_size - n_heads_corrupted):
            r = rels[i].item()
            corr.append(self.possible_tails[r][choice_tails[i].item()])
        neg_tails[mask == 0] = tensor(corr, device=device)

        return neg_heads.long(), neg_tails.long()
