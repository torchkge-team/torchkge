# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
armand.boschin@telecom-paristech.fr
"""

from torch import Tensor, randint, bernoulli
from torch.utils.data import Dataset


class KnowledgeGraph(Dataset):
    """Knowledge graph representation.

        Parameters
        ----------
        df : pandas Dataframe
            Data frame containing three columns [from, to, rel].
        ent2ix : dict, optional
            Dictionary mapping entity labels to their integer key.
        rel2ix : dict, optional
            Dictionary mapping relation labels to their integer key.

        Attributes
        ----------
        ent2ix : dict
            Dictionary mapping entity labels to their integer key.
        rel2ix : dict
            Dictionary mapping relation labels to their integer key.
        n_ent : int
            Number of distinct entities in the data set.
        n_rel : int
            Number of distinct entities in the data set.
        n_sample : int
            Number of samples in the data set. A sample is a fact : a triplet (h, r, l).
        head_idx : torch tensor, dtype = long, shape = (n_sample)
            List of the int key of heads for each sample (fact).
        tail_idx : torch tensor, dtype = long, shape = (n_sample)
            List of the int key of tails for each sample (facts).
        relations : torch tensor, dtype = long, shape = (n_sample)
            List of the int key of relations for each sample (facts).
        use_cuda : bool
            Indicates if current object has been moved to cuda.

    """

    def __init__(self, df, ent2ix=None, rel2ix=None):
        if ent2ix is None:
            tmp = list(set(df['from'].unique()).union(set(df['to'].unique())))
            self.ent2ix = {ent: i for i, ent in enumerate(tmp)}
            del tmp
        else:
            self.ent2ix = ent2ix

        if rel2ix is None:
            tmp = list(df['rel'].unique())
            self.rel2ix = {rel: i for i, rel in enumerate(tmp)}
            del tmp
        else:
            self.rel2ix = rel2ix

        self.n_ent = max(self.ent2ix.values()) + 1
        self.n_rel = max(self.rel2ix.values()) + 1
        self.n_sample = len(df)

        self.head_idx = Tensor(df['from'].map(self.ent2ix).values).long()
        self.tail_idx = Tensor(df['to'].map(self.ent2ix).values).long()
        self.relations = Tensor(df['rel'].map(self.rel2ix).values).long()

        self.use_cuda = False

    def __len__(self):
        return self.n_sample

    def __getitem__(self, item):
        return self.head_idx[item].item(), self.tail_idx[item].item(), self.relations[item].item()

    def cuda(self):
        """Move the KnowledgeGraph object to cuda
        """
        self.use_cuda = True
        self.head_idx = self.head_idx.cuda()
        self.tail_idx = self.tail_idx.cuda()
        self.relations = self.relations.cuda()

    def corrupt_batch(self, heads, tails):
        """For each golden triplet, produce a corrupted one not different from any other golden triplet.

        Parameters
        ----------
        heads : torch tensor, dtype = long, shape = (batch_size)
            Tensor containing the integer key of heads of the relations in the current batch.
        tails : torch tensor, dtype = long, shape = (batch_size)
            Tensor containing the integer key of tails of the relations in the current batch.

        Returns
        -------
        neg_heads : torch tensor, dtype = long, shape = (batch_size)
            Tensor containing the integer key of negatively sampled heads of the relations \
            in the current batch.
        neg_tails : torch tensor, dtype = long, shape = (batch_size)
            Tensor containing the integer key of negatively sampled tails of the relations \
            in the current batch.
        """
        batch_size = heads.shape[0]

        # TODO : implement smarter corruption (cf TransH paper)
        neg_heads, neg_tails = heads.clone(), tails.clone()

        # Randomly choose which samples will have head/tail corrupted
        mask = bernoulli(Tensor(size=(batch_size,)).uniform_(0, 1)).double()
        n_heads_corrupted = int(mask.sum().item())

        # Corrupt the samples
        if self.use_cuda:
            neg_heads[mask == 1] = randint(1, self.n_ent, (n_heads_corrupted,), device='cuda')
            neg_tails[mask == 0] = randint(1, self.n_ent, (batch_size - n_heads_corrupted,),
                                           device='cuda')

        else:
            neg_heads[mask == 1] = randint(1, self.n_ent, (n_heads_corrupted,))
            neg_tails[mask == 0] = randint(1, self.n_ent, (batch_size - n_heads_corrupted,))

        return neg_heads.long(), neg_tails.long()
