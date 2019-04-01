# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
armand.boschin@telecom-paristech.fr
"""

from torch import Tensor, randint, bernoulli
from torch.utils.data import Dataset


class KnowledgeGraph(Dataset):
    def __init__(self, df, ent2ix=None, rel2ix=None):
        """
        :param df: Pandas data frame containing three columns from, to and rel
        :return: KnowledgeGraph object
        """
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
        """
        Move the KnowledgeGraph object to cuda
        """
        self.use_cuda = True
        self.head_idx = self.head_idx.cuda()
        self.tail_idx = self.tail_idx.cuda()
        self.relations = self.relations.cuda()

    def corrupt_batch(self, heads, tails):
        """
        For each golden triplet, produce a corrupted one not different from any other golden triplet
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
