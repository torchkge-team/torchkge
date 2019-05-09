# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
aboschin@enst.fr
"""

from torch import empty, tensor, cat
from torch.utils.data import Dataset

from torchkge.data.utils import get_dictionaries

from tqdm import tqdm
from collections import defaultdict


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

    """

    def __init__(self, df=None, kg=None,
                 ent2ix=None, rel2ix=None, dict_of_heads=None, dict_of_tails=None):

        if ent2ix is None:
            self.ent2ix = get_dictionaries(df, ent=True)
        else:
            self.ent2ix = ent2ix

        if rel2ix is None:
            self.rel2ix = get_dictionaries(df, ent=False)
        else:
            self.rel2ix = rel2ix

        self.n_ent = max(self.ent2ix.values()) + 1
        self.n_rel = max(self.rel2ix.values()) + 1

        if df is not None:
            assert kg is None
            self.n_sample = len(df)
            self.head_idx = tensor(df['from'].map(self.ent2ix).values).long()
            self.tail_idx = tensor(df['to'].map(self.ent2ix).values).long()
            self.relations = tensor(df['rel'].map(self.rel2ix).values).long()
        else:
            assert kg is not None
            self.n_sample = kg['heads'].shape[0]
            self.head_idx = kg['heads']
            self.tail_idx = kg['tails']
            self.relations = kg['relations']

        if dict_of_heads is None or dict_of_tails is None:
            self.dict_of_heads = defaultdict(list)
            self.dict_of_tails = defaultdict(list)
            print('Evaluating dictionaries.')
            self.evaluate_dicts()

        else:
            self.dict_of_heads = dict_of_heads
            self.dict_of_tails = dict_of_tails

    def __len__(self):
        return self.n_sample

    def __getitem__(self, item):
        """
        if not self.list_evaluated:
            return self.head_idx[item].item(), self.tail_idx[item].item(), \
                   self.relations[item].item()
        else:
            return self.head_idx[item].item(), self.tail_idx[item].item(), \
                   self.relations[item].item(), self.list_of_heads[item], self.list_of_tails[item]
        """
        return self.head_idx[item].item(), self.tail_idx[item].item(), self.relations[item].item()

    def split_kg(self, share=0.8, train_size=None):
        """Split the knowledge graph into train and test.

        Parameters
        ----------
        share : float
            Percentage to allocate to train set.
        train_size : integer
            Length of the training set. If this is not None, the first values of the knowledge
            graph will be used as training set and the rest as test set.

        Returns
        -------
        train_kg : KnowledgeGraph
        test_kg : KnowledgeGraph
        """

        if train_size is None:
            mask = (empty(self.head_idx.shape).uniform_() < share)
        else:
            mask = cat([tensor([1 for _ in range(train_size)]),
                        tensor([0 for _ in range(self.head_idx.shape[0] - train_size)])])
            mask = mask.byte()

        train_kg = KnowledgeGraph(
            kg={'heads': self.head_idx[mask],
                'tails': self.tail_idx[mask],
                'relations': self.relations[mask]},
            ent2ix=self.ent2ix, rel2ix=self.rel2ix,
            dict_of_heads=self.dict_of_heads,
            dict_of_tails=self.dict_of_tails)

        test_kg = KnowledgeGraph(
            kg={'heads': self.head_idx[~mask],
                'tails': self.tail_idx[~mask],
                'relations': self.relations[~mask]},
            ent2ix=self.ent2ix, rel2ix=self.rel2ix,
            dict_of_heads=self.dict_of_heads,
            dict_of_tails=self.dict_of_tails)

        return train_kg, test_kg

    def evaluate_dicts(self):
        """Evaluate dicts of possible alternatives to an entity in a fact that still gives a true
        fact in the entire knowledge graph.

        """

        for i in tqdm(range(self.n_sample)):
            self.dict_of_heads[(self.tail_idx[i].item(),
                                self.relations[i].item())].extend([self.head_idx[i].item()])
            self.dict_of_tails[(self.head_idx[i].item(),
                                self.relations[i].item())].extend([self.tail_idx[i].item()])
