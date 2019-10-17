# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
aboschin@enst.fr
"""

from torch import empty, tensor, cat
from torch.utils.data import Dataset

from torchkge.utils import get_dictionaries
from torchkge.exceptions import SizeMismatchError

from tqdm import tqdm
from collections import defaultdict


class SmallKG(Dataset):
    """Minimalist version of a knowledge graph. Built with tensors of heads, tails and relations.

    """
    def __init__(self, heads, tails, relations):
        assert heads.shape == tails.shape == relations.shape
        self.heads = heads
        self.tails = tails
        self.relations = relations
        self.length = heads.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return self.heads[item].item(), self.tails[item].item(), self.relations[item].item()


class KnowledgeGraph(Dataset):
    """Knowledge graph representation.

        Parameters
        ----------
        df: pandas.DataFrame
            Data frame containing three columns [from, to, rel].
        ent2ix: dict, optional
            Dictionary mapping entity labels to their integer key.
        rel2ix: dict, optional
            Dictionary mapping relation labels to their integer key.

        Attributes
        ----------
        ent2ix: dict
            Dictionary mapping entity labels to their integer key.
        rel2ix: dict
            Dictionary mapping relation labels to their integer key.
        n_ent: int
            Number of distinct entities in the data set.
        n_rel: int
            Number of distinct entities in the data set.
        n_facts: int
            Number of samples in the data set. A sample is a fact: a triplet (h, r, l).
        head_idx: torch.Tensor, dtype = long, shape = (n_facts)
            List of the int key of heads for each fact.
        tail_idx: torch.Tensor, dtype = long, shape = (n_facts)
            List of the int key of tails for each fact.
        relations: torch.Tensor, dtype = long, shape = (n_facts)
            List of the int key of relations for each fact.

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
            self.df = df
            self.n_facts = len(df)
            self.head_idx = tensor(df['from'].map(self.ent2ix).values).long()
            self.tail_idx = tensor(df['to'].map(self.ent2ix).values).long()
            self.relations = tensor(df['rel'].map(self.rel2ix).values).long()
        else:
            assert kg is not None
            self.df = kg['df']
            self.n_facts = kg['heads'].shape[0]
            self.head_idx = kg['heads']
            self.tail_idx = kg['tails']
            self.relations = kg['relations']

        if dict_of_heads is None or dict_of_tails is None:
            self.dict_of_heads = defaultdict(set)
            self.dict_of_tails = defaultdict(set)
            print('Evaluating dictionaries of possible heads and tails for relations.')
            self.evaluate_dicts()

        else:
            self.dict_of_heads = dict_of_heads
            self.dict_of_tails = dict_of_tails

    def __len__(self):
        return self.n_facts

    def __getitem__(self, item):
        return self.head_idx[item].item(), self.tail_idx[item].item(), self.relations[item].item()

    def split_kg(self, share=0.8, sizes=None, validation=False):
        """Split the knowledge graph into train and test.

        Parameters
        ----------
        share: float
            Percentage to allocate to train set.
        sizes: tuple
            Tuple of ints of length 2 or 3. If len(sizes) == 2, then the first sizes[0] values of\
            the knowledge graph will be used as training set and the rest as test set.\
            If len(sizes) == 3, the first sizes[0] values of the knowledge graph will be used as\
            training set, the following sizes[1] as validation set and the last sizes[2] as testing\
            set.
        validation: bool
            Indicate if a validation set should be produced along with train and test sets.

        Returns
        -------
        train_kg: torchkge.data.KnowledgeGraph
        val_kg: torchkge.data.KnowledgeGraph (optional)
        test_kg: torchkge.data.KnowledgeGraph

        """

        if sizes is not None:
            try:
                if len(sizes) == 3:
                    assert (sizes[0] + sizes[1] + sizes[2] == self.n_facts)
                elif len(sizes) == 2:
                    assert (sizes[0] + sizes[1] == self.n_facts)
                else:
                    raise SizeMismatchError('Tuple `sizes` should be of length 2 or 3.')
            except AssertionError:
                raise SizeMismatchError('Tuple `sizes` should sum up to the number of facts in the '
                                        'knowledge graph.')

        if ((sizes is not None) and (len(sizes) == 3)) or ((sizes is None) and validation):
            if (sizes is None) and validation:
                samp = empty(self.head_idx.shape).uniform_()
                mask_tr = (samp < share)
                mask_val = (samp > share) & (samp < (1 + share) / 2)
                mask_te = ~(mask_tr | mask_val)
            else:
                mask_tr = cat([tensor([1 for _ in range(sizes[0])]),
                               tensor([0 for _ in range(sizes[1] + sizes[2])])]).bool()
                mask_val = cat([tensor([0 for _ in range(sizes[0])]),
                                tensor([1 for _ in range(sizes[1])]),
                                tensor([0 for _ in range(sizes[2])])]).bool()
                mask_te = ~(mask_tr | mask_val)

            return KnowledgeGraph(
                kg={'heads': self.head_idx[mask_tr], 'tails': self.tail_idx[mask_tr],
                    'relations': self.relations[mask_tr], 'df': self.df},
                ent2ix=self.ent2ix, rel2ix=self.rel2ix, dict_of_heads=self.dict_of_heads,
                dict_of_tails=self.dict_of_tails), KnowledgeGraph(
                kg={'heads': self.head_idx[mask_val], 'tails': self.tail_idx[mask_val],
                    'relations': self.relations[mask_val], 'df': self.df},
                ent2ix=self.ent2ix, rel2ix=self.rel2ix, dict_of_heads=self.dict_of_heads,
                dict_of_tails=self.dict_of_tails), KnowledgeGraph(
                kg={'heads': self.head_idx[mask_te], 'tails': self.tail_idx[mask_te],
                    'relations': self.relations[mask_te], 'df': self.df},
                ent2ix=self.ent2ix, rel2ix=self.rel2ix, dict_of_heads=self.dict_of_heads,
                dict_of_tails=self.dict_of_tails)
        else:
            assert (((sizes is not None) and len(sizes) == 2) or
                    ((sizes is None) and not validation))
            if sizes is None:
                mask_tr = (empty(self.head_idx.shape).uniform_() < share)
            else:
                mask_tr = cat([tensor([1 for _ in range(sizes[0])]),
                               tensor([0 for _ in range(sizes[1])])]).bool()
            return KnowledgeGraph(
                kg={'heads': self.head_idx[mask_tr], 'tails': self.tail_idx[mask_tr],
                    'relations': self.relations[mask_tr], 'df': self.df},
                ent2ix=self.ent2ix, rel2ix=self.rel2ix, dict_of_heads=self.dict_of_heads,
                dict_of_tails=self.dict_of_tails), KnowledgeGraph(
                kg={'heads': self.head_idx[~mask_tr], 'tails': self.tail_idx[~mask_tr],
                    'relations': self.relations[~mask_tr], 'df': self.df},
                ent2ix=self.ent2ix, rel2ix=self.rel2ix, dict_of_heads=self.dict_of_heads,
                dict_of_tails=self.dict_of_tails)

    def evaluate_dicts(self):
        """Evaluate dicts of possible alternatives to an entity in a fact that still gives a true\
        fact in the entire knowledge graph.

        """
        for i in tqdm(range(self.n_facts)):
            self.dict_of_heads[(self.tail_idx[i].item(),
                                self.relations[i].item())].add(self.head_idx[i].item())
            self.dict_of_tails[(self.head_idx[i].item(),
                                self.relations[i].item())].add(self.tail_idx[i].item())
