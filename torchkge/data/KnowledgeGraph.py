# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""

from torch import empty, tensor, cat, int64, Tensor, zeros_like, randperm
from torch.utils.data import Dataset

from torchkge.utils import get_dictionaries
from torchkge.exceptions import SizeMismatchError, WrongArgumentsError, SanityError, SplitabilityError

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
        head_idx: `torch.Tensor`, dtype = `torch.long`, shape: (n_facts)
            List of the int key of heads for each fact.
        tail_idx: `torch.Tensor`, dtype = `torch.long`, shape: (n_facts)
            List of the int key of tails for each fact.
        relations: `torch.Tensor`, dtype = `torch.long`, shape: (n_facts)
            List of the int key of relations for each fact.

    """

    def __init__(self, df=None, kg=None,
                 ent2ix=None, rel2ix=None, dict_of_heads=None, dict_of_tails=None):

        """
        :param df: `pandas.DataFrame`
        :param kg: dict
            keys should be exhaustively ('heads', 'tails', 'relations')
        :param ent2ix:
        :param rel2ix:
        :param dict_of_heads:
        :param dict_of_tails:
        """

        if df is None:
            if kg is None:
                raise WrongArgumentsError("Please provide at least one argument of `df` and kg`")
            else:
                try:
                    assert (type(kg) == dict) & ('heads' in kg.keys()) & ('tails' in kg.keys()) & \
                           ('relations' in kg.keys())
                except AssertionError:
                    raise WrongArgumentsError("Keys in the `kg` dict should contain `heads`, `tails`, `relations`.")
                try:
                    assert (rel2ix is not None) & (ent2ix is not None)
                except AssertionError:
                    raise WrongArgumentsError("Please provide the two dictionaries ent2ix and rel2ix if building from `kg`.")
        else:
            if kg is not None:
                raise WrongArgumentsError("`df` and kg` arguments should not both provided.")

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
            # build kg from a pandas dataframe
            self.n_facts = len(df)
            self.head_idx = tensor(df['from'].map(self.ent2ix).values).long()
            self.tail_idx = tensor(df['to'].map(self.ent2ix).values).long()
            self.relations = tensor(df['rel'].map(self.rel2ix).values).long()
        else:
            # build kg from another kg
            self.n_facts = kg['heads'].shape[0]
            self.head_idx = kg['heads']
            self.tail_idx = kg['tails']
            self.relations = kg['relations']

        if dict_of_heads is None or dict_of_tails is None:
            self.dict_of_heads = defaultdict(set)
            self.dict_of_tails = defaultdict(set)
            self.evaluate_dicts()

        else:
            self.dict_of_heads = dict_of_heads
            self.dict_of_tails = dict_of_tails
        try:
            self.sanity_check()
        except AssertionError:
            raise SanityError("Please check the sanity of arguments.")

    def __len__(self):
        return self.n_facts

    def __getitem__(self, item):
        return self.head_idx[item].item(), self.tail_idx[item].item(), self.relations[item].item()

    def sanity_check(self):
        assert (type(self.dict_of_heads) == defaultdict) & (type(self.dict_of_heads) == defaultdict)
        assert (type(self.ent2ix) == dict) & (type(self.rel2ix) == dict)
        assert (len(self.ent2ix) == self.n_ent) & (len(self.rel2ix) == self.n_rel)
        assert (type(self.head_idx) == Tensor) & (type(self.tail_idx) == Tensor) & (type(self.relations) == Tensor)
        assert (self.head_idx.dtype == int64) & (self.tail_idx.dtype == int64) & (self.relations.dtype == int64)
        assert (len(self.head_idx) == len(self.tail_idx) == len(self.relations))

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
        train_kg: `torchkge.data.KnowledgeGraph.KnowledgeGraph`
        val_kg: `torchkge.data.KnowledgeGraph.KnowledgeGraph`, optional
        test_kg: `torchkge.data.KnowledgeGraph.KnowledgeGraph`

        """

        if sizes is not None:
            try:
                if len(sizes) == 3:
                    try:
                        assert (sizes[0] + sizes[1] + sizes[2] == self.n_facts)
                    except AssertionError:
                        raise WrongArgumentsError('Sizes should sum to the number of facts.')
                elif len(sizes) == 2:
                    try:
                        assert (sizes[0] + sizes[1] == self.n_facts)
                    except AssertionError:
                        raise WrongArgumentsError('Sizes should sum to the number of facts.')
                else:
                    raise SizeMismatchError('Tuple `sizes` should be of length 2 or 3.')
            except AssertionError:
                raise SizeMismatchError('Tuple `sizes` should sum up to the number of facts in the '
                                        'knowledge graph.')
        else:
            assert share < 1

        _, counts = self.relations.unique(return_counts=True)

        if ((sizes is not None) and (len(sizes) == 3)) or ((sizes is None) and validation):
            # return training, validation and a testing graphs
            if (counts < 3).sum().item() > 0:
                raise SplitabilityError("Cannot split in three subsets and keep on fact of each relation in the three.")

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
                    'relations': self.relations[mask_tr]},
                ent2ix=self.ent2ix, rel2ix=self.rel2ix, dict_of_heads=self.dict_of_heads,
                dict_of_tails=self.dict_of_tails), KnowledgeGraph(
                kg={'heads': self.head_idx[mask_val], 'tails': self.tail_idx[mask_val],
                    'relations': self.relations[mask_val]},
                ent2ix=self.ent2ix, rel2ix=self.rel2ix, dict_of_heads=self.dict_of_heads,
                dict_of_tails=self.dict_of_tails), KnowledgeGraph(
                kg={'heads': self.head_idx[mask_te], 'tails': self.tail_idx[mask_te],
                    'relations': self.relations[mask_te]},
                ent2ix=self.ent2ix, rel2ix=self.rel2ix, dict_of_heads=self.dict_of_heads,
                dict_of_tails=self.dict_of_tails)
        else:
            # return training and testing graphs
            if (counts < 2).sum().item() > 0:
                raise SplitabilityError("Cannot split in two subsets and keep on fact of each relation in both.")
            assert (((sizes is not None) and len(sizes) == 2) or
                    ((sizes is None) and not validation))
            if sizes is None:
                mask_tr = (empty(self.head_idx.shape).uniform_() < share)
            else:
                mask_tr = cat([tensor([1 for _ in range(sizes[0])]),
                               tensor([0 for _ in range(sizes[1])])]).bool()
            return KnowledgeGraph(
                kg={'heads': self.head_idx[mask_tr], 'tails': self.tail_idx[mask_tr],
                    'relations': self.relations[mask_tr]},
                ent2ix=self.ent2ix, rel2ix=self.rel2ix, dict_of_heads=self.dict_of_heads,
                dict_of_tails=self.dict_of_tails), KnowledgeGraph(
                kg={'heads': self.head_idx[~mask_tr], 'tails': self.tail_idx[~mask_tr],
                    'relations': self.relations[~mask_tr]},
                ent2ix=self.ent2ix, rel2ix=self.rel2ix, dict_of_heads=self.dict_of_heads,
                dict_of_tails=self.dict_of_tails)

    def get_mask(self, share, validation=False):
        """ TODO: include the use of this function to split
        TODO: also include the fact that all entities should be at least once in the training set

        :param share:
        :param validation:
        :return:
        """
        mask = zeros_like(self.relations).bool()
        if validation:
            val_mask = zeros_like(self.relations).bool()
        uniques, counts = self.relations.unique(return_counts=True)
        for i, r in enumerate(uniques):
            rand = tensor(list(range(counts[i])))[randperm(counts[i])]

            sub_mask = (self.relations == r).nonzero()[:, 0]  # list of indices k such that relations[k] == r

            assert len(sub_mask) == counts[i].item()

            if validation:
                train_size, val_size, test_size = self.get_sizes(counts[i], share=share, validation=True)
                mask[sub_mask[rand[:train_size]]] = True
                val_mask[sub_mask[rand[train_size:train_size + val_size]]] = True


            else:
                train_size, test_size = self.get_sizes(counts[i], share=share, validation=False)
                mask[sub_mask[rand[:train_size]]] = True
                print(mask, train_size, (~mask).sum().item(), test_size)

        if validation:
            return mask, val_mask, ~(mask | val_mask)
        else:
            return mask, ~mask

    def get_sizes(self, count, share, validation=False):
        """
        with `count` samples, return how many should go to train and test
        """
        n_train = int(count * share)
        assert n_train < count
        if n_train == 0:
            n_train += 1

        if not validation:
            return n_train, count - n_train
        else:
            if count - n_train == 1:
                n_train -= 1
                return n_train, 1, 1
            else:
                n_val = int(int(count - n_train) / 2)
                return n_train, n_val, count - n_train - n_val

    def evaluate_dicts(self):
        """Evaluate dicts of possible alternatives to an entity in a fact that still gives a true\
        fact in the entire knowledge graph.

        """
        for i in range(self.n_facts):
            self.dict_of_heads[(self.tail_idx[i].item(),
                                self.relations[i].item())].add(self.head_idx[i].item())
            self.dict_of_tails[(self.head_idx[i].item(),
                                self.relations[i].item())].add(self.tail_idx[i].item())
