# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
aboschin@enst.fr
"""

from torch import Tensor, cuda, randint, bernoulli
from torch.utils.data import Dataset, DataLoader

from torchkge.utils import get_dictionaries, lists_from_dicts, concatenate_diff_sizes

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
        use_cuda : bool
            Indicates if current object has been moved to cuda.

    """

    def __init__(self, df=None, kg=None,
                 ent2ix=None, rel2ix=None, list_of_heads=None, list_of_tails=None):

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
            self.head_idx = Tensor(df['from'].map(self.ent2ix).values).long()
            self.tail_idx = Tensor(df['to'].map(self.ent2ix).values).long()
            self.relations = Tensor(df['rel'].map(self.rel2ix).values).long()
        else:
            assert kg is not None
            self.n_sample = kg['heads'].shape[0]
            self.head_idx = kg['heads']
            self.tail_idx = kg['tails']
            self.relations = kg['relations']

        self.dict_of_heads = defaultdict(list)
        self.dict_of_tails = defaultdict(list)
        self.list_of_heads = list_of_heads
        self.list_of_tails = list_of_tails

        self.use_cuda = False
        if self.list_of_heads is None or self.list_of_tails is None:
            self.list_evaluated = False
        else:
            self.list_evaluated = True

    def __len__(self):
        return self.n_sample

    def __getitem__(self, item):
        if not self.list_evaluated:
            return self.head_idx[item].item(), self.tail_idx[item].item(), \
                   self.relations[item].item()
        else:
            return self.head_idx[item].item(), self.tail_idx[item].item(), \
                   self.relations[item].item(), self.list_of_heads[item], self.list_of_tails[item]

    def cuda(self):
        """Move the KnowledgeGraph object to cuda
        """
        self.use_cuda = True
        self.head_idx = self.head_idx.cuda()
        self.tail_idx = self.tail_idx.cuda()
        self.relations = self.relations.cuda()
        if self.list_of_heads is not None:
            self.list_of_heads = self.list_of_heads.cuda()
        if self.list_of_tails is not None:
            self.list_of_tails = self.list_of_tails.cuda()

    def evaluate_lists(self, batch_size=1000):
        """Evaluate lists of possible alternatives to an entity in a fact that still gives a true
        fact in the entire knowledge graph.

        Parameters
        ----------
        batch_size : int
        """

        for i in tqdm(range(self.n_sample)):
            self.dict_of_heads[(self.tail_idx[i].item(),
                                self.relations[i].item())].append(self.head_idx[i].item())
            self.dict_of_tails[(self.head_idx[i].item(),
                                self.relations[i].item())].append(self.tail_idx[i].item())

        if self.use_cuda:
            self.cuda()

        dataloader = DataLoader(self, batch_size=batch_size, shuffle=False)

        self.list_of_heads = Tensor().long()
        self.list_of_tails = Tensor().long()

        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            heads, tails, rels = batch[0], batch[1], batch[2]

            if self.use_cuda and cuda.is_available():
                heads, tails, rels = heads.cuda(), tails.cuda(), rels.cuda()

            self.list_of_heads = concatenate_diff_sizes(self.list_of_heads,
                                                        lists_from_dicts(self.dict_of_heads,
                                                                         tails, rels, heads))
            self.list_of_tails = concatenate_diff_sizes(self.list_of_tails,
                                                        lists_from_dicts(self.dict_of_tails,
                                                                         heads, rels, tails))

        self.list_evaluated = True

    def split_kg(self, share=0.8):
        """Split the knowledge graph into train and test.

        Parameters
        ----------
        share : float
            Percentage to allocate to train set.

        Returns
        -------
        train_kg : KnowledgeGraph
        test_kg : KnowledgeGraph
        """
        if not self.list_evaluated:
            print('Please note that lists of heads and tails are not evaluated.')
            print('Those should be evaluated before splitting the graph.')

        mask = (Tensor(self.head_idx.shape).uniform_() < share)

        train_kg = KnowledgeGraph(
            kg={'heads': self.head_idx[mask],
                'tails': self.tail_idx[mask],
                'relations': self.relations[mask]},
            ent2ix=self.ent2ix, rel2ix=self.rel2ix,
            list_of_heads=self.list_of_heads[mask], list_of_tails=self.list_of_tails[mask])

        test_kg = KnowledgeGraph(
            kg={'heads': self.head_idx[~mask],
                'tails': self.tail_idx[~mask],
                'relations': self.relations[~mask]},
            ent2ix=self.ent2ix, rel2ix=self.rel2ix,
            list_of_heads=self.list_of_heads[~mask], list_of_tails=self.list_of_tails[~mask])

        if self.use_cuda:
            train_kg.cuda()
            test_kg.cuda()

        return train_kg, test_kg

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
