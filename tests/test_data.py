import pandas as pd
import unittest

from torch import Tensor, int64

from torchkge.data_structures import KnowledgeGraph
from torchkge.exceptions import WrongArgumentsError, SanityError, SizeMismatchError


class TestUtils(unittest.TestCase):
    """Tests for `torchkge.utils`."""

    def setUp(self):
        self.df = pd.DataFrame([[0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 4, 0], [1, 2, 1], [1, 3, 2], [2, 4, 0], [3, 4, 4],
                                [5, 4, 0]], columns=['from', 'to', 'rel'])
        self.kg = KnowledgeGraph(self.df)

    def test_KnowledgeGraph_Builder(self):
        assert len(self.kg) == 9
        assert self.kg.n_ent == 6
        assert self.kg.n_rel == 4
        assert (type(self.kg.rel2ix) == dict) & (type(self.kg.rel2ix) == dict)
        assert (type(self.kg.head_idx) == Tensor) & (type(self.kg.tail_idx) == Tensor) & \
               (type(self.kg.relations) == Tensor)
        assert (self.kg.head_idx.dtype == int64) & (self.kg.tail_idx.dtype == int64) & \
               (self.kg.relations.dtype == int64)
        assert (len(self.kg.head_idx) == len(self.kg.tail_idx) == len(self.kg.relations))

        kg_dict = {'heads': self.kg.head_idx, 'tails': self.kg.tail_idx, 'relations': self.kg.relations}
        with self.assertRaises(WrongArgumentsError):
            KnowledgeGraph()
        with self.assertRaises(WrongArgumentsError):
            KnowledgeGraph(kg=kg_dict, df=self.df)
        with self.assertRaises(WrongArgumentsError):
            KnowledgeGraph(kg=kg_dict)
        with self.assertRaises(WrongArgumentsError):
            KnowledgeGraph(kg={'heads': self.kg.head_idx, 'tails': self.kg.tail_idx},
                           ent2ix=self.kg.ent2ix, rel2ix=self.kg.rel2ix)
        with self.assertRaises(SanityError):
            KnowledgeGraph(kg={'heads': self.kg.head_idx[:-1],
                               'tails': self.kg.tail_idx,
                               'relations': self.kg.relations},
                           ent2ix=self.kg.ent2ix, rel2ix=self.kg.rel2ix)
        with self.assertRaises(SanityError):
            KnowledgeGraph(kg={'heads': self.kg.head_idx.int(),
                               'tails': self.kg.tail_idx,
                               'relations': self.kg.relations},
                           ent2ix=self.kg.ent2ix, rel2ix=self.kg.rel2ix)

    def test_split_kg(self):
        assert (len(self.kg.split_kg()) == 2) & (len(self.kg.split_kg(validation=True)) == 3)
        with self.assertRaises(SizeMismatchError):
            self.kg.split_kg(sizes=(1, 2, 3, 4))
        with self.assertRaises(WrongArgumentsError):
            self.kg.split_kg(sizes=(9, 9, 9))
        with self.assertRaises(WrongArgumentsError):
            self.kg.split_kg(sizes=(9, 9))

