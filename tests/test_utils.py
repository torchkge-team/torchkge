import pandas as pd
import unittest

from collections import defaultdict
from torch import tensor, cat, eq, bool
from torch.nn import Embedding

from torchkge.data_structures import KnowledgeGraph
from torchkge.utils.dissimilarities import l1_dissimilarity, l2_dissimilarity, \
    l1_torus_dissimilarity, l2_torus_dissimilarity, el2_torus_dissimilarity
from torchkge.utils.modeling import init_embedding, get_true_targets
from torchkge.sampling import get_possible_heads_tails
from torchkge.utils.operations import get_mask, get_rank
from torchkge.utils.operations import get_dictionaries, get_tph, get_hpt, \
    get_bernoulli_probs


class TestUtils(unittest.TestCase):
    """Tests for `torchkge.utils`."""

    def setUp(self):
        self.df = pd.DataFrame([[0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 4, 0],
                                [1, 2, 1], [1, 3, 2], [2, 4, 0], [3, 4, 4],
                                [5, 4, 0]], columns=['from', 'to', 'rel'])
        self.heads = tensor([0, 0, 0, 0, 1, 1, 2, 3, 5]).long()
        self.tails = tensor([1, 2, 3, 4, 2, 3, 4, 4, 4]).long()
        self.rels = tensor([0, 0, 0, 0, 1, 2, 0, 4, 0]).long()

        # get_dictionaries
        self.d1 = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
        self.d2 = {'R1': 0, 'R2': 1}

        # dissimilarities
        self.a = tensor([[1.4, 2, 3, 4], [5.4, 6, 7, 8]]).float()
        self.b = tensor([[1.3, 4, 2, 10], [5.9, 8, 6, 7]]).float()

        # bernoulli
        self.t1 = tensor([[0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 4, 0],
                          [1, 2, 1], [1, 3, 2], [2, 4, 0], [3, 4, 4],
                          [5, 4, 0]])
        self.t2 = tensor([[1, 44], [2, 33], [3, 44], [4, 33]])
        self.r1_mean = {tensor([1, 0]): 0.0, tensor([2, 0]): 0.0,
                        tensor([2, 1]): 1.0, tensor([3, 0]): 0.0,
                        tensor([3, 2]): 1.0, tensor([4, 0]): 2.3333333,
                        tensor([4, 4]): 3.0}
        self.r1_count = {tensor([1, 0]): 1, tensor([2, 0]): 1,
                         tensor([2, 1]): 1, tensor([3, 0]): 1,
                         tensor([3, 2]): 1, tensor([4, 0]): 3,
                         tensor([4, 4]): 1}
        self.r2_mean = {33: 3., 44: 2.}
        self.r2_count = {33: 2, 44: 2}

        # get_true_targets
        self.e_idx = tensor([0, 0, 0]).long()
        self.r_idx = tensor([0, 0, 1]).long()
        self.true_idx = tensor([1, 2, 1]).long()
        self.dictionary = {(0, 0): [0, 1, 2], (0, 1): [1]}

    @staticmethod
    def compare_dicts_tensorkeys(d1, d2):
        for k in d1.keys():
            found = False
            for kk in d2.keys():
                if eq(k, kk).all():
                    found = True
                    assert (d1[k] - d2[kk]) < 1e-03
                    continue
            if not found:
                raise AssertionError

    def test_get_dictionaries(self):
        df = pd.DataFrame([['a', 'R1', 'b'], ['c', 'R2', 'd']],
                          columns=['from', 'rel', 'to'])
        assert get_dictionaries(df, ent=True) == self.d1
        assert get_dictionaries(df, ent=False) == self.d2

    def test_get_tph(self):
        kg = KnowledgeGraph(df=self.df)
        t = cat((kg.head_idx.view(-1, 1), kg.tail_idx.view(-1, 1),
                 kg.relations.view(-1, 1)), dim=1)
        assert get_tph(t) == {0: 2., 1: 1., 2: 1., 3: 1.}

    def test_get_hpt(self):
        kg = KnowledgeGraph(df=self.df)
        t = cat((kg.head_idx.view(-1, 1), kg.tail_idx.view(-1, 1),
                 kg.relations.view(-1, 1)), dim=1)
        assert get_hpt(t) == {0: 1.5, 1: 1., 2: 1., 3: 1.}

    def test_get_bernoulli_probs(self):
        kg = KnowledgeGraph(df=self.df)
        probs = get_bernoulli_probs(kg)
        res = {0: 0.5714, 1: 0.5, 2: 0.5, 3: 0.5}

        for k in probs.keys():
            assert (res[k] - probs[k]) < 1e-03

    def test_dissimilarities(self):
        assert ((l1_dissimilarity(self.a, self.b) ==
                 tensor([9.1000, 4.5000])).all() == 1)
        assert ((l2_dissimilarity(self.a, self.b) -
                 tensor([41.0100, 6.2500])).sum() < 1e-03)
        assert ((l1_torus_dissimilarity(self.a, self.b) -
                 2 * tensor([0.1000, 0.5000])).sum() < 1e-03)
        assert ((l2_torus_dissimilarity(self.a, self.b) -
                 4 * tensor([0.1000, 0.5000])**2).sum() < 1e-03)
        assert ((el2_torus_dissimilarity(self.a, self.b) -
                 tensor([0.6180, 2.0000])).sum() < 1e-03)

    def test_init_embedding(self):
        n = 10
        dim = 100

        p = init_embedding(n, dim)

        assert type(p) == Embedding
        assert p.weight.requires_grad
        assert p.weight.shape == (10, 100)

    def test_get_true_targets(self):
        assert eq(get_true_targets(self.dictionary, self.e_idx,
                                   self.r_idx, self.true_idx, 0),
                  tensor([0, 2]).long()).all().item()
        assert eq(get_true_targets(self.dictionary, self.e_idx,
                                   self.r_idx, self.true_idx, 1),
                  tensor([0, 1]).long()).all().item()
        assert get_true_targets(self.dictionary, self.e_idx,
                                self.r_idx, self.true_idx, 2) is None

    def test_get_possible_heads_tails(self):
        kg = KnowledgeGraph(self.df)
        h, t = get_possible_heads_tails(kg)

        assert (type(h) == dict) & (type(t) == dict)

        assert h == {0: {0, 2, 5}, 1: {1}, 2: {1}, 3: {3}}
        assert t == {0: {1, 2, 3, 4}, 1: {2}, 2: {3}, 3: {4}}

        p_h, p_t = defaultdict(set), defaultdict(set)
        p_h[0].add(40)
        p_h[10].add(50)
        p_t[0].add(41)
        p_t[10].add(51)

        h, t = get_possible_heads_tails(kg, possible_heads=dict(p_h),
                                        possible_tails=dict(p_t))

        assert h == {0: {0, 2, 5, 40}, 1: {1}, 2: {1}, 3: {3}, 10: {50}}
        assert t == {0: {1, 2, 3, 4, 41}, 1: {2}, 2: {3}, 3: {4}, 10: {51}}

    def test_get_mask(self):
        m = get_mask(10, 1, 2)
        assert m.dtype == bool
        assert len(m.shape) == 1
        assert m.shape[0] == 10

    def test_get_rank(self):
        data = tensor([[1, 2, 3, 4, 0], [1, 2, 1, 3, 0]]).float()
        true = tensor([4, 2])
        r1 = get_rank(data, true)
        r2 = get_rank(data, true, low_values=True)

        assert eq(r1, tensor([5, 4])).all()
        assert eq(r2, tensor([1, 3])).all()
