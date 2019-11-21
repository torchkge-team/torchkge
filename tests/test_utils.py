import pytest
import pandas as pd

from torch import tensor, Tensor, cat, eq
from torch.nn import Embedding

from torchkge.data import KnowledgeGraph

from torchkge.utils.data_preprocessing import get_dictionaries, get_tph, get_hpt, get_bernoulli_probs
from torchkge.utils.dissimilarities import l1_dissimilarity, l2_dissimilarity, l1_torus_dissimilarity, \
    l2_torus_dissimilarity, el2_torus_dissimilarity
from torchkge.utils.operations import one_hot, get_col, groupby_count, groupby_mean
from torchkge.utils.models_utils import init_embedding, get_true_targets


def get_df():
    return pd.DataFrame([[0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 4, 0], [1, 2, 1], [1, 3, 2], [2, 4, 0], [3, 4, 4],
                         [5, 4, 0]], columns=['from', 'to', 'rel'])


def get_graph():
    heads = tensor([0, 0, 0, 0, 1, 1, 2, 3, 5]).long()
    tails = tensor([1, 2, 3, 4, 2, 3, 4, 4, 4]).long()
    rels = tensor([0, 0, 0, 0, 1, 2, 0, 4, 0]).long()
    return heads, tails, rels


def test_get_dictionaries():
    df = pd.DataFrame([['a', 'R1', 'b'], ['c', 'R2', 'd']], columns=['from', 'rel', 'to'])
    d1 = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    d2 = {'R1': 0, 'R2': 1}
    assert get_dictionaries(df, ent=True) == d1
    assert get_dictionaries(df, ent=False) == d2


def test_get_tph():
    df = get_df()
    kg = KnowledgeGraph(df=df)
    t = cat((kg.head_idx.view(-1, 1), kg.tail_idx.view(-1, 1), kg.relations.view(-1, 1)), dim=1)
    print(get_tph(t))
    assert get_tph(t) == {0: 2., 1: 1., 2: 1., 3: 1.}


def test_get_hpt():
    df = get_df()
    kg = KnowledgeGraph(df=df)
    t = cat((kg.head_idx.view(-1, 1), kg.tail_idx.view(-1, 1), kg.relations.view(-1, 1)), dim=1)
    print(get_hpt(t))
    assert get_hpt(t) == {0: 1.5, 1: 1., 2: 1., 3: 1.}


def test_get_bernoulli_probs():
    df = get_df()
    kg = KnowledgeGraph(df=df)
    probs = get_bernoulli_probs(kg)
    res = {0: 0.5714, 1: 0.5, 2: 0.5, 3: 0.5}

    for k in probs.keys():
        assert(res[k] == pytest.approx(probs[k], abs=0.001))


def test_dissimilarities():
    a = tensor([[1.4, 2, 3, 4], [5.4, 6, 7, 8]]).float()
    b = tensor([[1.3, 4, 2, 10], [5.9, 8, 6, 7]]).float()

    assert((l1_dissimilarity(a, b) == tensor([9.1000, 4.5000])).all() == 1)
    assert((l2_dissimilarity(a, b) - tensor([41.0100,  6.2500])).sum() < 1e-03)
    assert((l1_torus_dissimilarity(a, b) - tensor([0.1000, 0.5000])).sum() < 1e-03)
    assert((l2_torus_dissimilarity(a, b) - tensor([0.1000, 0.5000])).sum() < 1e-03)
    assert((el2_torus_dissimilarity(a, b) - tensor([0.6180, 2.0000])).sum() < 1e-03)


def test_one_hot():
    m = tensor([1, 3, 0, 4, 1])
    res = one_hot(m)
    true = tensor([[0., 1., 0., 0., 0.],
                   [0., 0., 0., 1., 0.],
                   [1., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 1.],
                   [0., 1., 0., 0., 0.]])

    assert((res == true).all().item() == 1)


def test_get_col():
    m1 = Tensor(10, 2)
    m2 = Tensor(10, 3)

    assert get_col(m1, 1) == 0
    assert get_col(m1, 0) == 1
    assert get_col(m2, [0, 1]) == 2
    assert get_col(m2, [0, 2]) == 1
    assert get_col(m2, [1, 2]) == 0
    with pytest.raises(AssertionError):
        get_col(m1, [1, 0])
    with pytest.raises(AssertionError):
        get_col(m2, 1)


def get_t_r(mean=False):
    t1 = tensor([[0, 1, 0],
                 [0, 2, 0],
                 [0, 3, 0],
                 [0, 4, 0],
                 [1, 2, 1],
                 [1, 3, 2],
                 [2, 4, 0],
                 [3, 4, 4],
                 [5, 4, 0]])

    t2 = tensor([[1, 44],
                 [2, 33],
                 [3, 44],
                 [4, 33]])

    if mean:
        r1 = {tensor([1, 0]): 0.0, tensor([2, 0]): 0.0, tensor([2, 1]): 1.0, tensor([3, 0]): 0.0, tensor([3, 2]): 1.0,
              tensor([4, 0]): 2.3333333, tensor([4, 4]): 3.0}
    else:
        r1 = {tensor([1, 0]): 1, tensor([2, 0]): 1, tensor([2, 1]): 1, tensor([3, 0]): 1, tensor([3, 2]): 1,
              tensor([4, 0]): 3, tensor([4, 4]): 1}

    if mean:
        r2 = {33: 3., 44: 2.}
    else:
        r2 = {33: 2, 44: 2}

    return t1, t2, r1, r2


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


def test_groupby_count():
    t1, t2, r1, r2 = get_t_r()
    compare_dicts_tensorkeys(groupby_count(t1, by=[1, 2]), r1)
    assert groupby_count(t2, by=1) == r2


def test_groupby_mean():
    t1, t2, r1, r2 = get_t_r(mean=True)
    compare_dicts_tensorkeys(groupby_mean(t1, by=[1, 2]), r1)
    assert groupby_mean(t2, by=1) == r2


def test_init_embedding():
    n = 10
    dim = 100

    p = init_embedding(n, dim)

    assert type(p) == Embedding
    assert p.weight.requires_grad
    assert p.weight.shape == (10, 100)


def test_get_true_targets():
    e_idx = tensor([0, 0, 0]).long()
    r_idx = tensor([0, 0, 1]).long()
    true_idx = tensor([1, 2, 1]).long()
    dictionary = {(0, 0): [0, 1, 2], (0, 1): [1]}

    assert eq(get_true_targets(dictionary, e_idx, r_idx, true_idx, 0), tensor([0, 2]).long()).all().item()
    assert eq(get_true_targets(dictionary, e_idx, r_idx, true_idx, 1), tensor([0, 1]).long()).all().item()
    assert get_true_targets(dictionary, e_idx, r_idx, true_idx, 2) is None
