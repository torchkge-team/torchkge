import pytest
import pandas as pd

from torch import tensor, Tensor
from torch.nn import Embedding

from torchkge.data import KnowledgeGraph

from torchkge.utils.data_preprocessing import get_dictionaries, get_tph, get_hpt, get_bernoulli_probs
from torchkge.utils.dissimilarities import l1_dissimilarity, l2_dissimilarity, l1_torus_dissimilarity, \
    l2_torus_dissimilarity, el2_torus_dissimilarity
from torchkge.utils.operations import one_hot, get_col, groupby_count, groupby_mean
from torchkge.utils.models_utils import init_embedding, get_true_targets


@pytest.fixture()
def get_df():
    return pd.DataFrame([[0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 4, 0], [1, 2, 1], [1, 3, 2], [2, 4, 0], [3, 4, 4],
                         [5, 4, 0]], columns=['from', 'to', 'rel'])


@pytest.fixture()
def get_graph():
    heads = tensor([0, 0, 0, 0, 1, 1, 2, 3, 5]).long()
    tails = tensor([1, 2, 3, 4, 2, 3, 4, 4, 4]).long()
    rels = tensor([0, 0, 0, 0, 1, 2, 0, 4, 0]).long()
    return heads, tails, rels


def test_get_dictionaries():
    df = pd.DataFrame([['a', 'R1', 'b'], ['c', 'R2', 'd']], columns=['from', 'rel', 'to'])
    d1 = {'a': 0, 'b': 2, 'c': 3, 'd': 1}
    d2 = {'R1': 0, 'R2': 1}
    assert get_dictionaries(df, ent=True) == d1
    assert get_dictionaries(df, ent=False) == d2


def test_get_tph():
    df = get_df()
    kg = KnowledgeGraph(df=df)
    assert get_tph(kg) == {0: 2, 1: 1, 2: 1, 4: 1}


def test_get_hpt():
    df = get_df()
    kg = KnowledgeGraph(df=df)
    assert get_hpt(kg) == {0.0: 1.5, 1.0: 1.0, 2.0: 1.0, 4.0: 1.0}


def test_get_bernoulli_probs():
    df = get_df()
    kg = KnowledgeGraph(df=df)
    probs = get_bernoulli_probs(kg)
    res = {0: 0.5714, 1: 0.5, 2: 0.5, 4: 0.5}

    for k in probs.keys():
        assert(res[k] == pytest.approx(probs[k], abs=0.001))


def test_dissimilarities():
    a = tensor([[1.4, 2, 3, 4], [5.4, 6, 7, 8]]).float()
    b = tensor([[1.3, 4, 2, 10], [5.9, 8, 6, 7]]).float()

    assert(l1_dissimilarity(a, b) == tensor([9.1000, 4.5000]))
    assert(l2_dissimilarity(a, b) == tensor([41.0100,  6.2500]))
    assert(l1_torus_dissimilarity(a, b) == tensor([0.1000, 0.5000]))
    assert(l2_torus_dissimilarity(a, b) == tensor([0.1000, 0.5000]))
    assert(el2_torus_dissimilarity(a, b) == tensor([0.6180, 2.0000]))


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


def test_groupby_count():
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

    r1 = {tensor([1, 0]): 1,
          tensor([2, 0]): 1,
          tensor([2, 1]): 1,
          tensor([3, 0]): 1,
          tensor([3, 2]): 1,
          tensor([4, 0]): 3,
          tensor([4, 4]): 1}

    r2 = {33: 2, 44: 2}

    assert groupby_count(t1, by=[1, 2]) == r1
    assert groupby_count(t2, by=1) == r2


def test_groupby_mean():
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

    r1 = {tensor([1, 0]): 0.0,
          tensor([2, 0]): 0.0,
          tensor([2, 1]): 1.0,
          tensor([3, 0]): 0.0,
          tensor([3, 2]): 1.0,
          tensor([4, 0]): 2.3333333,
          tensor([4, 4]): 3.0}

    r2 = {33: 3.0, 44: 2.0}

    assert groupby_mean(t1, by=[1, 2]) == r1
    assert groupby_mean(t2, by=1) == r2


def test_init_embedding():
    n = 10
    dim = 100

    p = init_embedding(n, dim)

    assert type(p) == Embedding
    assert p.weight.requires_grad
    assert p.weight.shape == (10, 100)

def test_get_true_targets():

