# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>

This module contains functions implementing methods explained in `this
paper<https://arxiv.org/pdf/2003.08001.pdf>`__ by Akrami et al.
"""
from itertools import combinations
from torch import cat
from tqdm.autonotebook import tqdm


def concat_kgs(kg_tr, kg_val, kg_te):
    h = cat((kg_tr.head_idx, kg_val.head_idx, kg_te.head_idx))
    t = cat((kg_tr.tail_idx, kg_val.tail_idx, kg_te.tail_idx))
    r = cat((kg_tr.relations, kg_val.relations, kg_te.relations))
    return h, t, r


def get_pairs(kg, r, type='ht'):
    mask = (kg.relations == r)

    if type == 'ht':
        return set((i.item(), j.item()) for i, j in cat(
            (kg.head_idx[mask].view(-1, 1),
             kg.tail_idx[mask].view(-1, 1)), dim=1))
    else:
        assert type == 'th'
        return set((j.item(), i.item()) for i, j in cat(
            (kg.head_idx[mask].view(-1, 1),
             kg.tail_idx[mask].view(-1, 1)), dim=1))


def count_triplets(kg1, kg2, duplicates, rev_duplicates):
    """
    Parameters
    ----------
    kg1: torchkge.data_structures.KnowledgeGraph
    kg2: torchkge.data_structures.KnowledgeGraph
    duplicates: list
        List returned by torchkge.utils.data_redundancy.duplicates.
    rev_duplicates: list
        List returned by torchkge.utils.data_redundancy.duplicates.

    Returns
    -------
    n_duplicates: int
        Number of triplets in kg2 that have their duplicate triplet
        in kg1
    n_rev_duplicates: int
        Number of triplets in kg2 that have their reverse duplicate
        triplet in kg1.
    """
    n_duplicates = 0
    for r1, r2 in duplicates:
        ht_tr = get_pairs(kg1, r2, type='ht')
        ht_te = get_pairs(kg2, r1, type='ht')

        n_duplicates += len(ht_te.intersection(ht_tr))

        ht_tr = get_pairs(kg1, r1, type='ht')
        ht_te = get_pairs(kg2, r2, type='ht')

        n_duplicates += len(ht_te.intersection(ht_tr))

    n_rev_duplicates = 0
    for r1, r2 in rev_duplicates:
        th_tr = get_pairs(kg1, r2, type='th')
        ht_te = get_pairs(kg2, r1, type='ht')

        n_rev_duplicates += len(ht_te.intersection(th_tr))

        th_tr = get_pairs(kg1, r1, type='th')
        ht_te = get_pairs(kg2, r2, type='ht')

        n_rev_duplicates += len(ht_te.intersection(th_tr))

    return n_duplicates, n_rev_duplicates


def duplicates(kg_tr, kg_val, kg_te, theta1=0.8, theta2=0.8,
               verbose=False, counts=False, reverses=None):
    """Return the duplicate and reverse duplicate relations as explained
    in paper by Akrami et al.

    References
    ----------
    * Farahnaz Akrami, Mohammed Samiul Saeef, Quingheng Zhang.
      `Realistic Re-evaluation of Knowledge Graph Completion Methods:
      An Experimental Study. <https://arxiv.org/pdf/2003.08001.pdf>`_
      SIGMOD’20, June 14–19, 2020, Portland, OR, USA

    Parameters
    ----------
    kg_tr: torchkge.data_structures.KnowledgeGraph
        Train set
    kg_val: torchkge.data_structures.KnowledgeGraph
        Validation set
    kg_te: torchkge.data_structures.KnowledgeGraph
        Test set
    theta1: float
        First threshold (see paper).
    theta2: float
        Second threshold (see paper).
    verbose: bool
    counts: bool
        Should the triplets involving (reverse) duplicate relations be
        counted in all sets.
    reverses: list
        List of known reverse relations.

    Returns
    -------
    duplicates: list
        List of pairs giving duplicate relations.
    rev_duplicates: list
        List of pairs giving reverse duplicate relations.
    """
    if verbose:
        print('Computing Ts')

    if reverses is None:
        reverses = []

    T = dict()
    T_inv = dict()
    lengths = dict()

    h, t, r = concat_kgs(kg_tr, kg_val, kg_te)

    for r_ in tqdm(range(kg_tr.n_rel)):
        mask = (r == r_)
        lengths[r_] = mask.sum().item()

        pairs = cat((h[mask].view(-1, 1), t[mask].view(-1, 1)), dim=1)

        T[r_] = set([(h_.item(), t_.item()) for h_, t_ in pairs])
        T_inv[r_] = set([(t_.item(), h_.item()) for h_, t_ in pairs])

    if verbose:
        print('Finding duplicate relations')

    duplicates = []
    rev_duplicates = []

    iter_ = list(combinations(range(1345), 2))

    for r1, r2 in tqdm(iter_):
        a = len(T[r1].intersection(T[r2])) / lengths[r1]
        b = len(T[r1].intersection(T[r2])) / lengths[r2]

        if a > theta1 and b > theta2:
            duplicates.append((r1, r2))

        if (r1, r2) not in reverses:
            a = len(T[r1].intersection(T_inv[r2])) / lengths[r1]
            b = len(T[r1].intersection(T_inv[r2])) / lengths[r2]

            if a > theta1 and b > theta2:
                rev_duplicates.append((r1, r2))

    if verbose:
        print('Duplicate relations: {}'.format(len(duplicates)))
        print('Reverse duplicate relations: '
              '{}\n'.format(len(rev_duplicates)))

    if counts:
        dupl, rev = count_triplets(kg_tr, kg_tr, duplicates, rev_duplicates)
        print('{} train triplets have duplicate in train set '
              '({}%)'.format(dupl, int(dupl / len(kg_tr))))
        print('{} train triplets have reverse duplicate in train set '
              '({}%)\n'.format(rev, int(rev / len(kg_tr) * 100)))

        dupl, rev = count_triplets(kg_tr, kg_te, duplicates, rev_duplicates)
        print('{} test triplets have duplicate in train set '
              '({}%)'.format(dupl, int(dupl / len(kg_te))))
        print('{} test triplets have reverse duplicate in train set '
              '({}%)\n'.format(rev, int(rev / len(kg_te) * 100)))

        dupl, rev = count_triplets(kg_te, kg_te, duplicates, rev_duplicates)
        print('{} test triplets have duplicate in test set '
              '({}%)'.format(dupl, int(dupl / len(kg_te))))
        print('{} test triplets have reverse duplicate in test set '
              '({}%)\n'.format(rev, int(rev / len(kg_te) * 100)))

    return duplicates, rev_duplicates


def cartesian_product_relations(kg_tr, kg_val, kg_te, theta=0.8):
    """Return the cartesian product relations as explained in paper by
    Akrami et al.

    References
    ----------
    * Farahnaz Akrami, Mohammed Samiul Saeef, Quingheng Zhang.
      `Realistic Re-evaluation of Knowledge Graph Completion Methods: An
      Experimental Study. <https://arxiv.org/pdf/2003.08001.pdf>`_
      SIGMOD’20, June 14–19, 2020, Portland, OR, USA

    Parameters
    ----------
    kg_tr: torchkge.data_structures.KnowledgeGraph
        Train set
    kg_val: torchkge.data_structures.KnowledgeGraph
        Validation set
    kg_te: torchkge.data_structures.KnowledgeGraph
        Test set
    theta: float
        Threshold used to compute the cartesian product relations.

    Returns
    -------
    selected_relations: list
        List of relations index that are cartesian product relations
        (see paper for details).

    """
    selected_relations = []

    h, t, r = concat_kgs(kg_tr, kg_val, kg_te)

    S = dict()
    O = dict()
    lengths = dict()

    for r_ in tqdm(range(kg_tr.n_rel)):
        mask = (r == r_)
        lengths[r_] = mask.sum().item()

        S[r_] = set(h_.item() for h_ in h[mask])
        O[r_] = set(t_.item() for t_ in t[mask])

        if lengths[r_] / (len(S[r_]) * len(O[r_])) > theta:
            selected_relations.append(r_)

    return selected_relations
