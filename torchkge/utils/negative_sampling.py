# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
aboschin@enst.fr
"""
from collections import defaultdict
from tqdm import tqdm


def fill_in_dicts(kg, possible_heads=None, possible_tails=None):
    if possible_heads is None:
        possible_heads = defaultdict(set)
    if possible_tails is None:
        possible_tails = defaultdict(set)

    for i in tqdm(range(kg.n_facts)):
        possible_heads[kg.relations[i].item()].add(kg.head_idx[i].item())
        possible_tails[kg.relations[i].item()].add(kg.tail_idx[i].item())

    return possible_heads, possible_tails
