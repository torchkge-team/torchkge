# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""

from collections import defaultdict


def get_possible_heads_tails(kg, possible_heads=None, possible_tails=None):
    """Gets for each relation of the knowledge graph the possible heads and possible tails.

    Parameters
    ----------
    kg: `torchkge.data.KnowledgeGraph.KnowledgeGraph`
    possible_heads: defaultdict, optional (default=None)
    possible_tails: defaultdict, optional (default=None)

    Returns
    -------
    possible_heads: defaultdict, optional (default=None)
        keys: relation indices, values: set of possible heads for each relations
    possible_tails: defaultdict, optional (default=None)
        keys: relation indices, values: set of possible tails for each relations

    """

    if possible_heads is None:
        possible_heads = defaultdict(set)
    else:
        assert type(possible_heads) == defaultdict
    if possible_tails is None:
        possible_tails = defaultdict(set)
    else:
        assert type(possible_tails) == defaultdict

    for i in range(kg.n_facts):
        possible_heads[kg.relations[i].item()].add(kg.head_idx[i].item())
        possible_tails[kg.relations[i].item()].add(kg.tail_idx[i].item())

    return possible_heads, possible_tails
