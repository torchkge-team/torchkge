# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
aboschin@enst.fr
"""


class Config:
    def __init__(self, ent_emb_dim=None, rel_emb_dim=None, n_ent=None, n_rel=None, norm_type=None):
        self.entities_embedding_dimension = ent_emb_dim
        self.relations_embedding_dimension = rel_emb_dim
        self.number_entities = n_ent
        self.number_relations = n_rel
        self.norm_type = norm_type
