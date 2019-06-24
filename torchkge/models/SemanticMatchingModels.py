# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
aboschin@enst.fr
"""

from torch import empty, matmul
from torch.nn import Module, Embedding, Parameter
from torch.nn.functional import normalize
from torch.nn.init import xavier_uniform_


class RESCALModel(Module):
    """
    This module should be implemented with ALS. For the time being it is implemented with SGD
    but the loss is still missing sum(true - predicted)**2.
    """
    def __init__(self, config):
        super().__init__()
        self.ent_emb_dim = config.entities_embedding_dimension
        self.number_entities = config.number_entities
        self.number_relations = config.number_relations

        # initialize embedding objects
        self.entity_embeddings = Embedding(self.number_entities, self.ent_emb_dim)
        self.relation_matrices = Parameter(xavier_uniform_(empty(size=(self.number_relations,
                                                                       self.ent_emb_dim,
                                                                       self.ent_emb_dim))))

        # fill the embedding weights with Xavier initialized values
        self.entity_embeddings.weight = Parameter(xavier_uniform_(
            empty(size=(self.number_entities, self.ent_emb_dim))))

        # normalize the embeddings
        self.entity_embeddings.weight.data = normalize(self.entity_embeddings.weight.data,
                                                       p=2, dim=1)

    def forward(self, heads, tails, negative_heads, negative_tails, relations):
        # recover entities embeddings
        heads_embeddings = self.entity_embeddings(heads)
        tails_embeddings = self.entity_embeddings(tails)
        neg_heads_embeddings = self.entity_embeddings(negative_heads)
        neg_tails_embeddings = self.entity_embeddings(negative_tails)

        # recover relation matrices
        relation_matrices = self.relation_matrices[relations]

        return self.compute_product(heads_embeddings, tails_embeddings, relation_matrices), \
            self.compute_product(neg_heads_embeddings, neg_tails_embeddings, relation_matrices)

    def compute_product(self, heads, tails, rel_mat):
        b_size = len(heads)
        tmp = matmul(heads.view(b_size, 1, self.ent_emb_dim), rel_mat)
        return matmul(tmp, tails.view(b_size, self.ent_emb_dim, 1)).view(b_size)

    def normalize_parameters(self):
        pass


class DistMulModel(Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass

    def normalize_parameters(self):
        pass


class HolEModel(Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass

    def normalize_parameters(self):
        pass


class ComplexModel(Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass

    def normalize_parameters(self):
        pass


class AnalogyModel(Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass

    def normalize_parameters(self):
        pass
