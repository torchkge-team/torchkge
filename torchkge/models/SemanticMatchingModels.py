# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
aboschin@enst.fr
"""

from torch import empty, matmul, tensor, diag_embed
from torch.nn import Module, Embedding, Parameter
from torch.nn.functional import normalize
from torch.nn.init import xavier_uniform_
from torchkge.utils import get_rank


class RESCALModel(Module):
    """Implement torch.nn.Module interface. This model should be implemented with ALS as in the\
    original paper.

    References
    ----------
    * Maximilian Nickel, Volker Tresp, and Hans-Peter Kriegel.
      A Three-way Model for Collective Learning on Multi-relational Data.
      In Proceedings of the 28th International Conference on Machine Learning, 2011.
      https://dl.acm.org/citation.cfm?id=3104584

    Parameters
    ----------
    config: Config object
        Contains all configuration parameters.

    Attributes
    ----------
    ent_emb_dim: int
        Dimension of the embedding of entities
    number_entities: int
        Number of entities in the current data set.
    entity_embeddings: torch Embedding, shape = (number_entities, ent_emb_dim)
        Contains the embeddings of the entities. It is initialized with Xavier uniform and then\
         normalized.
    relation_matrices: torch Parameter, shape = (number_relations, ent_emb_dim, ent_emb_dim)
        Contains the matrices of the relations. It is initialized with Xavier uniform.

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
        self.normalize_parameters()

    def forward(self, heads, tails, negative_heads, negative_tails, relations):
        """Forward pass on the current batch.

        Parameters
        ----------
        heads: torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's heads
        tails: torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's tails.
        negative_heads: torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's negatively sampled heads.
        negative_tails: torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's negatively sampled tails.
        relations: torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's relations.

        Returns
        -------
        golden_triplets: torch tensor, dtype = float, shape = (batch_size)
            Estimation of the true value that should be 1 (by matrix factorization).
        negative_triplets: torch tensor, dtype = float, shape = (batch_size)
            Estimation of the true value that should be 0 (by matrix factorization).

        """
        # recover entities embeddings
        heads_embeddings = normalize(self.entity_embeddings(heads), p=2, dim=1)
        tails_embeddings = normalize(self.entity_embeddings(tails), p=2, dim=1)
        neg_heads_embeddings = normalize(self.entity_embeddings(negative_heads), p=2, dim=1)
        neg_tails_embeddings = normalize(self.entity_embeddings(negative_tails), p=2, dim=1)

        # recover relation matrices
        relation_matrices = self.relation_matrices[relations]

        return self.compute_product(heads_embeddings, tails_embeddings, relation_matrices), \
            self.compute_product(neg_heads_embeddings, neg_tails_embeddings, relation_matrices)

    def normalize_parameters(self):
        self.entity_embeddings.weight.data = normalize(self.entity_embeddings.weight.data,
                                                       p=2, dim=1)

    def compute_product(self, heads, tails, rel_mat):
        """Compute the matrix product hRt^t with proper reshapes. It can do the batch matrix
        product both in the forward pass and in the evaluation pass with one matrix containing
        all candidates.

        Parameters
        ----------
        heads: torch Tensor, shape = (b_size, self.ent_emb_dim) or (b_size, self.number_entities,\
        self.ent_emb_dim), dtype = float
            Tensor containing embeddings of current head entities or candidates.
        tails: torch.Tensor, shape = (b_size, self.ent_emb_dim) or (b_size, self.number_entities,\
        self.ent_emb_dim), dtype = float
            Tensor containing embeddings of current tail entities or canditates.
        rel_mat: torch.Tensor, shape = (b_size, self.ent_emb_dim, self.ent_emb_dim), dtype = float
            Tensor containing relation matrices for current relations.

        Returns
        -------
        product: torch.Tensor, shape = (b_size) or (b_size, self.number_entities), dtype = float
            Tensor containing the matrix products h.W.t^t for each sample of the batch.

        """
        b_size = len(heads)
        if len(tails.shape) == 3:
            tails = tails.transpose(1, 2)
        else:
            tails = tails.view(b_size, self.ent_emb_dim, 1)

        if len(heads.shape) == 2:
            heads = heads.view(b_size, 1, self.ent_emb_dim)

        if (len(heads.shape) == 3) or (len(tails.shape) == 3):
            return matmul(matmul(heads, rel_mat), tails).view(b_size, -1)
        else:
            return matmul(matmul(heads, rel_mat), tails).view(b_size)

    def evaluation_helper(self, h_idx, t_idx, r_idx):
        """

        Parameters
        ----------
        h_idx: torch.Tensor, shape = (b_size,), dtype = long
            Tensor containing indices of current head entities.
        t_idx: torch.Tensor, shape = (b_size,), dtype = long
            Tensor containing indices of current tail entities.
        r_idx: torch.Tensor, shape = (b_size,), dtype = long
            Tensor containing indices of current relations.

        Returns
        -------
        h_emb: torch.Tensor, shape = (b_size, ent_emb_dim), dtype = float
            Tensor containing embeddings of current head entities.
        t_emb: torch.Tensor, shape = (b_size, ent_emb_dim), dtype = float
            Tensor containing embeddings of current tail entities.
        r_mat: torch.Tensor, shape = (b_size, ent_emb_dim, ent_emb_dim), dtype = float
            Tensor containing matrices of current relations.
        candidates: torch.Tensor, shape = (b_size, number_entities, ent_emb_dim), dtype = float
            Tensor containing all entities as candidates for each sample of the batch.

        """
        b_size = h_idx.shape[0]

        candidates = self.entity_embeddings.weight.data
        candidates = candidates.view(1, self.number_entities, self.ent_emb_dim)
        candidates = candidates.expand(b_size, self.number_entities, self.ent_emb_dim)

        h_emb = self.entity_embeddings(h_idx)
        t_emb = self.entity_embeddings(t_idx)
        r_mat = self.relation_matrices[r_idx]

        return h_emb, t_emb, r_mat, candidates

    def compute_ranks(self, e_emb, candidates, r_mat, e_idx, r_idx, true_idx, dictionary, heads=1):
        """

        Parameters
        ----------
        e_emb: torch tensor, shape = (batch_size, rel_emb_dim), dtype = float
            Tensor containing current embeddings of entities.
        candidates: torch tensor, shape = (b_size, number_entities, ent_emb_dim), dtype = float
            Tensor containing projected embeddings of all entities.
        r_mat: torch.Tensor, shape = (b_size, ent_emb_dim, ent_emb_dim), dtype = float
            Tensor containing current matrices of relations.
        e_idx: torch tensor, shape = (batch_size), dtype = long
            Tensor containing the indices of entities.
        r_idx: torch tensor, shape = (batch_size), dtype = long
            Tensor containing the indices of relations.
        true_idx: torch tensor, shape = (batch_size), dtype = long
            Tensor containing the true entity for each sample.
        dictionary: default dict
            Dictionary of keys (int, int) and values list of ints giving all possible entities for\
            the (entity, relation) pair.
        heads: integer
            1 ou -1 (must be 1 if entities are heads and -1 if entities are tails). We test\
             dissimilarity between heads * entities + relations and heads * targets.


        Returns
        -------
        rank_true_entities: torch Tensor, shape = (b_size), dtype = int
            Tensor containing the rank of the true entities when ranking any entity based on\
            estimation of 1 or 0.
        filtered_rank_true_entities: torch Tensor, shape = (b_size), dtype = int
            Tensor containing the rank of the true entities when ranking only true false entities\
            based on estimation of 1 or 0.

        """
        current_batch_size, _ = e_emb.shape

        if heads == 1:
            scores = self.compute_product(e_emb, candidates, r_mat)
        else:
            scores = self.compute_product(candidates, e_emb, r_mat)

        # filter out the true negative samples by assigning negative score
        filt_scores = scores.clone()
        for i in range(current_batch_size):
            true_targets = dictionary[e_idx[i].item(), r_idx[i].item()].copy()
            if len(true_targets) == 1:
                continue
            true_targets.remove(true_idx[i].item())
            true_targets = tensor(true_targets).long()
            filt_scores[i][true_targets] = float(-1)

        # from dissimilarities, extract the rank of the true entity.
        rank_true_entities = get_rank(scores, true_idx)
        filtered_rank_true_entities = get_rank(filt_scores, true_idx)

        return rank_true_entities, filtered_rank_true_entities

    def evaluate_candidates(self, h_idx, t_idx, r_idx, kg):
        h_emb, t_emb, r_mat, candidates = self.evaluation_helper(h_idx, t_idx, r_idx)

        rank_true_tails, filt_rank_true_tails = self.compute_ranks(h_emb,
                                                                   candidates,
                                                                   r_mat, h_idx, r_idx,
                                                                   t_idx,
                                                                   kg.dict_of_tails,
                                                                   heads=1)
        rank_true_heads, filt_rank_true_heads = self.compute_ranks(t_emb,
                                                                   candidates,
                                                                   r_mat, t_idx, r_idx,
                                                                   h_idx,
                                                                   kg.dict_of_heads,
                                                                   heads=-1)

        return rank_true_tails, filt_rank_true_tails, rank_true_heads, filt_rank_true_heads


class DistMultModel(RESCALModel):
    """Implement torch.nn.Module interface and inherits\
    torchkge.models.SemanticMatchingModels.RESCALModel.

    References
    ----------
    * Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng.
      Embedding Entities and Relations for Learning and Inference in Knowledge Bases.
      arXiv :1412.6575 [cs], December 2014. arXiv : 1412.6575.
      https://arxiv.org/abs/1412.6575

    Parameters
    ----------
    config: Config object
        Contains all configuration parameters.

    Attributes
    ----------
    ent_emb_dim: int
        Dimension of the embedding of entities
    number_entities: int
        Number of entities in the current data set.
    entity_embeddings: torch Embedding, shape = (number_entities, ent_emb_dim)
        Contains the embeddings of the entities. It is initialized with Xavier uniform and then\
         normalized.
    relation_vectors: torch Parameter, shape = (number_relations, ent_emb_dim)
        Contains the vectors to build diagonal matrices of the relations. It is initialized with
        Xavier uniform.

    """
    def __init__(self, config):
        super().__init__(config)

        del self.relation_matrices
        self.relation_vectors = Parameter(
            xavier_uniform_(empty(size=(self.number_relations, self.ent_emb_dim))))

    def forward(self, heads, tails, negative_heads, negative_tails, relations):
        """Forward pass on the current batch.

        Parameters
        ----------

        heads: torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's heads
        tails: torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's tails.
        negative_heads: torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's negatively sampled heads.
        negative_tails: torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's negatively sampled tails.
        relations: torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's relations.

        Returns
        -------

        golden_triplets: torch tensor, dtype = float, shape = (batch_size)
            Estimation of the true value that should be 1 (by matrix factorization).
        negative_triplets: torch tensor, dtype = float, shape = (batch_size)
            Estimation of the true value that should be 0 (by matrix factorization).

        """
        # recover entities embeddings
        heads_embeddings = normalize(self.entity_embeddings(heads), p=2, dim=1)
        tails_embeddings = normalize(self.entity_embeddings(tails), p=2, dim=1)
        neg_heads_embeddings = normalize(self.entity_embeddings(negative_heads), p=2, dim=1)
        neg_tails_embeddings = normalize(self.entity_embeddings(negative_tails), p=2, dim=1)

        # recover relation matrices
        relation_vectors = self.relation_vectors[relations]
        relation_matrices = diag_embed(relation_vectors)

        return self.compute_product(heads_embeddings, tails_embeddings, relation_matrices), \
            self.compute_product(neg_heads_embeddings, neg_tails_embeddings, relation_matrices)

    def evaluation_helper(self, h_idx, t_idx, r_idx):
        """

        Parameters
        ----------
        h_idx: torch.Tensor, shape = (b_size,), dtype = long
            Tensor containing indices of current head entities.
        t_idx: torch.Tensor, shape = (b_size,), dtype = long
            Tensor containing indices of current tail entities.
        r_idx: torch.Tensor, shape = (b_size,), dtype = long
            Tensor containing indices of current relations.

        Returns
        -------
        h_emb: torch.Tensor, shape = (b_size, ent_emb_dim), dtype = float
            Tensor containing embeddings of current head entities.
        t_emb: torch.Tensor, shape = (b_size, ent_emb_dim), dtype = float
            Tensor containing embeddings of current tail entities.
        r_mat: torch.Tensor, shape = (b_size, ent_emb_dim, ent_emb_dim), dtype = float
            Tensor containing matrices of current relations.
        candidates: torch.Tensor, shape = (b_size, number_entities, ent_emb_dim), dtype = float
            Tensor containing all entities as candidates for each sample of the batch.
        
        """
        b_size = h_idx.shape[0]

        candidates = self.entity_embeddings.weight.data
        candidates = candidates.view(1, self.number_entities, self.ent_emb_dim)
        candidates = candidates.expand(b_size, self.number_entities, self.ent_emb_dim)

        h_emb = self.entity_embeddings(h_idx)
        t_emb = self.entity_embeddings(t_idx)
        r_vec = self.relation_vectors[r_idx]
        r_mat = diag_embed(r_vec)

        return h_emb, t_emb, r_mat, candidates


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
