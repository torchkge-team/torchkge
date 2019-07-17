# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
aboschin@enst.fr
"""

from torch import empty, matmul, tensor, diag_embed
from torch.nn import Module, Embedding, Parameter
from torch.nn.functional import normalize
from torch.nn.init import xavier_uniform_
from torchkge.utils import get_rank, get_mask, get_rolling_matrix
from torchkge.exceptions import WrongDimensionError


class RESCALModel(Module):
    """Implementation of RESCAL model detailed in 2011 paper by Nickel et al..\
    In the original paper, optimization is done using Alternating Least Squares (ALS). Here we use\
    iterative gradient descent optimization.

    References
    ----------
    * Maximilian Nickel, Volker Tresp, and Hans-Peter Kriegel.
      A Three-way Model for Collective Learning on Multi-relational Data.
      In Proceedings of the 28th International Conference on Machine Learning, 2011.
      https://dl.acm.org/citation.cfm?id=3104584

    Parameters
    ----------
    ent_emb_dim: int
        Dimension of embedding space.
    n_entities: int
        Number of entities in the current data set.
    n_relations: int
        Number of relations in the current data set.

    Attributes
    ----------
    ent_emb_dim: int
        Dimension of the embedding of entities
    number_entities: int
        Number of entities in the current data set.
    entity_embeddings: torch.nn.Embedding, shape = (number_entities, ent_emb_dim)
        Contains the embeddings of the entities. It is initialized with Xavier uniform and then\
         normalized.
    relation_matrices: torch Parameter, shape = (number_relations, ent_emb_dim, ent_emb_dim)
        Contains the matrices of the relations. It is initialized with Xavier uniform.

    """

    def __init__(self, ent_emb_dim, n_entities, n_relations):
        super().__init__()
        self.ent_emb_dim = ent_emb_dim
        self.number_entities = n_entities
        self.number_relations = n_relations

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
        """Compute the matrix product h^tRt with proper reshapes. It can do the batch matrix
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

        if len(heads.shape) == 2 and len(tails.shape) == 2:
            heads = heads.view(b_size, 1, self.ent_emb_dim)
            tails = tails.view(b_size, self.ent_emb_dim, 1)
            return matmul(matmul(heads, rel_mat), tails).view(b_size)

        elif len(heads.shape) == 2 and len(tails.shape) == 3:
            heads = heads.view(b_size, 1, self.ent_emb_dim)
            tails = tails.transpose(1, 2)
        else:
            tails = tails.view(b_size, self.ent_emb_dim, 1)

        return matmul(matmul(heads, rel_mat), tails).view(b_size, -1)

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
    """Implementation of DistMult model detailed in 2014 paper by Yang et al..

    References
    ----------
    * Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng.
      Embedding Entities and Relations for Learning and Inference in Knowledge Bases.
      arXiv :1412.6575 [cs], December 2014. arXiv : 1412.6575.
      https://arxiv.org/abs/1412.6575

    Parameters
    ----------
    ent_emb_dim: int
        Dimension of embedding space.
    n_entities: int
        Number of entities in the current data set.
    n_relations: int
        Number of relations in the current data set.

    Attributes
    ----------
    ent_emb_dim: int
        Dimension of the embedding of entities
    number_entities: int
        Number of entities in the current data set.
    entity_embeddings: torch.nn.Embedding, shape = (number_entities, ent_emb_dim)
        Contains the embeddings of the entities. It is initialized with Xavier uniform and then\
         normalized.
    relation_vectors: torch Parameter, shape = (number_relations, ent_emb_dim)
        Contains the vectors to build diagonal matrices of the relations. It is initialized with
        Xavier uniform.

    """

    def __init__(self, ent_emb_dim, n_entities, n_relations):
        super().__init__(ent_emb_dim, n_entities, n_relations)

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
        relation_matrices = diag_embed(self.relation_vectors[relations])

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
        r_mat = diag_embed(self.relation_vectors[r_idx])

        return h_emb, t_emb, r_mat, candidates


class HolEModel(RESCALModel):
    """Implementation of HolE model detailed in 2015 paper by Nickel et al..

        References
        ----------
        * Maximilian Nickel, Lorenzo Rosasco, and Tomaso Poggio.
          Holographic Embeddings of Knowledge Graphs.
          arXiv :1510.04935 [cs, stat], October 2015. arXiv : 1510.04935.
          https://arxiv.org/abs/1510.04935

        Parameters
        ----------
        ent_emb_dim: int
            Dimension of embedding space.
        n_entities: int
            Number of entities in the current data set.
        n_relations: int
            Number of relations in the current data set.

        Attributes
        ----------
        ent_emb_dim: int
            Dimension of the embedding of entities
        number_entities: int
            Number of entities in the current data set.
        entity_embeddings: torch.nn.Embedding, shape = (number_entities, ent_emb_dim)
            Contains the embeddings of the entities. It is initialized with Xavier uniform and then\
             normalized.
        relation_vectors: torch Parameter, shape = (number_relations, ent_emb_dim)
            Contains the vectors to build circular matrices of the relations. It is initialized
            with Xavier uniform.

        """
    def __init__(self, ent_emb_dim, n_entities, n_relations):
        super().__init__(ent_emb_dim, n_entities, n_relations)

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
        relation_matrices = get_rolling_matrix(self.relation_vectors[relations])

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
        r_mat = get_rolling_matrix(self.relation_vectors[r_idx])

        return h_emb, t_emb, r_mat, candidates

    def normalize_parameters(self):
        pass


class ComplExModel(DistMultModel):
    """Implementation of ComplEx model detailed in 2016 paper by Trouillon et al..

    References
    ----------
    * Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier, and Guillaume Bouchard.
      Complex Embeddings for Simple Link Prediction.
      arXiv :1606.06357 [cs, stat], June 2016. arXiv : 1606.06357.
      https://arxiv.org/abs/1606.06357

    Parameters
    ----------
    ent_emb_dim: int
        Dimension of embedding space.
    n_entities: int
        Number of entities in the current data set.
    n_relations: int
        Number of relations in the current data set.

    Attributes
    ----------
    ent_emb_dim: int
        Dimension of the embedding of entities
    number_entities: int
        Number of entities in the current data set.
    entity_embeddings: torch.nn.Embedding, shape = (number_entities, ent_emb_dim)
        Contains the embeddings of the entities. It is initialized with Xavier uniform and then\
         normalized.
    relation_vectors: torch Parameter, shape = (number_relations, ent_emb_dim)
        Contains the vectors to build block-diagonal matrices of the relations. It is initialized
        with Xavier uniform.
    smaller_dim: int
        Number of 2x2 matrices on the diagonals of relation-specific matrices.
    """
    def __init__(self, ent_emb_dim, n_entities, n_relations):
        try:
            assert ent_emb_dim % 2 == 0
        except AssertionError:
            raise WrongDimensionError('Embedding dimension should be pair.')

        super().__init__(ent_emb_dim, n_entities, n_relations)

        self.smaller_dim = int(self.ent_emb_dim / 2)

        self.real_mask = get_mask(self.ent_emb_dim, 0, self.smaller_dim)
        self.im_mask = get_mask(self.ent_emb_dim, self.smaller_dim, self.ent_emb_dim)

    def compute_product(self, heads, tails, rel_mat):
        """Compute the matrix product h^tRt with proper reshapes. It can do the batch matrix
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
        product: torch.Tensor, shape = (b_size, 1) or (b_size, self.number_entities), dtype = float
            Tensor containing the matrix products h^t.W.t for each sample of the batch.

        """
        b_size = len(heads)
        r_re = rel_mat[:, self.real_mask][:, :, self.real_mask]
        r_im = rel_mat[:, self.im_mask][:, :, self.im_mask]

        if len(heads.shape) == 2 and len(tails.shape) == 2:
            h_re = heads[:, self.real_mask].view(b_size, 1, self.smaller_dim)
            h_im = heads[:, self.im_mask].view(b_size, 1, self.smaller_dim)

            t_re = tails[:, self.real_mask].view(b_size, self.smaller_dim, 1)
            t_im = tails[:, self.im_mask].view(b_size, self.smaller_dim, 1)

        elif len(heads.shape) == 2 and len(tails.shape) == 3:
            h_re = heads[:, self.real_mask].view(b_size, 1, self.smaller_dim)
            h_im = heads[:, self.im_mask].view(b_size, 1, self.smaller_dim)

            t_re = tails[:, :, self.real_mask].transpose(2, 1)
            t_im = tails[:, :, self.im_mask].transpose(2, 1)

        else:
            h_re = heads[:, :, self.real_mask]
            h_im = heads[:, :, self.im_mask]

            t_re = tails[:, self.real_mask].view(b_size, self.smaller_dim, 1)
            t_im = tails[:, self.im_mask].view(b_size, self.smaller_dim, 1)

        re = matmul(h_re, matmul(r_re, t_re) - matmul(r_im, t_im)).view(b_size, -1)
        im = matmul(h_im, matmul(r_re, t_im) - matmul(r_im, t_re)).view(b_size, -1)

        return re + im

    def normalize_parameters(self):
        pass


class AnalogyModel(DistMultModel):
    """Implementation of ANALOGY model detailed in 2017 paper by Liu et al..\
    According to their remark in the implementation details, the number of scalars on\
    the diagonal of each relation-specific matrix is by default set to be half the embedding\
    dimension.

    References
    ----------
    * Hanxiao Liu, Yuexin Wu, and Yiming Yang.
      Analogical Inference for Multi-Relational Embeddings.
      arXiv :1705.02426 [cs], May 2017. arXiv : 1705.02426.
      https://arxiv.org/abs/1705.02426

    Parameters
    ----------
    ent_emb_dim: int
        Dimension of embedding space.
    n_entities: int
        Number of entities in the current data set.
    n_relations: int
        Number of relations in the current data set.
    scalar_share: float
        Share of the diagonal elements of the relation-specific matrices to be scalars. By default\
        it is set to half according to the original paper.

    Attributes
    ----------
    ent_emb_dim: int
        Dimension of the embedding of entities
    number_entities: int
        Number of entities in the current data set.
    entity_embeddings: torch.nn.Embedding, shape = (number_entities, ent_emb_dim)
        Contains the embeddings of the entities. It is initialized with Xavier uniform and then\
         normalized.
    relation_vectors: torch Parameter, shape = (number_relations, ent_emb_dim)
        Contains the vectors to build almost-diagonal matrices of the relations. It is initialized
        with Xavier uniform.
    number_scalars: int
        Number of diagonal elements of the relation-specific matrices to be scalars. By default\
        it is set to half according to the original paper.
    smaller_dim: int
        Number of 2x2 matrices on the diagonals of relation-specific matrices.
    """

    def __init__(self, ent_emb_dim, n_entities, n_relations, scalar_share=0.5):
        try:
            assert ent_emb_dim % 2 == 0
        except AssertionError:
            raise WrongDimensionError('Embedding dimension should be pair.')

        super().__init__(ent_emb_dim, n_entities, n_relations)

        self.number_scalars = int(self.ent_emb_dim * scalar_share)

        if (self.ent_emb_dim - self.number_scalars) % 2 == 1:
            # the diagonal blocks are 2x2 so this dimension needs to be pair
            self.number_scalars -= 1
        self.smaller_dim = int((self.ent_emb_dim - self.number_scalars) / 2)

        self.scalar_mask = get_mask(self.ent_emb_dim, 0, self.number_scalars)
        self.real_mask = get_mask(self.ent_emb_dim, self.number_scalars,
                                  self.number_scalars + self.smaller_dim)
        self.im_mask = get_mask(self.ent_emb_dim, self.number_scalars + self.smaller_dim,
                                self.ent_emb_dim)

        assert (self.real_mask.sum() == self.im_mask.sum() == self.smaller_dim)
        assert (self.scalar_mask.sum() == self.real_mask.sum() * 2)

    def compute_product(self, heads, tails, rel_mat):
        """Compute the matrix product h^tRt with proper reshapes. It can do the batch matrix
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
        product: torch.Tensor, shape = (b_size, 1) or (b_size, self.number_entities), dtype = float
            Tensor containing the matrix products h^t.W.t for each sample of the batch.

        """
        b_size = len(heads)
        r_scalar = rel_mat[:, self.scalar_mask][:, :, self.scalar_mask]
        r_re = rel_mat[:, self.real_mask][:, :, self.real_mask]
        r_im = rel_mat[:, self.im_mask][:, :, self.im_mask]

        if len(heads.shape) == 2 and len(tails.shape) == 2:
            h_scalar = heads[:, self.scalar_mask].view(b_size, 1, self.number_scalars)
            h_re = heads[:, self.real_mask].view(b_size, 1, self.smaller_dim)
            h_im = heads[:, self.im_mask].view(b_size, 1, self.smaller_dim)

            t_scalar = tails[:, self.scalar_mask].view(b_size, self.number_scalars, 1)
            t_re = tails[:, self.real_mask].view(b_size, self.smaller_dim, 1)
            t_im = tails[:, self.im_mask].view(b_size, self.smaller_dim, 1)

        elif len(heads.shape) == 2 and len(tails.shape) == 3:
            h_scalar = heads[:, self.scalar_mask].view(b_size, 1, self.number_scalars)
            h_re = heads[:, self.real_mask].view(b_size, 1, self.smaller_dim)
            h_im = heads[:, self.im_mask].view(b_size, 1, self.smaller_dim)

            t_scalar = tails[:, :, self.scalar_mask].transpose(2, 1)
            t_re = tails[:, :, self.real_mask].transpose(2, 1)
            t_im = tails[:, :, self.im_mask].transpose(2, 1)

        else:
            h_scalar = heads[:, :, self.scalar_mask]
            h_re = heads[:, :, self.real_mask]
            h_im = heads[:, :, self.im_mask]

            t_scalar = tails[:, self.scalar_mask].view(b_size, self.number_scalars, 1)
            t_re = tails[:, self.real_mask].view(b_size, self.smaller_dim, 1)
            t_im = tails[:, self.im_mask].view(b_size, self.smaller_dim, 1)

        scalar = matmul(matmul(h_scalar, r_scalar), t_scalar).view(b_size, -1)
        re = matmul(h_re, matmul(r_re, t_re) - matmul(r_im, t_im)).view(b_size, -1)
        im = matmul(h_im, matmul(r_re, t_im) - matmul(r_im, t_re)).view(b_size, -1)

        return scalar + re + im

    def normalize_parameters(self):
        """According to original paper, no normalization should be done on the parameters.

        """
        pass
