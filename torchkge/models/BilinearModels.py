# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""

from torch import matmul, cat
from torch.nn import Embedding
from torch.nn.functional import normalize
from torch.nn.init import xavier_uniform_

from torchkge.models import BilinearModel


class RESCALModel(BilinearModel):
    """Implementation of RESCAL model detailed in 2011 paper by Nickel et al..\
    In the original paper, optimization is done using Alternating Least Squares (ALS). Here we use\
    iterative gradient descent optimization. This class inherits from the
    :class:`torchkge.models.interfaces.Model` interface. It then has its attributes as well.

    References
    ----------
    * Maximilian Nickel, Volker Tresp, and Hans-Peter Kriegel.
      `A Three-way Model for Collective Learning on Multi-relational Data.
      <https://dl.acm.org/citation.cfm?id=3104584>`_
      In Proceedings of the 28th International Conference on Machine Learning, 2011.


    Parameters
    ----------
    emb_dim: int
        Dimension of embedding space.
    n_entities: int
        Number of entities in the current data set.
    n_relations: int
        Number of relations in the current data set.

    Attributes
    ----------
    ent_emb: torch.nn.Embedding, shape: (number_entities, emb_dim)
        Contains the embeddings of the entities. It is initialized with Xavier uniform and then\
         normalized.
    rel_mat: torch.nn.Parameter, shape: (number_relations, emb_dim * emb_dim)
        Contains the matrices of the relations. It is initialized with Xavier uniform.

    """

    def __init__(self, emb_dim, n_entities, n_relations):
        super().__init__(emb_dim, n_entities, n_relations)

        # initialize embedding objects
        self.ent_emb = Embedding(self.n_ent, self.emb_dim)
        self.rel_mat = Embedding(self.n_rel, self.emb_dim * self.emb_dim)

        xavier_uniform_(self.ent_emb.weight.data)
        xavier_uniform_(self.rel_mat.weight.data)

        # normalize the embeddings
        self.normalize_parameters()

    def scoring_function(self, h_idx, t_idx, r_idx):
        h = normalize(self.ent_emb(h_idx), p=2, dim=1)
        t = normalize(self.ent_emb(t_idx), p=2, dim=1)
        r = self.rel_mat(r_idx).view(-1, self.emb_dim, self.emb_dim)

        return (matmul(h.view(-1, 1, self.emb_dim), r).view(-1, self.emb_dim) * t).sum(dim=1)

    def normalize_parameters(self):  # TODO
        self.ent_emb.weight.data = normalize(self.ent_emb.weight.data, p=2, dim=1)

    def lp_batch_scoring_function(self, h, t, r):
        b_size = h.shape[0]

        if len(h.shape) == 2 and len(t.shape) == 3:
            # this is the tail completion case in link prediction
            h = h.view(b_size, 1, self.emb_dim)
            return (matmul(h, r).view(b_size, self.emb_dim, 1) * t.transpose(1, 2)).sum(dim=1)
        else:
            # this is the head completion case in link prediction
            t = t.view(b_size, self.emb_dim, 1)
            return (h.transpose(1, 2) * matmul(r, t)).sum(dim=1)

    def lp_get_emb_cand(self, h_idx, t_idx, r_idx):
        b_size = h_idx.shape[0]

        h_emb = self.ent_emb(h_idx)
        t_emb = self.ent_emb(t_idx)
        r_mat = self.rel_mat(r_idx).view(-1, self.emb_dim, self.emb_dim)

        candidates = self.ent_emb.weight.data.view(1, self.n_ent, self.emb_dim).expand(b_size, self.n_ent, self.emb_dim)

        return h_emb, t_emb, candidates, r_mat


class DistMultModel(BilinearModel):
    """Implementation of DistMult model detailed in 2014 paper by Yang et al.. This class inherits from the
    :class:`torchkge.models.BilinearModels.RESCALModel` class interpreted as an interface.
    It then has its attributes as well.

    References
    ----------
    * Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng.
      `Embedding Entities and Relations for Learning and Inference in Knowledge Bases.
      <https://arxiv.org/abs/1412.6575>`_
      arXiv :1412.6575 [cs], December 2014.


    Parameters
    ----------
    emb_dim: int
        Dimension of embedding space.
    n_entities: int
        Number of entities in the current data set.
    n_relations: int
        Number of relations in the current data set.

    Attributes
    ----------
    ent_emb: `torch.nn.Embedding`, shape: (number_relations, emb_dim)
        Contains the vectors to build diagonal matrices of the relations. It is initialized with
        Xavier uniform.

    """

    def __init__(self, emb_dim, n_entities, n_relations):
        super().__init__(emb_dim, n_entities, n_relations)
        self.emb_dim = emb_dim

        self.ent_emb = Embedding(self.n_ent, self.emb_dim)
        self.rel_emb = Embedding(self.n_rel, self.emb_dim)

        xavier_uniform_(self.ent_emb.weight.data)
        xavier_uniform_(self.rel_emb.weight.data)

    def scoring_function(self, h_idx, t_idx, r_idx):
        h = normalize(self.ent_emb(h_idx), p=2, dim=1)
        t = normalize(self.ent_emb(t_idx), p=2, dim=1)
        r = self.rel_emb(r_idx)

        return (h * r * t).sum(dim=1)

    def normalize_parameters(self):  # TODO
        self.ent_emb.weight.data = normalize(self.ent_emb.weight.data, p=2, dim=1)

    def lp_batch_scoring_function(self, h, t, r):
        b_size = h.shape[0]

        if len(h.shape) == 2 and len(t.shape) == 3:
            # this is the tail completion case in link prediction
            return ((h * r).view(b_size, self.emb_dim, 1) * t.transpose(1, 2)).sum(dim=1)
        else:
            # this is the head completion case in link prediction
            return (h.transpose(1, 2) * (r * t).view(b_size, self.emb_dim, 1)).sum(dim=1)

    def lp_get_emb_cand(self, h_idx, t_idx, r_idx):
        b_size = h_idx.shape[0]

        h_emb = self.ent_emb(h_idx)
        t_emb = self.ent_emb(t_idx)
        r_emb = self.rel_emb(r_idx)

        candidates = self.ent_emb.weight.data.view(1, self.n_ent, self.emb_dim).expand(b_size, self.n_ent, self.emb_dim)

        return h_emb, t_emb, candidates, r_emb


class HolEModel(BilinearModel):
    """Implementation of HolE model detailed in 2015 paper by Nickel et al.. This class inherits from the
    :class:`torchkge.models.BilinearModels.RESCALModel` class interpreted as an interface.
    It then has its attributes as well.

    References
    ----------
    * Maximilian Nickel, Lorenzo Rosasco, and Tomaso Poggio.
      `Holographic Embeddings of Knowledge Graphs.
      <https://arxiv.org/abs/1510.04935>`_
      arXiv :1510.04935 [cs, stat], October 2015.

    Parameters
    ----------
    emb_dim: int
        Dimension of embedding space.
    n_entities: int
        Number of entities in the current data set.
    n_relations: int
        Number of relations in the current data set.

    Attributes
    ----------
    emb_dim: int
        Dimension of the embedding of entities
    rel_emb: torch Parameter, shape: (number_relations, emb_dim)
        Contains the vectors to build circular matrices of the relations. It is initialized
        with Xavier uniform.

    """

    def __init__(self, emb_dim, n_entities, n_relations):
        super().__init__(emb_dim, n_entities, n_relations)

        self.ent_emb = Embedding(self.n_ent, self.emb_dim)
        self.rel_emb = Embedding(self.n_rel, self.emb_dim)

        xavier_uniform_(self.ent_emb.weight.data)
        xavier_uniform_(self.rel_emb.weight.data)

    def scoring_function(self, h_idx, t_idx, r_idx):
        h = normalize(self.ent_emb(h_idx), p=2, dim=1)
        t = normalize(self.ent_emb(t_idx), p=2, dim=1)
        r = self.get_rolling_matrix(self.rel_emb(r_idx))

        return (matmul(h.view(-1, 1, self.emb_dim), r).view(-1, self.emb_dim) * t).sum(dim=1)

    @staticmethod
    def get_rolling_matrix(x):
        """Build a rolling matrix.

        Parameters
        ----------
        x: `torch.Tensor`, shape: (b_size, dim)

        Returns
        -------
        mat: `torch.Tensor`, shape: (b_size, dim, dim)
            Rolling matrix such that mat[i,j] = x[i - j mod(dim)]
        """
        b_size, dim = x.shape
        x = x.view(b_size, 1, dim)
        return cat([x.roll(i, dims=2) for i in range(dim)], dim=1)

    def normalize_parameters(self):
        pass  # TODO

    def lp_batch_scoring_function(self, h, t, r):
        b_size = h.shape[0]

        if len(h.shape) == 2 and len(t.shape) == 3:
            # this is the tail completion case in link prediction
            h = h.view(b_size, 1, self.emb_dim)
            return (matmul(h, r).view(b_size, self.emb_dim, 1) * t.transpose(1, 2)).sum(dim=1)
        else:
            # this is the head completion case in link prediction
            t = t.view(b_size, self.emb_dim, 1)
            return (h.transpose(1, 2) * matmul(r, t)).sum(dim=1)

    def lp_get_emb_cand(self, h_idx, t_idx, r_idx):
        """

        Parameters
        ----------
        h_idx: `torch.Tensor`, shape: (b_size,), dtype: `torch.long`
            Tensor containing indices of current head entities.
        t_idx: `torch.Tensor`, shape: (b_size,), dtype: `torch.long`
            Tensor containing indices of current tail entities.
        r_idx: `torch.Tensor`, shape: (b_size,), dtype: `torch.long`
            Tensor containing indices of current relations.

        Returns
        -------
        h_emb: `torch.Tensor`, shape: (b_size, emb_dim), dtype: `torch.float`
            Tensor containing embeddings of current head entities.
        t_emb: `torch.Tensor`, shape: (b_size, emb_dim), dtype: `torch.float`
            Tensor containing embeddings of current tail entities.
        r_mat: `torch.Tensor`, shape: (b_size, emb_dim, emb_dim), dtype: `torch.float`
            Tensor containing matrices of current relations.
        candidates: `torch.Tensor`, shape: (b_size, number_entities, emb_dim), dtype: `torch.float`
            Tensor containing all entities as candidates for each sample of the batch.

        """
        b_size = h_idx.shape[0]
        h_emb = self.ent_emb(h_idx)
        t_emb = self.ent_emb(t_idx)
        r_mat = self.get_rolling_matrix(self.rel_emb(r_idx))

        candidates = self.ent_emb.weight.data.view(1, self.n_ent, self.emb_dim).expand(b_size, self.n_ent, self.emb_dim)

        return h_emb, t_emb, candidates, r_mat


class ComplExModel(BilinearModel):
    """Implementation of ComplEx model detailed in 2016 paper by Trouillon et al.. This class inherits from the
    :class:`torchkge.models.BilinearModels.DistMultModel` class interpreted as an interface.
    It then has its attributes as well.

    References
    ----------
    * Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier, and Guillaume Bouchard.
      `Complex Embeddings for Simple Link Prediction.
      <https://arxiv.org/abs/1606.06357>`_
      arXiv :1606.06357 [cs, stat], June 2016.

    Parameters
    ----------
    emb_dim: int
        Dimension of embedding space.
    n_entities: int
        Number of entities in the current data set.
    n_relations: int
        Number of relations in the current data set.

    Attributes
    ----------
    smaller_dim: int
        Number of 2x2 matrices on the diagonals of relation-specific matrices.
    """

    def __init__(self, emb_dim, n_entities, n_relations):
        super().__init__(emb_dim, n_entities, n_relations)
        self.re_ent_emb = Embedding(self.n_ent, self.emb_dim)
        self.im_ent_emb = Embedding(self.n_ent, self.emb_dim)
        self.re_rel_emb = Embedding(self.n_rel, self.emb_dim)
        self.im_rel_emb = Embedding(self.n_rel, self.emb_dim)

        xavier_uniform_(self.re_ent_emb.weight.data)
        xavier_uniform_(self.im_ent_emb.weight.data)
        xavier_uniform_(self.re_rel_emb.weight.data)
        xavier_uniform_(self.im_rel_emb.weight.data)

    def scoring_function(self, h_idx, t_idx, r_idx):
        """Compute the real part of the Hermitian product :math:`\\Re(h^T \\cdot diag(r) \\cdot \\bar{t})`
        for each sample of the batch.

        Parameters
        ----------
        h_idx:
        t_idx:
        r_idx:

        Returns
        -------
        product: `torch.Tensor`, shape: (b_size), dtype: `torch.float`
            Tensor containing the scoring function.

        """

        re_h, im_h = self.re_ent_emb(h_idx), self.im_ent_emb(h_idx)
        re_t, im_t = self.re_ent_emb(t_idx), self.im_ent_emb(t_idx)
        re_r, im_r = self.re_rel_emb(r_idx), self.im_rel_emb(r_idx)

        return (re_h * (re_r * re_t + im_r * im_t) + im_h * (re_r * im_t - im_r * re_t)).sum(dim=1)

    def normalize_parameters(self):
        pass  # TODO

    def lp_batch_scoring_function(self, h, t, r):
        re_h, im_h = h[0], h[1]
        re_t, im_t = t[0], t[1]
        re_r, im_r = r[0], r[1]
        b_size = re_h.shape[0]

        if len(re_h.shape) == 2 and len(re_t.shape) == 3:
            # this is the tail completion case in link prediction
            return ((re_h * re_r).view(b_size, self.emb_dim, 1) * re_t.transpose(1, 2)
                    + (re_h * im_r).view(b_size, self.emb_dim, 1) * im_t.transpose(1, 2)
                    + (im_h * re_r).view(b_size, self.emb_dim, 1) * im_t.transpose(1, 2)
                    - (im_h * im_r).view(b_size, self.emb_dim, 1) * re_t.transpose(1, 2)).sum(dim=1)

        if len(re_h.shape) == 3 and len(re_t.shape) == 2:
            # this is the head completion case in link prediction
            return (re_h.transpose(1, 2) * (re_r * re_t + im_r * im_t).view(b_size, self.emb_dim, 1)
                    + im_h.transpose(1, 2) * (re_r * im_t - im_r * re_t).view(b_size, self.emb_dim, 1)).sum(dim=1)

    def lp_get_emb_cand(self, h_idx, t_idx, r_idx):
        """

        Parameters
        ----------
        h_idx: `torch.Tensor`, shape: (b_size,), dtype: `torch.long`
            Tensor containing indices of current head entities.
        t_idx: `torch.Tensor`, shape: (b_size,), dtype: `torch.long`
            Tensor containing indices of current tail entities.
        r_idx: `torch.Tensor`, shape: (b_size,), dtype: `torch.long`
            Tensor containing indices of current relations.

        Returns
        -------
        h_emb: `torch.Tensor`, shape: (b_size, emb_dim), dtype: `torch.float`
            Tensor containing embeddings of current head entities.
        t_emb: `torch.Tensor`, shape: (b_size, emb_dim), dtype: `torch.float`
            Tensor containing embeddings of current tail entities.
        r_mat: `torch.Tensor`, shape: (b_size, emb_dim, emb_dim), dtype: `torch.float`
            Tensor containing matrices of current relations.
        candidates: `torch.Tensor`, shape: (b_size, number_entities, emb_dim), dtype: `torch.float`
            Tensor containing all entities as candidates for each sample of the batch.

        """
        b_size = h_idx.shape[0]

        re_candidates = self.re_ent_emb.weight.data.view(1, self.n_ent, self.emb_dim)
        re_candidates = re_candidates.expand(b_size, self.n_ent, self.emb_dim)

        im_candidates = self.im_ent_emb.weight.data.view(1, self.n_ent, self.emb_dim)
        im_candidates = im_candidates.expand(b_size, self.n_ent, self.emb_dim)

        re_h, im_h = self.re_ent_emb(h_idx), self.im_ent_emb(h_idx)
        re_t, im_t = self.re_ent_emb(t_idx), self.im_ent_emb(t_idx)
        re_r, im_r = self.re_rel_emb(r_idx), self.im_rel_emb(r_idx)

        return (re_h, im_h), (re_t, im_t), (re_candidates, im_candidates), (re_r, im_r)


class AnalogyModel(BilinearModel):
    """Implementation of ANALOGY model detailed in 2017 paper by Liu et al..\
    According to their remark in the implementation details, the number of scalars on\
    the diagonal of each relation-specific matrix is by default set to be half the embedding\
    dimension. This class inherits from the
    :class:`torchkge.models.BilinearModels.DistMultModel` class interpreted as an interface.
    It then has its attributes as well.

    References
    ----------
    * Hanxiao Liu, Yuexin Wu, and Yiming Yang.
      `Analogical Inference for Multi-Relational Embeddings.
      <https://arxiv.org/abs/1705.02426>`_
      arXiv :1705.02426 [cs], May 2017.

    Parameters
    ----------
    emb_dim: int
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
    scalar_dim: int
        Number of diagonal elements of the relation-specific matrices to be scalars. By default\
        it is set to half according to the original paper.
    complex_dim: int
        Number of 2x2 matrices on the diagonals of relation-specific matrices.
    """

    def __init__(self, emb_dim, n_entities, n_relations, scalar_share=0.5):
        super().__init__(emb_dim, n_entities, n_relations)

        self.scalar_dim = int(self.emb_dim * scalar_share)
        self.complex_dim = int((self.emb_dim - self.scalar_dim))

        self.sc_ent_emb = Embedding(self.n_ent, self.scalar_dim)
        self.re_ent_emb = Embedding(self.n_ent, self.complex_dim)
        self.im_ent_emb = Embedding(self.n_ent, self.complex_dim)

        self.sc_rel_emb = Embedding(self.n_rel, self.scalar_dim)
        self.re_rel_emb = Embedding(self.n_rel, self.complex_dim)
        self.im_rel_emb = Embedding(self.n_rel, self.complex_dim)

        xavier_uniform_(self.sc_ent_emb.weight.data)
        xavier_uniform_(self.re_ent_emb.weight.data)
        xavier_uniform_(self.im_ent_emb.weight.data)
        xavier_uniform_(self.sc_rel_emb.weight.data)
        xavier_uniform_(self.re_rel_emb.weight.data)
        xavier_uniform_(self.im_rel_emb.weight.data)

    def scoring_function(self, h_idx, t_idx, r_idx):
        sc_h, re_h, im_h = self.sc_ent_emb(h_idx), self.re_ent_emb(h_idx), self.im_ent_emb(h_idx)
        sc_t, re_t, im_t = self.sc_ent_emb(t_idx), self.re_ent_emb(t_idx), self.im_ent_emb(t_idx)
        sc_r, re_r, im_r = self.sc_rel_emb(r_idx), self.re_rel_emb(r_idx), self.im_rel_emb(r_idx)

        return (sc_h * sc_r * sc_t + re_h * (re_r * re_t + im_r * im_t) + im_h * (re_r * im_t - im_r * re_t)).sum(dim=1)

    def normalize_parameters(self):  # TODO
        """According to original paper, no normalization should be done on the parameters.

        """
        pass

    def lp_batch_scoring_function(self, h, t, r):
        sc_h, re_h, im_h = h[0], h[1], h[2]
        sc_t, re_t, im_t = t[0], t[1], t[2]
        sc_r, re_r, im_r = r[0], r[1], r[2]
        b_size = re_h.shape[0]

        if len(re_h.shape) == 2 and len(re_t.shape) == 3:
            # this is the tail completion case in link prediction
            return (((sc_h * sc_r).view(b_size, self.scalar_dim, 1) * sc_t.transpose(1, 2)).sum(dim=1)
                    + ((re_h * re_r).view(b_size, self.complex_dim, 1) * re_t.transpose(1, 2)).sum(dim=1)
                    + ((re_h * im_r).view(b_size, self.complex_dim, 1) * im_t.transpose(1, 2)).sum(dim=1)
                    + ((im_h * re_r).view(b_size, self.complex_dim, 1) * im_t.transpose(1, 2)).sum(dim=1)
                    - ((im_h * im_r).view(b_size, self.complex_dim, 1) * re_t.transpose(1, 2)).sum(dim=1))

        if len(re_h.shape) == 3 and len(re_t.shape) == 2:
            # this is the head completion case in link prediction
            return ((sc_h.transpose(1, 2) * (sc_r * sc_t).view(b_size, self.scalar_dim, 1)).sum(dim=1)
                    + (re_h.transpose(1, 2) * (re_r * re_t + im_r * im_t).view(b_size, self.complex_dim, 1)).sum(dim=1)
                    + (im_h.transpose(1, 2) * (re_r * im_t - im_r * re_t).view(b_size, self.complex_dim, 1)).sum(dim=1))

    def lp_get_emb_cand(self, h_idx, t_idx, r_idx):
        b_size = h_idx.shape[0]

        sc_candidates = self.sc_ent_emb.weight.data.view(1, self.n_ent, self.scalar_dim)
        sc_candidates = sc_candidates.expand(b_size, self.n_ent, self.scalar_dim)

        re_candidates = self.re_ent_emb.weight.data.view(1, self.n_ent, self.complex_dim)
        re_candidates = re_candidates.expand(b_size, self.n_ent, self.complex_dim)

        im_candidates = self.im_ent_emb.weight.data.view(1, self.n_ent, self.complex_dim)
        im_candidates = im_candidates.expand(b_size, self.n_ent, self.complex_dim)

        sc_h, re_h, im_h = self.sc_ent_emb(h_idx), self.re_ent_emb(h_idx), self.im_ent_emb(h_idx)
        sc_t, re_t, im_t = self.sc_ent_emb(t_idx), self.re_ent_emb(t_idx), self.im_ent_emb(t_idx)
        sc_r, re_r, im_r = self.sc_rel_emb(r_idx), self.re_rel_emb(r_idx), self.im_rel_emb(r_idx)

        return (sc_h, re_h, im_h), (sc_t, re_t, im_t), (sc_candidates, re_candidates, im_candidates), (sc_r, re_r, im_r)
