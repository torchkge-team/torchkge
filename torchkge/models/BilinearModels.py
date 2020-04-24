# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""

from torch import empty, matmul, diag_embed, cat
from torch.nn import Embedding, Parameter
from torch.nn.functional import normalize
from torch.nn.init import xavier_uniform_

from torchkge.models import Model, BilinearModel
from torchkge.utils import get_rank, get_mask, get_true_targets
from torchkge.exceptions import WrongDimensionError


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
        super(RESCALModel, self).__init__(emb_dim, n_entities, n_relations)

        # initialize embedding objects
        self.ent_emb = Embedding(self.n_ent, self.emb_dim)
        self.rel_mat = Embedding(self.n_rel, self.emb_dim * self.emb_dim)

        xavier_uniform_(self.ent_emb.weight.data)
        xavier_uniform_(self.rel_mat.weight.data)

        # normalize the embeddings
        self.normalize_parameters()

    def scoring_function(self, h_idx, t_idx, r_idx):
        return self.compute_product(normalize(self.ent_emb(h_idx), p=2, dim=1),
                                    normalize(self.ent_emb(t_idx), p=2, dim=1),
                                    self.rel_mat(r_idx).view(-1, self.ent_emb_dim, self.ent_emb_dim),
                                    self.emb_dim)

    def normalize_parameters(self):
        self.ent_emb.weight.data = normalize(self.ent_emb.weight.data, p=2, dim=1)

    def evaluation_helper(self, h_idx, t_idx, r_idx):
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
        h_emb, t_emb, candidates = self.get_head_tail_candidates(h_idx, t_idx)
        r_mat = self.rel_mat(r_idx).view(-1, self.ent_emb_dim, self.ent_emb_dim)

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
    ent_emb: `torch.nn.Embedding`, shape: (number_relations, ent_emb_dim)
        Contains the vectors to build diagonal matrices of the relations. It is initialized with
        Xavier uniform.

    """

    def __init__(self, emb_dim, n_entities, n_relations):
        super(DistMultModel, self).__init__(emb_dim, n_entities, n_relations)
        self.emb_dim = emb_dim

        self.ent_emb = Embedding(self.n_ent, self.emb_dim)
        self.rel_emb = Embedding(self.n_rel, self.emb_dim)

        xavier_uniform_(self.ent_emb.weight.data)
        xavier_uniform_(self.rel_emb.weight.data)

    def scoring_function(self, h_idx, t_idx, r_idx):
        return self.compute_product(normalize(self.ent_emb(h_idx), p=2, dim=1),
                                    normalize(self.ent_emb(t_idx), p=2, dim=1),
                                    self.rel_emb(r_idx),
                                    self.emb_dim)

    def normalize_parameters(self):
        self.ent_emb.weight.data = normalize(self.ent_emb.weight.data, p=2, dim=1)

    @staticmethod
    def compute_product(h, t, r, emb_dim):
        b_size = h.shape[0]

        if len(h.shape) == 2 and len(t.shape) == 2:
            # this is the easy forward case
            return (h * r * t).sum(dim=1)

        elif len(h.shape) == 2 and len(t.shape) == 3:
            # this is the tail completion case in link prediction
            return ((h * r).view(b_size, emb_dim, 1) * t.transpose(1, 2)).sum(dim=1)
        else:
            # this is the head completion case in link prediction
            return ((r * t).view(b_size, emb_dim, 1) * h.transpose(1, 2)).sum(dim=1)

    def evaluation_helper(self, h_idx, t_idx, r_idx):
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
        h_emb, t_emb, candidates = self.get_head_tail_candidates(h_idx, t_idx)
        r_mat = diag_embed(self.relation_vectors[r_idx])

        return h_emb, t_emb, candidates, r_mat


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
    relation_vectors: torch Parameter, shape: (number_relations, emb_dim)
        Contains the vectors to build circular matrices of the relations. It is initialized
        with Xavier uniform.

    """

    def __init__(self, emb_dim, n_entities, n_relations):
        super(HolEModel, self).__init__(emb_dim, n_entities, n_relations)

        self.ent_emb = Embedding(self.n_ent, self.emb_dim)
        self.rel_emb = Embedding(self.n_rel, self.emb_dim)

        xavier_uniform_(self.ent_emb.weight.data)
        xavier_uniform_(self.rel_emb.weight.data)

    def scoring_function(self, h_idx, t_idx, r_idx):
        return self.compute_product(normalize(self.ent_emb(h_idx), p=2, dim=1),
                                    normalize(self.ent_emb(t_idx), p=2, dim=1),
                                    self.get_rolling_matrix(self.rel_emb(r_idx)),
                                    self.emb_dim)

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

    def evaluation_helper(self, h_idx, t_idx, r_idx):
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
        h_emb, t_emb, candidates = self.get_head_tail_candidates(h_idx, t_idx)
        r_mat = self.get_rolling_matrix(self.rel_emb(r_idx))

        return h_emb, t_emb, candidates, r_mat

    def normalize_parameters(self):
        pass


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
        super(ComplExModel, self).__init__(emb_dim, n_entities, n_relations)
        self.re_ent_emb = Embedding(self.n_ent, self.emb_dim)
        self.im_ent_emb = Embedding(self.n_ent, self.emb_dim)
        self.re_rel_emb = Embedding(self.n_ent, self.emb_dim)
        self.im_rel_emb = Embedding(self.n_ent, self.emb_dim)

        xavier_uniform_(self.re_ent_emb.weight.data)
        xavier_uniform_(self.im_ent_emb.weight.data)
        xavier_uniform_(self.re_rel_emb.weight.data)
        xavier_uniform_(self.im_rel_emb.weight.data)

    def scoring_function(self, h_idx, t_idx, r_idx):
        return self.compute_product((self.re_ent_emb(h_idx), self.im_ent_emb(h_idx)),
                                    (self.re_ent_emb(t_idx), self.im_ent_emb(t_idx)),
                                    (self.re_rel_emb(r_idx), self.im_rel_emb(r_idx)),
                                    self.emb_dim)

    @staticmethod
    def compute_product(h, t, r, emb_dim):
        """Compute the matrix product h^tRt with proper reshapes. It can do the batch matrix
        product both in the forward pass and in the evaluation pass with one matrix containing
        all candidates.

        Parameters TODO
        ----------
        h: tuple[`torch.Tensor`, `torch.Tensor`], shape: (b_size, self.emb_dim) or (b_size, self.number_entities,\
        self.emb_dim), dtype: `torch.float`
            Tensor containing embeddings of current head entities or candidates.
        t: tuple(`torch.Tensor`, `torch.Tensor`), shape: (b_size, self.emb_dim) or (b_size, self.number_entities,\
        self.emb_dim), dtype: `torch.float`
            Tensor containing embeddings of current tail entities or canditates.
        r: tuple(`torch.Tensor`, `torch.Tensor`), shape: (b_size, self.emb_dim, self.emb_dim), dtype: `torch.float`
            Tensor containing relation matrices for current relations.

        Returns
        -------
        product: `torch.Tensor`, shape: (b_size, 1) or (b_size, self.number_entities), dtype: `torch.float`
            Tensor containing the matrix products h^t.W.t for each sample of the batch.

        """
        re_h, im_h = h[0], h[1]
        re_t, im_t = t[0], t[1]
        re_r, im_r = r[0], r[1]

        if len(re_h.shape) == 2 and len(re_t.shape) == 2:
            # this is the easy forward case
            return (re_h * (re_r * re_t + im_r + im_t) + im_h * (re_r * im_t - im_r * re_t)).sum(dim=1)

        if len(re_h.shape) == 2 and len(re_t.shape) == 3:
            # this is the tail completion case in link prediction
            return ((re_h * re_r).view(-1, emb_dim, 1) * re_t.transpose(1, 2)
                    + (re_h * im_r).view(-1, emb_dim, 1) * im_t.transpose(1, 2)
                    + (im_h * re_r).view(-1, emb_dim, 1) * im_t.transpose(1, 2)
                    - (im_h * im_r).view(-1, emb_dim, 1) * re_t.transpose(1, 2)).sum(dim=1)

        if len(re_h.shape) == 3 and len(re_t.shape) == 2:
            # this is the head completion case in link prediction
            return ((re_t * re_r).view(-1, emb_dim, 1) * re_h.transpose(1, 2)
                    + (re_t * im_r).view(-1, emb_dim, 1) * im_h.transpose(1, 2)
                    + (im_t * re_r).view(-1, emb_dim, 1) * im_h.transpose(1, 2)
                    - (im_t * im_r).view(-1, emb_dim, 1) * re_h.transpose(1, 2)).sum(dim=1)

    def normalize_parameters(self):
        pass

    def get_head_tail_candidates(self, h_idx, t_idx):
        b_size = h_idx.shape[0]

        re_candidates = self.re_ent_emb.weight.data.view(1, self.n_ent, self.emb_dim)
        im_candidates = self.im_ent_emb.weight.data.view(1, self.n_ent, self.emb_dim)

        re_candidates = re_candidates.expand(b_size, self.n_ent, self.emb_dim)
        im_candidates = im_candidates.expand(b_size, self.n_ent, self.emb_dim)

        re_h_emb = self.re_ent_emb(h_idx)
        im_h_emb = self.im_ent_emb(h_idx)
        re_t_emb = self.re_ent_emb(t_idx)
        im_t_emb = self.im_ent_emb(t_idx)

        return (re_h_emb, im_h_emb), (re_t_emb, im_t_emb), (re_candidates, im_candidates)

    def evaluation_helper(self, h_idx, t_idx, r_idx):
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
        h_emb, t_emb, candidates = self.get_head_tail_candidates(h_idx, t_idx)
        r_mat = self.get_rolling_matrix(self.rel_emb(r_idx))

        return h_emb, t_emb, candidates, r_mat


class AnalogyModel(DistMultModel):
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
    number_scalars: int
        Number of diagonal elements of the relation-specific matrices to be scalars. By default\
        it is set to half according to the original paper.
    smaller_dim: int
        Number of 2x2 matrices on the diagonals of relation-specific matrices.
    """

    def __init__(self, emb_dim, n_entities, n_relations, scalar_share=0.5):
        try:
            assert emb_dim % 2 == 0
        except AssertionError:
            raise WrongDimensionError('Embedding dimension should be pair.')

        super().__init__(emb_dim, n_entities, n_relations)

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

    def compute_product(self, heads, tails, rel_mat):
        """Compute the matrix product h^tRt with proper reshapes. It can do the batch matrix
        product both in the forward pass and in the evaluation pass with one matrix containing
        all candidates.

        Parameters
        ----------
        heads: torch Tensor, shape: (b_size, self.emb_dim) or (b_size, self.number_entities,\
        self.emb_dim), dtype: `torch.float`
            Tensor containing embeddings of current head entities or candidates.
        tails: `torch.Tensor`, shape: (b_size, self.emb_dim) or (b_size, self.number_entities,\
        self.emb_dim), dtype: `torch.float`
            Tensor containing embeddings of current tail entities or canditates.
        rel_mat: `torch.Tensor`, shape: (b_size, self.emb_dim, self.emb_dim), dtype: `torch.float`
            Tensor containing relation matrices for current relations.

        Returns
        -------
        product: `torch.Tensor`, shape: (b_size, 1) or (b_size, self.number_entities), dtype: `torch.float`
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
        re = matmul(h_re, matmul(r_re, t_re) + matmul(r_im, t_im)).view(b_size, -1)
        im = matmul(h_im, matmul(r_re, t_im) - matmul(r_im, t_re)).view(b_size, -1)

        return scalar + re + im

    def normalize_parameters(self):
        """According to original paper, no normalization should be done on the parameters.

        """
        pass
