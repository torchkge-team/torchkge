# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""

from torch import matmul, cat
from torch.nn.functional import normalize

from ..models.interfaces import BilinearModel
from ..utils import init_embedding


class RESCALModel(BilinearModel):
    """Implementation of RESCAL model detailed in 2011 paper by Nickel et al..
    In the original paper, optimization is done using Alternating Least Squares
    (ALS). Here we use iterative gradient descent optimization. This class
    inherits from the :class:`torchkge.models.interfaces.BilinearModel`
    interface. It then has its attributes as well.

    References
    ----------
    * Maximilian Nickel, Volker Tresp, and Hans-Peter Kriegel.
      `A Three-way Model for Collective Learning on Multi-relational Data.
      <https://dl.acm.org/citation.cfm?id=3104584>`_
      In Proceedings of the 28th International Conference on Machine Learning,
      2011.


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
    ent_emb: torch.nn.Embedding, shape: (n_ent, emb_dim)
        Embeddings of the entities, initialized with Xavier uniform
        distribution and then normalized.
    rel_mat: torch.nn.Embedding, shape: (n_rel, emb_dim x emb_dim)
        Matrices of the relations, initialized with Xavier uniform
        distribution.

    """

    def __init__(self, emb_dim, n_entities, n_relations):
        super().__init__(emb_dim, n_entities, n_relations)

        # initialize embedding objects
        self.ent_emb = init_embedding(self.n_ent, self.emb_dim)
        self.rel_mat = init_embedding(self.n_rel, self.emb_dim * self.emb_dim)

        # normalize the embeddings
        self.normalize_parameters()

    def scoring_function(self, h_idx, t_idx, r_idx):
        """Compute the scoring function for the triplets given as argument:
        :math:`h^T \\cdot M_r \\cdot t`. See referenced paper for more details
        on the score. See torchkge.models.interfaces.Models for more details
        on the API.

        """
        h = normalize(self.ent_emb(h_idx), p=2, dim=1)
        t = normalize(self.ent_emb(t_idx), p=2, dim=1)
        r = self.rel_mat(r_idx).view(-1, self.emb_dim, self.emb_dim)
        hr = matmul(h.view(-1, 1, self.emb_dim), r)
        return (hr.view(-1, self.emb_dim) * t).sum(dim=1)

    def normalize_parameters(self):
        """Normalize the entity embeddings, as explained in original paper.
        This methods should be called at the end of each training epoch and at
        the end of training as well.

        """
        self.ent_emb.weight.data = normalize(self.ent_emb.weight.data,
                                             p=2, dim=1)

    def get_embeddings(self):
        """Return the embeddings of entities and matrices of relations.

        Returns
        -------
        ent_emb: torch.Tensor, shape: (n_ent, emb_dim), dtype: torch.float
            Embeddings of entities.
        rel_mat: torch.Tensor, shape: (n_rel, emb_dim, emb_dim),
        dtype: torch.float
            Matrices of relations.

        """
        self.normalize_parameters()
        return self.ent_emb.weight.data, \
            self.rel_mat.weight.data.view(-1, self.emb_dim, self.emb_dim)

    def inference_scoring_function(self, h, t, r):
        """Link prediction evaluation helper function. See
        torchkge.models.interfaces.Models for more details of the API.

        """
        b_size = h.shape[0]

        if len(h.shape) == 3:
            assert (len(t.shape) == 2) & (len(r.shape) == 3)
            # this is the head completion case in link prediction
            tr = matmul(r, t.view(b_size, self.emb_dim, 1)).view(b_size, 1, self.emb_dim)
            return (h * tr).sum(dim=2)
        elif len(t.shape) == 3:
            assert (len(h.shape) == 2) & (len(r.shape) == 3)
            # this is the tail completion case in link prediction
            hr = matmul(h.view(b_size, 1, self.emb_dim), r).view(b_size, 1, self.emb_dim)
            return (hr * t).sum(dim=2)
        elif len(r.shape) == 4:
            assert (len(h.shape) == 2) & (len(t.shape) == 2)
            # this is the relation completion case in link prediction
            h = h.view(b_size, 1, 1, self.emb_dim)
            t = t.view(b_size, 1, self.emb_dim)
            hr = matmul(h, r).view(b_size, self.n_rel, self.emb_dim)
            return (hr * t).sum(dim=2)

    def inference_prepare_candidates(self, h_idx, t_idx, r_idx, entities=True):
        """Link prediction evaluation helper function. Get entities embeddings
        and relations embeddings. The output will be fed to the
        `inference_scoring_function` method. See torchkge.models.interfaces.Models for
        more details on the API.

        """
        b_size = h_idx.shape[0]

        h_emb = self.ent_emb(h_idx)
        t_emb = self.ent_emb(t_idx)
        r_mat = self.rel_mat(r_idx).view(-1, self.emb_dim, self.emb_dim)

        if entities:
            candidates = self.ent_emb.weight.data.view(1, self.n_ent, self.emb_dim)
            candidates = candidates.expand(b_size, self.n_ent, self.emb_dim)
        else:
            candidates = self.rel_mat.weight.data.view(1, self.n_rel, self.emb_dim, self.emb_dim)
            candidates = candidates.expand(b_size, self.n_rel, self.emb_dim, self.emb_dim)

        return h_emb, t_emb, r_mat, candidates


class DistMultModel(BilinearModel):
    """Implementation of DistMult model detailed in 2014 paper by Yang et al..
    This class inherits from the
    :class:`torchkge.models.interfaces.BilinearModel` interface. It then has
    its attributes as well.

    References
    ----------
    * Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng.
      `Embedding Entities and Relations for Learning and Inference in
      Knowledge Bases. <https://arxiv.org/abs/1412.6575>`_
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
    ent_emb: torch.nn.Embedding, shape: (n_ent, emb_dim)
        Embeddings of the entities, initialized with Xavier uniform
        distribution and then normalized.
    rel_emb: torch.nn.Embedding, shape: (n_rel, emb_dim)
        Embeddings of the relations, initialized with Xavier uniform
        distribution.

    """

    def __init__(self, emb_dim, n_entities, n_relations):
        super().__init__(emb_dim, n_entities, n_relations)

        self.ent_emb = init_embedding(self.n_ent, self.emb_dim)
        self.rel_emb = init_embedding(self.n_rel, self.emb_dim)

        self.normalize_parameters()

    def scoring_function(self, h_idx, t_idx, r_idx):
        """Compute the scoring function for the triplets given as argument:
        :math:`h^T \\cdot diag(r) \\cdot t`. See referenced paper for more
        details on the score. See torchkge.models.interfaces.Models for more
        details on the API.

        """
        h = normalize(self.ent_emb(h_idx), p=2, dim=1)
        t = normalize(self.ent_emb(t_idx), p=2, dim=1)
        r = self.rel_emb(r_idx)

        return (h * r * t).sum(dim=1)

    def normalize_parameters(self):
        """Normalize the entity embeddings, as explained in original paper.
        This methods should be called at the end of each training epoch and at
        the end of training as well.

        """
        self.ent_emb.weight.data = normalize(self.ent_emb.weight.data, p=2,
                                             dim=1)

    def get_embeddings(self):
        """Return the embeddings of entities and relations.

        Returns
        -------
        ent_emb: torch.Tensor, shape: (n_ent, emb_dim), dtype: torch.float
            Embeddings of entities.
        rel_emb: torch.Tensor, shape: (n_rel, emb_dim), dtype: torch.float
            Embeddings of relations.

        """
        self.normalize_parameters()
        return self.ent_emb.weight.data, self.rel_emb.weight.data

    def inference_scoring_function(self, h, t, r):
        """Link prediction evaluation helper function. See
        torchkge.models.interfaces.Models for more details on the API.

        """
        b_size = h.shape[0]

        if len(t.shape) == 3:
            assert (len(h.shape) == 2) & (len(r.shape) == 2)
            # this is the tail completion case in link prediction
            hr = (h * r).view(b_size, 1, self.emb_dim)
            return (hr * t).sum(dim=2)
        elif len(h.shape) == 3:
            assert (len(t.shape) == 2) & (len(r.shape) == 2)
            # this is the head completion case in link prediction
            rt = (r * t).view(b_size, 1, self.emb_dim)
            return (h * rt).sum(dim=2)
        elif len(r.shape) == 3:
            assert (len(h.shape) == 2) & (len(t.shape) == 2)
            # this is the relation prediction case
            hr = (h.view(b_size, 1, self.emb_dim) * r)  # hr has shape (b_size, self.n_rel, self.emb_dim)
            return (hr * t.view(b_size, 1, self.emb_dim)).sum(dim=2)

    def inference_prepare_candidates(self, h_idx, t_idx, r_idx, entities=True):
        """Link prediction evaluation helper function. Get entities embeddings
        and relations embeddings. The output will be fed to the
        `inference_scoring_function` method. See torchkge.models.interfaces.Models for
        more details on the API.

        """
        b_size = h_idx.shape[0]

        h = self.ent_emb(h_idx)
        t = self.ent_emb(t_idx)
        r = self.rel_emb(r_idx)

        if entities:
            candidates = self.ent_emb.weight.data.view(1, self.n_ent, self.emb_dim)
            candidates = candidates.expand(b_size, self.n_ent, self.emb_dim)
        else:
            candidates = self.rel_emb.weight.data.view(1, self.n_rel, self.emb_dim)
            candidates = candidates.expand(b_size, self.n_rel, self.emb_dim)

        return h, t, r, candidates


class HolEModel(BilinearModel):
    """Implementation of HolE model detailed in 2015 paper by Nickel et al..
    This class inherits from the
    :class:`torchkge.models.interfaces.BilinearModel` interface. It then has
    its attributes as well.

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
    ent_emb: torch.nn.Embedding, shape: (n_ent, emb_dim)
        Embeddings of the entities, initialized with Xavier uniform
        distribution and then normalized.
    rel_emb: torch.nn.Embedding, shape: (n_rel, emb_dim)
        Embeddings of the relations, initialized with Xavier uniform
        distribution.

    """

    def __init__(self, emb_dim, n_entities, n_relations):
        super().__init__(emb_dim, n_entities, n_relations)

        self.ent_emb = init_embedding(self.n_ent, self.emb_dim)
        self.rel_emb = init_embedding(self.n_rel, self.emb_dim)

        self.normalize_parameters()

    def scoring_function(self, h_idx, t_idx, r_idx):
        """Compute the scoring function for the triplets given as argument:
        :math:`h^T \\cdot M_r \\cdot t` where :math:`M_r` is the rolling matrix
        built from the relation embedding `r`. See referenced paper for more
        details on the score. See torchkge.models.interfaces.Models for more
        details on the API.

        """
        h = normalize(self.ent_emb(h_idx), p=2, dim=1)
        t = normalize(self.ent_emb(t_idx), p=2, dim=1)
        r = self.get_rolling_matrix(self.rel_emb(r_idx))
        hr = matmul(h.view(-1, 1, self.emb_dim), r)
        return (hr.view(-1, self.emb_dim) * t).sum(dim=1)

    @staticmethod
    def get_rolling_matrix(x):
        """Build a rolling matrix.

        Parameters
        ----------
        x: torch.Tensor, shape: (b_size, dim)

        Returns
        -------
        mat: torch.Tensor, shape: (b_size, dim, dim)
            Rolling matrix such that mat[i,j] = x[j - i mod(dim)]
        """
        b_size, dim = x.shape
        x = x.view(b_size, 1, dim)
        return cat([x.roll(i, dims=2) for i in range(dim)], dim=1)

    def normalize_parameters(self):
        """Normalize the entity embeddings, as explained in original paper.
        This methods should be called at the end of each training epoch and at
        the end of training as well.

        """
        self.ent_emb.weight.data = normalize(self.ent_emb.weight.data, p=2,
                                             dim=1)

    def get_embeddings(self):
        """Return the embeddings of entities and relations.

        Returns
        -------
        ent_emb: torch.Tensor, shape: (n_ent, emb_dim), dtype: torch.float
            Embeddings of entities.
        rel_emb: torch.Tensor, shape: (n_rel, emb_dim), dtype: torch.float
            Embeddings of relations.

        """
        self.normalize_parameters()
        return self.ent_emb.weight.data, self.rel_emb.weight.data

    def inference_scoring_function(self, h, t, r):
        """Link prediction evaluation helper function. See
        torchkge.models.interfaces.Models for more details on the API.

        """
        b_size = h.shape[0]

        if len(t.shape) == 3:
            assert (len(h.shape) == 2) & (len(r.shape) == 3)
            # this is the tail completion case in link prediction
            h = h.view(b_size, 1, self.emb_dim)
            hr = matmul(h, r).view(b_size, self.emb_dim, 1)
            return (hr * t.transpose(1, 2)).sum(dim=1)
        elif len(h.shape) == 3:
            assert (len(t.shape) == 2) & (len(r.shape) == 3)
            # this is the head completion case in link prediction
            t = t.view(b_size, self.emb_dim, 1)
            return (h.transpose(1, 2) * matmul(r, t)).sum(dim=1)
        elif len(r.shape) == 4:
            assert (len(h.shape) == 2) & (len(t.shape) == 2)
            # this is the relation completion case in link prediction
            h = h.view(b_size, 1, 1, self.emb_dim)
            t = t.view(b_size, 1, self.emb_dim)
            hr = matmul(h, r).view(b_size, self.n_rel, self.emb_dim)
            return (hr * t).sum(dim=2)

    def inference_prepare_candidates(self, h_idx, t_idx, r_idx, entities=True):
        """Link prediction evaluation helper function. Get entities embeddings
        and relations embeddings. The output will be fed to the
        `inference_scoring_function` method. See torchkge.models.interfaces.Models
        for more details on the API.

        """
        b_size = h_idx.shape[0]
        h_emb = self.ent_emb(h_idx)
        t_emb = self.ent_emb(t_idx)
        r_mat = self.get_rolling_matrix(self.rel_emb(r_idx))

        if entities:
            candidates = self.ent_emb.weight.data.view(1, self.n_ent, self.emb_dim)
            candidates = candidates.expand(b_size, self.n_ent, self.emb_dim)
        else:
            r_mat = self.get_rolling_matrix(self.rel_emb.weight.data)  # TODO: do not recompute for each batch
            candidates = r_mat.view(1, self.n_rel, self.emb_dim, self.emb_dim)
            candidates = candidates.expand(b_size, self.n_rel, self.emb_dim, self.emb_dim)

        return h_emb, t_emb, r_mat, candidates


class ComplExModel(BilinearModel):
    """Implementation of ComplEx model detailed in 2016 paper by Trouillon et
    al.. This class inherits from the
    :class:`torchkge.models.interfaces.BilinearModel` interface. It then has
    its attributes as well.

    References
    ----------
    * Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier, and
      Guillaume Bouchard.
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
    re_ent_emb: torch.nn.Embedding, shape: (n_ent, emb_dim)
        Real part of the entities complex embeddings. Initialized with Xavier
        uniform distribution.
    im_ent_emb: torch.nn.Embedding, shape: (n_ent, emb_dim)
        Imaginary part of the entities complex embeddings. Initialized with
        Xavier uniform distribution.
    re_rel_emb: torch.nn.Embedding, shape: (n_rel, emb_dim)
        Real part of the relations complex embeddings. Initialized with Xavier
        uniform distribution.
    im_rel_emb: torch.nn.Embedding, shape: (n_rel, emb_dim)
        Imaginary part of the relations complex embeddings. Initialized with
        Xavier uniform distribution.
    """

    def __init__(self, emb_dim, n_entities, n_relations):
        super().__init__(emb_dim, n_entities, n_relations)
        self.re_ent_emb = init_embedding(self.n_ent, self.emb_dim)
        self.im_ent_emb = init_embedding(self.n_ent, self.emb_dim)
        self.re_rel_emb = init_embedding(self.n_rel, self.emb_dim)
        self.im_rel_emb = init_embedding(self.n_rel, self.emb_dim)

    def scoring_function(self, h_idx, t_idx, r_idx):
        """Compute the real part of the Hermitian product
        :math:`\\Re(h^T \\cdot diag(r) \\cdot \\bar{t})` for each sample of
        the batch. See referenced paper for more details on the score. See
        torchkge.models.interfaces.Models for more details on the API.

        """

        re_h, im_h = self.re_ent_emb(h_idx), self.im_ent_emb(h_idx)
        re_t, im_t = self.re_ent_emb(t_idx), self.im_ent_emb(t_idx)
        re_r, im_r = self.re_rel_emb(r_idx), self.im_rel_emb(r_idx)

        return (re_h * (re_r * re_t + im_r * im_t) + im_h * (
                    re_r * im_t - im_r * re_t)).sum(dim=1)

    def normalize_parameters(self):
        """According to original paper, the embeddings should not be
        normalized.

        """
        pass

    def get_embeddings(self):
        """Return the embeddings of entities and relations.

        Returns
        -------
        re_ent_emb: torch.Tensor, shape: (n_ent, emb_dim), dtype: torch.float
            Real part of embeddings of entities.
        im_ent_emb: torch.Tensor, shape: (n_ent, emb_dim), dtype: torch.float
            Imaginary part of embeddings of entities.
        re_rel_emb: torch.Tensor, shape: (n_rel, emb_dim), dtype: torch.float
            Real part of embeddings of relations.
        im_rel_emb: torch.Tensor, shape: (n_rel, emb_dim), dtype: torch.float
            Imaginary part of embeddings of relations.

        """
        self.normalize_parameters()
        return self.re_ent_emb.weight.data, self.im_ent_emb.weight.data,\
            self.re_rel_emb.weight.data, self.im_rel_emb.weight.data

    def inference_scoring_function(self, h, t, r):
        """Link prediction evaluation helper function. See
        torchkge.models.interfaces.Models for more details one the API.

        """
        re_h, im_h = h[0], h[1]
        re_t, im_t = t[0], t[1]
        re_r, im_r = r[0], r[1]
        b_size = re_h.shape[0]

        if len(re_t.shape) == 3:
            assert (len(re_h.shape) == 2) & (len(re_r.shape) == 2)
            # this is the tail completion case in link prediction
            return ((re_h * re_r - im_h * im_r).view(b_size, 1, self.emb_dim) * re_t
                    + (re_h * im_r + im_h * re_r).view(b_size, 1, self.emb_dim) * im_t).sum(dim=2)

        elif len(re_h.shape) == 3:
            assert (len(re_t.shape) == 2) & (len(re_r.shape) == 2)
            # this is the head completion case in link prediction

            return (re_h * (re_r * re_t + im_r * im_t).view(b_size, 1, self.emb_dim)
                    + im_h * (re_r * im_t - im_r * re_t).view(b_size, 1, self.emb_dim)).sum(dim=2)

        elif len(re_r.shape) == 3:
            assert (len(re_h.shape) == 2) & (len(re_t.shape) == 2)
            # this is the relation prediction case
            return ((re_h * re_t + im_h * im_t).view(b_size, 1, self.emb_dim) * re_r
                    + (re_h * im_t - im_h * re_t).view(b_size, 1, self.emb_dim) * im_r).sum(dim=2)

    def inference_prepare_candidates(self, h_idx, t_idx, r_idx, entities=True):
        """Link prediction evaluation helper function. Get entities embeddings
        and relations embeddings. The output will be fed to the
        `inference_scoring_function` method. See torchkge.models.interfaces.Models for
        more details on the API.

        """
        b_size = h_idx.shape[0]

        re_h, im_h = self.re_ent_emb(h_idx), self.im_ent_emb(h_idx)
        re_t, im_t = self.re_ent_emb(t_idx), self.im_ent_emb(t_idx)
        re_r, im_r = self.re_rel_emb(r_idx), self.im_rel_emb(r_idx)

        if entities:
            re_candidates = self.re_ent_emb.weight.data.view(1, self.n_ent, self.emb_dim)
            re_candidates = re_candidates.expand(b_size, self.n_ent, self.emb_dim)

            im_candidates = self.im_ent_emb.weight.data.view(1, self.n_ent, self.emb_dim)
            im_candidates = im_candidates.expand(b_size, self.n_ent, self.emb_dim)
        else:
            re_candidates = self.re_rel_emb.weight.data.view(1, self.n_rel, self.emb_dim)
            re_candidates = re_candidates.expand(b_size, self.n_rel, self.emb_dim)

            im_candidates = self.im_rel_emb.weight.data.view(1, self.n_rel, self.emb_dim)
            im_candidates = im_candidates.expand(b_size, self.n_rel, self.emb_dim)

        return (re_h, im_h), (re_t, im_t), (re_r, im_r), (re_candidates, im_candidates)


class AnalogyModel(BilinearModel):
    """Implementation of ANALOGY model detailed in 2017 paper by Liu et al..
    According to their remark in the implementation details, the number of
    scalars on the diagonal of each relation-specific matrix is by default set
    to be half the embedding dimension. This class inherits from the
    :class:`torchkge.models.interfaces.BilinearModel` interface. It then has
    its attributes as well.

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
        Share of the diagonal elements of the relation-specific matrices to be
        scalars. By default it is set to half according to the original paper.

    Attributes
    ----------
    scalar_dim: int
        Number of diagonal elements of the relation-specific matrices to be
        scalars. By default it is set to half the embedding dimension according
        to the original paper.
    complex_dim: int
        Number of 2x2 matrices on the diagonals of relation-specific matrices.
    sc_ent_emb: torch.nn.Embedding, shape: (n_ent, emb_dim)
        Part of the entities embeddings associated to the scalar part of the
        relation specific matrices. Initialized with Xavier uniform
        distribution.
    re_ent_emb: torch.nn.Embedding, shape: (n_ent, emb_dim)
        Real part of the entities complex embeddings. Initialized with Xavier
        uniform distribution. As explained in the authors' paper, almost
        diagonal matrices can be seen as complex matrices.
    im_ent_emb: torch.nn.Embedding, shape: (n_ent, emb_dim)
        Imaginary part of the entities complex embeddings. Initialized with
        Xavier uniform distribution. As explained in the authors' paper, almost
        diagonal matrices can be seen as complex matrices.
    sc_rel_emb: torch.nn.Embedding, shape: (n_rel, emb_dim)
        Part of the entities embeddings associated to the scalar part of the
        relation specific matrices. Initialized with Xavier uniform
        distribution.
    re_rel_emb: torch.nn.Embedding, shape: (n_rel, emb_dim)
        Real part of the relations complex embeddings. Initialized with Xavier
        uniform distribution. As explained in the authors' paper, almost
        diagonal matrices can be seen as complex matrices.
    im_rel_emb: torch.nn.Embedding, shape: (n_rel, emb_dim)
        Imaginary part of the relations complex embeddings. Initialized with
        Xavier uniform distribution. As explained in the authors' paper, almost
        diagonal matrices can be seen as complex matrices.
    """

    def __init__(self, emb_dim, n_entities, n_relations, scalar_share=0.5):
        super().__init__(emb_dim, n_entities, n_relations)

        self.scalar_dim = int(self.emb_dim * scalar_share)
        self.complex_dim = int((self.emb_dim - self.scalar_dim))

        self.sc_ent_emb = init_embedding(self.n_ent, self.scalar_dim)
        self.re_ent_emb = init_embedding(self.n_ent, self.complex_dim)
        self.im_ent_emb = init_embedding(self.n_ent, self.complex_dim)

        self.sc_rel_emb = init_embedding(self.n_rel, self.scalar_dim)
        self.re_rel_emb = init_embedding(self.n_rel, self.complex_dim)
        self.im_rel_emb = init_embedding(self.n_rel, self.complex_dim)

    def scoring_function(self, h_idx, t_idx, r_idx):
        """Compute the scoring function for the triplets given as argument:
        :math:`h_{sc}^T \\cdot diag(r_{sc}) \\cdot t_{sc} + \\Re(h_{compl}
        \\cdot diag(r_{compl} \\cdot t_{compl}))`. See referenced paper for
        more details on the score. See torchkge.models.interfaces.Models for
        more details on the API.
        """

        sc_h, re_h, im_h = self.sc_ent_emb(h_idx), self.re_ent_emb(
            h_idx), self.im_ent_emb(h_idx)
        sc_t, re_t, im_t = self.sc_ent_emb(t_idx), self.re_ent_emb(
            t_idx), self.im_ent_emb(t_idx)
        sc_r, re_r, im_r = self.sc_rel_emb(r_idx), self.re_rel_emb(
            r_idx), self.im_rel_emb(r_idx)

        return ((sc_h * sc_r * sc_t).sum(dim=1) +
                (re_h * (re_r * re_t + im_r * im_t) + im_h * (re_r * im_t - im_r * re_t)).sum(dim=1))

    def normalize_parameters(self):
        """According to original paper, the embeddings should not be
        normalized.
        """
        pass

    def get_embeddings(self):
        """Return the embeddings of entities and relations.

        Returns
        -------
        sc_ent_emb: torch.Tensor, shape: (n_ent, emb_dim), dtype: torch.float
            Scalar part of embeddings of entities.
        re_ent_emb: torch.Tensor, shape: (n_ent, emb_dim), dtype: torch.float
            Real part of embeddings of entities.
        im_ent_emb: torch.Tensor, shape: (n_ent, emb_dim), dtype: torch.float
            Imaginary part of embeddings of entities.
        sc_rel_emb: torch.Tensor, shape: (n_rel, emb_dim), dtype: torch.float
            Scalar part of embeddings of relations.
        re_rel_emb: torch.Tensor, shape: (n_rel, emb_dim), dtype: torch.float
            Real part of embeddings of relations.
        im_rel_emb: torch.Tensor, shape: (n_rel, emb_dim), dtype: torch.float
            Imaginary part of embeddings of relations.

        """
        self.normalize_parameters()
        return self.sc_ent_emb.weight.data, self.re_ent_emb.weight.data, \
            self.im_ent_emb.weight.data, self.sc_rel_emb.weight.data, \
            self.re_rel_emb.weight.data, self.im_rel_emb.weight.data

    def inference_scoring_function(self, h, t, r):
        """Link prediction evaluation helper function. See
        torchkge.models.interfaces.Models for more details one the API.

        """
        sc_h, re_h, im_h = h[0], h[1], h[2]
        sc_t, re_t, im_t = t[0], t[1], t[2]
        sc_r, re_r, im_r = r[0], r[1], r[2]
        b_size = re_h.shape[0]

        if len(re_t.shape) == 3:
            assert (len(re_h.shape) == 2) & (len(re_r.shape) == 2)
            # this is the tail completion case in link prediction
            return ((sc_h * sc_r).view(b_size, 1, self.scalar_dim) * sc_t
                    + (re_h * re_r - im_h * im_r).view(b_size, 1, self.complex_dim) * re_t
                    + (re_h * im_r + im_h * re_r).view(b_size, 1, self.complex_dim) * im_t).sum(dim=2)

        elif len(re_h.shape) == 3:
            assert (len(re_t.shape) == 2) & (len(re_r.shape) == 2)
            # this is the head completion case in link prediction
            return (sc_h * (sc_r * sc_t).view(b_size, 1, self.scalar_dim)
                    + re_h * (re_r * re_t + im_r * im_t).view(b_size, 1, self.complex_dim)
                    + im_h * (re_r * im_t - im_r * re_t).view(b_size, 1, self.complex_dim)).sum(dim=2)

        elif len(re_r.shape) == 3:
            assert (len(re_h.shape) == 2) & (len(re_t.shape) == 2)
            # this is the relation prediction case
            return (sc_r * (sc_h * sc_t).view(b_size, 1, self.scalar_dim)
                    + re_r * (re_h * re_t + im_h * im_t).view(b_size, 1, self.complex_dim)
                    + im_r * (re_h * im_t - im_h * re_t).view(b_size, 1, self.complex_dim)).sum(dim=2)

    def inference_prepare_candidates(self, h_idx, t_idx, r_idx, entities=True):
        """Link prediction evaluation helper function. Get entities embeddings
        and relations embeddings. The output will be fed to the
        `inference_scoring_function` method. See torchkge.models.interfaces.Models for
        more details on the API.

        """
        b_size = h_idx.shape[0]

        sc_h = self.sc_ent_emb(h_idx)
        re_h = self.re_ent_emb(h_idx)
        im_h = self.im_ent_emb(h_idx)

        sc_t = self.sc_ent_emb(t_idx)
        re_t = self.re_ent_emb(t_idx)
        im_t = self.im_ent_emb(t_idx)

        sc_r = self.sc_rel_emb(r_idx)
        re_r = self.re_rel_emb(r_idx)
        im_r = self.im_rel_emb(r_idx)

        if entities:
            sc_candidates = self.sc_ent_emb.weight.data
            sc_candidates = sc_candidates.view(1, self.n_ent, self.scalar_dim)
            sc_candidates = sc_candidates.expand(b_size, self.n_ent, self.scalar_dim)

            re_candidates = self.re_ent_emb.weight.data
            re_candidates = re_candidates.view(1, self.n_ent, self.complex_dim)
            re_candidates = re_candidates.expand(b_size, self.n_ent, self.complex_dim)

            im_candidates = self.im_ent_emb.weight.data
            im_candidates = im_candidates.view(1, self.n_ent, self.complex_dim)
            im_candidates = im_candidates.expand(b_size, self.n_ent, self.complex_dim)

        else:
            sc_candidates = self.sc_rel_emb.weight.data
            sc_candidates = sc_candidates.view(1, self.n_rel, self.scalar_dim)
            sc_candidates = sc_candidates.expand(b_size, self.n_rel, self.scalar_dim)

            re_candidates = self.re_rel_emb.weight.data
            re_candidates = re_candidates.view(1, self.n_rel, self.complex_dim)
            re_candidates = re_candidates.expand(b_size, self.n_rel, self.complex_dim)

            im_candidates = self.im_rel_emb.weight.data
            im_candidates = im_candidates.view(1, self.n_rel, self.complex_dim)
            im_candidates = im_candidates.expand(b_size, self.n_rel, self.complex_dim)

        return (sc_h, re_h, im_h), \
               (sc_t, re_t, im_t), \
               (sc_r, re_r, im_r), \
               (sc_candidates, re_candidates, im_candidates)
