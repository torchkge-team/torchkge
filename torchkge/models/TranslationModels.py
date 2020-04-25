# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""

from torch import empty, matmul, eye, tensor
from torch.cuda import empty_cache
from torch.nn import Parameter, Embedding
from torch.nn.functional import normalize
from torch.nn.init import xavier_uniform_

from ..models.interfaces import TranslationModel
from ..utils import init_embedding
from ..utils import l1_torus_dissimilarity, l2_torus_dissimilarity, \
    el2_torus_dissimilarity

from tqdm.autonotebook import tqdm


class TransEModel(TranslationModel):
    """Implementation of TransE model detailed in 2013 paper by Bordes et al..
    This class inherits from the
    :class:`torchkge.models.interfaces.TranslationalModel` interface. It then
    has its attributes as well.


    References
    ----------
    * Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and
      Oksana Yakhnenko.
      `Translating Embeddings for Modeling Multi-relational Data.
      <https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data>`_
      In Advances in Neural Information Processing Systems 26, pages 2787–2795,
      2013.

    Parameters
    ----------
    emb_dim: int
        Dimension of the embedding.
    n_entities: int
        Number of entities in the current data set.
    n_relations: int
        Number of relations in the current data set.
    dissimilarity_type: int
        Either 1 or 2 for L1 or L2 dissimilarity.

    Attributes
    ----------
    dissimilarity_type: int
        Either 1 or 2 for L1 or L2 dissimilarity.
    emb_dim: int
        Dimension of the embedding.
    ent_emb: `torch.nn.Embedding`, shape: (n_ent, emb_dim)
        Embeddings of the entities, initialized with Xavier uniform
        distribution and then normalized.
    rel_emb: `torch.nn.Embedding`, shape: (n_rel, emb_dim)
        Embeddings of the relations, initialized with Xavier uniform
        distribution and then normalized.

    """

    def __init__(self, emb_dim, n_entities, n_relations, dissimilarity_type):

        super().__init__(n_entities, n_relations, dissimilarity_type)

        self.dissimilarity_type = dissimilarity_type
        self.emb_dim = emb_dim

        self.ent_emb = Embedding(self.n_ent, self.emb_dim)
        self.rel_emb = Embedding(self.n_rel, self.emb_dim)

        xavier_uniform_(self.ent_emb.weight.data)
        xavier_uniform_(self.rel_emb.weight.data)

        self.normalize_parameters()
        self.rel_emb.weight.data = normalize(self.rel_emb.weight.data,
                                             p=2, dim=1)

    def scoring_function(self, h_idx, t_idx, r_idx):
        """Compute the scoring function for the triplets given as argument.

        Parameters
        ----------
        h_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Indices of current head entities.
        t_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Indices of current tail entities.
        r_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Indices of current relations.

        Returns
        -------
        score: `torch.Tensor`, dtype: `torch.float`, shape: (batch_size)
            Score function: opposite of dissimilarities between h+r and t.

        """
        h_emb = normalize(self.ent_emb(h_idx), p=2, dim=1)
        t_emb = normalize(self.ent_emb(t_idx), p=2, dim=1)
        r_emb = self.rel_emb(r_idx)

        return - (h_emb + r_emb - t_emb).norm(p=self.dissimilarity_type, dim=1)

    def normalize_parameters(self):  # TODO
        """Normalize the parameters of the model using the L2 norm.
        """
        self.ent_emb.weight.data = normalize(self.ent_emb.weight.data,
                                             p=2, dim=1)

    def lp_get_emb_cand(self, h_idx, t_idx, r_idx):
        """Project current entities and candidates into relation-specific
        sub-spaces.

        Parameters
        ----------
        h_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Indices of current head entities.
        t_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Indices of current tail entities.
        r_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Indices of current relations.

        Returns
        -------
        h_emb: `torch.Tensor`, shape: (b_size, emb_dim), dtype: `torch.float`
            Embeddings of current head entities.
        t_emb: `torch.Tensor`, shape: (b_size, emb_dim), dtype: `torch.float`
            Embeddings of current tail entities.
        candidates: `torch.Tensor`, shape: (b_size, n_ent, emb_dim), \
            dtype: `torch.float`
            All entities' embeddings duplicated along first axis.
        r_emb: `torch.Tensor`, shape: (b_size, emb_dim), dtype: `torch.float`
            Current relations embeddings.

        """
        b_size = h_idx.shape[0]

        h_emb = self.ent_emb(h_idx)
        t_emb = self.ent_emb(t_idx)
        r_emb = self.rel_emb(r_idx)

        candidates = self.ent_emb.weight.data.view(1, self.n_ent, self.emb_dim)
        candidates = candidates.expand(b_size, self.n_ent, self.emb_dim)

        return h_emb, t_emb, candidates, r_emb


class TransHModel(TranslationModel):
    """Implementation of TransH model detailed in 2014 paper by Wang et al..
    This class inherits from the
    :class:`torchkge.models.interfaces.TranslationalModel` interface. It then
    has its attributes as well.

    References
    ----------
    * Zhen Wang, Jianwen Zhang, Jianlin Feng, and Zheng Chen.
      `Knowledge Graph Embedding by Translating on Hyperplanes.
      <https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8531>`_
      In Twenty-Eighth AAAI Conference on Artificial Intelligence, June 2014.

    Parameters
    ----------
    emb_dim: int
        Dimension of the embedding space.
    n_entities: int
        Number of entities in the current data set.
    n_relations: int
        Number of relations in the current data set.

    Attributes
    ----------
    emb_dim: int
        Dimension of the embedding.
    ent_emb: `torch.nn.Embedding`, shape: (n_ent, emb_dim)
        Embeddings of the entities, initialized with Xavier uniform
        distribution and then normalized.
    rel_emb: `torch.nn.Embedding`, shape: (n_rel, emb_dim)
        Embeddings of the relations, initialized with Xavier uniform
        distribution.
    norm_vect: `torch.nn.Embedding`, shape: (n_rel, emb_dim)
        Normal vectors associated to each relation and used to compute the
        relation-specific hyperplanes entities are projected on. See paper for
        more details. Initialized with Xavier uniform distribution and then
        normalized.

    """

    def __init__(self, emb_dim, n_entities, n_relations):
        super().__init__(n_entities, n_relations, dissimilarity_type=2)
        self.emb_dim = emb_dim
        self.ent_emb = Embedding(self.n_ent, self.emb_dim)
        self.rel_emb = Embedding(self.n_rel, self.emb_dim)
        self.norm_vect = Embedding(self.n_rel, self.emb_dim)

        xavier_uniform_(self.ent_emb.weight.data)
        xavier_uniform_(self.rel_emb.weight.data)
        xavier_uniform_(self.norm_vect.weight.data)

        self.normalize_parameters()

        self.evaluated_projections = False
        self.projected_entities = Parameter(empty(size=(self.n_rel,
                                                        self.n_ent,
                                                        self.emb_dim)),
                                            requires_grad=False)

    def scoring_function(self, h_idx, t_idx, r_idx):
        """Compute the scoring function for the triplets given as argument.

        Parameters
        ----------
        h_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Indices of current head entities.
        t_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Indices of current tail entities.
        r_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Indices of current relations.

        Returns
        -------
        score: `torch.Tensor`, dtype: `torch.float`, shape: (batch_size)
            Score function: opposite of dissimilarities between h+r and t after
            projection.

        """
        self.evaluated_projections = False

        h = normalize(self.ent_emb(h_idx), p=2, dim=1)
        t = normalize(self.ent_emb(t_idx), p=2, dim=1)
        r = self.rel_emb(r_idx)
        norm_vect = normalize(self.norm_vect(r_idx), p=2, dim=1)
        hrt = self.project(h, norm_vect) + r - self.project(t, norm_vect)
        return - hrt.norm(p=2, dim=1)

    @staticmethod
    def project(ent, norm_vect):
        return ent - (ent * norm_vect).sum(dim=1).view(-1, 1) * norm_vect

    def normalize_parameters(self):  # TODO
        """Normalize the parameters of the model using the L2 norm.
        """
        self.ent_emb.weight.data = normalize(self.ent_emb.weight.data,
                                             p=2, dim=1)
        self.norm_vect.weight.data = normalize(self.norm_vect.weight.data,
                                               p=2, dim=1)

    def lp_get_emb_cand(self, h_idx, t_idx, r_idx):
        """Project current entities and candidates into relation-specific
        sub-spaces.

        Parameters
        ----------
        h_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Indices of current head entities.
        t_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Indices of current tail entities.
        r_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Indices of current relations.

        Returns
        -------
        proj_h: `torch.Tensor`, shape: (b_size, emb_dim), dtype: `torch.float`
            Embeddings of current head entities projected on relation
            hyperplane.
        proj_t: `torch.Tensor`, shape: (b_size, emb_dim), dtype: `torch.float`
            Embeddings of current tail entities projected on relation
            hyperplane.
        proj_candidates: `torch.Tensor`, shape: (b_size, emb_dim, n_ent), \
            dtype: `torch.float`
            Tensor containing all entities projected on each relation-specific
            hyperplane (relations corresponding to current batch's relations).
        r: `torch.Tensor`, shape: (b_size, emb_dim), dtype: `torch.float`
            Embeddings of current relations.

        """
        if not self.evaluated_projections:
            self.lp_evaluate_projections()

        r = self.rel_emb(r_idx)
        proj_h = self.projected_entities[r_idx, h_idx]
        proj_t = self.projected_entities[r_idx, t_idx]
        proj_candidates = self.projected_entities[r_idx]

        return proj_h, proj_t, proj_candidates, r

    def lp_evaluate_projections(self):
        """Project all entities according to each relation.
                """
        if self.evaluated_projections:
            return

        for i in tqdm(range(self.n_ent), unit='entities',
                      desc='Projecting entities'):

            norm_vect = self.norm_vect.weight.data.view(self.n_rel,
                                                        self.emb_dim)

            mask = tensor([i], device=norm_vect.device).long()

            if norm_vect.is_cuda:
                empty_cache()

            ent = self.ent_emb(mask)
            norm_components = (ent.view(1, -1) * norm_vect).sum(dim=1)
            self.projected_entities[:, i, :] = (ent.view(1, -1) -
                                                norm_components.view(-1, 1) *
                                                norm_vect)

            del norm_components

        self.evaluated_projections = True


class TransRModel(TranslationModel):
    """Implementation of TransR model detailed in 2015 paper by Lin et al..
    This class inherits from the
    :class:`torchkge.models.interfaces.TranslationalModel` interface. It then
    has its attributes as well.

    References
    ----------
    * Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, and Xuan Zhu.
      `Learning Entity and Relation Embeddings for Knowledge Graph Completion.
      <https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9571/9523>`_
      In Twenty-Ninth AAAI Conference on Artificial Intelligence, February 2015

    Parameters
    ----------
    ent_emb_dim: int
        Dimension of the embedding of entities.
    rel_emb_dim: int
        Dimension of the embedding of relations.
    n_entities: int
        Number of entities in the current data set.
    n_relations: int
        Number of relations in the current data set.

    Attributes
    ----------
    ent_emb_dim: int
        Dimension nof the embedding of entities.
    rel_emb_dim: int
        Dimension of the embedding of relations.
    ent_emb: `torch.nn.Embedding`, shape: (n_ent, ent_emb_dim)
        Embeddings of the entities, initialized with Xavier uniform
        distribution and then normalized.
    rel_emb: `torch.nn.Embedding`, shape: (n_rel, rel_emb_dim)
        Embeddings of the relations, initialized with Xavier uniform
        distribution and then normalized.
    proj_mat: `torch.nn.Embedding`, shape: (n_rel, rel_emb_dim x ent_emb_dim)
        Relation-specific projection matrices. See paper for more details.
    projected_entities: `torch.nn.Parameter`, \
        shape: (n_rel, n_ent, rel_emb_dim)
        Contains the projection of each entity in each relation-specific
        sub-space.
    evaluated_projections: bool
        Indicates whether `projected_entities` has been computed. This should
        be set to true every time a backward pass is done in train mode.
    """

    def __init__(self, ent_emb_dim, rel_emb_dim, n_entities, n_relations):

        super().__init__(n_entities, n_relations, dissimilarity_type=2)
        self.ent_emb_dim = ent_emb_dim
        self.rel_emb_dim = rel_emb_dim

        self.ent_emb = Embedding(self.n_ent, self.ent_emb_dim)
        self.rel_emb = Embedding(self.n_rel, self.rel_emb_dim)
        self.proj_mat = Embedding(self.n_rel,
                                  self.rel_emb_dim * self.ent_emb_dim)

        xavier_uniform_(self.ent_emb.weight.data)
        xavier_uniform_(self.rel_emb.weight.data)
        xavier_uniform_(self.proj_mat.weight.data)

        self.normalize_parameters()

        self.evaluated_projections = False
        self.projected_entities = Parameter(empty(size=(self.n_rel, self.n_ent,
                                                        self.rel_emb_dim)),
                                            requires_grad=False)

    def scoring_function(self, h_idx, t_idx, r_idx):
        """Compute the scoring function for the triplets given as argument.

        Parameters
        ----------
        h_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Indices of current head entities.
        t_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Indices of current tail entities.
        r_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Indices of current relations.

        Returns
        -------
        score: `torch.Tensor`, dtype: `torch.float`, shape: (batch_size)
            Score function: opposite of dissimilarities between h+r and t
            after projection.

        """
        self.evaluated_projections = False

        b_size = h_idx.shape[0]
        h = normalize(self.ent_emb(h_idx), p=2, dim=1)
        t = normalize(self.ent_emb(t_idx), p=2, dim=1)
        r = self.rel_emb(r_idx)
        proj_mat = self.proj_mat(r_idx).view(b_size,
                                             self.rel_emb_dim,
                                             self.ent_emb_dim)
        hrt = self.project(h, proj_mat) + r - self.project(t, proj_mat)
        return - hrt.norm(p=2, dim=1)

    def project(self, ent, proj_mat):
        proj_e = matmul(proj_mat, ent.view(-1, self.ent_emb_dim, 1))
        return proj_e.view(-1, self.rel_emb_dim)

    def normalize_parameters(self):
        """Normalize the parameters of the model using the L2 norm.
        """
        self.ent_emb.weight.data = normalize(self.ent_emb.weight.data,
                                             p=2, dim=1)
        self.rel_emb.weight.data = normalize(self.rel_emb.weight.data,
                                             p=2, dim=1)
        # self.proj_mat.weight.data = normalize(self.proj_mat.weight.data,
        # p=2, dim=1) #TODO fix this

    def lp_get_emb_cand(self, h_idx, t_idx, r_idx):
        """Project current entities and candidates into relation-specific
        sub-spaces.

        Parameters
        ----------
        h_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Indices of current head entities.
        t_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Indices of current tail entities.
        r_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Indices of current relations.

        Returns
        -------
        proj_h: `torch.Tensor`, shape: (b_size, rel_emb_dim), \
            dtype: `torch.float`
            Embeddings of current head entities projected in relation space.
        proj_t: `torch.Tensor`, shape: (b_size, rel_emb_dim), \
            dtype: `torch.float`
            Embeddings of current tail entities projected in relation space.
        proj_candidates: `torch.Tensor`, shape: (b_size, rel_emb_dim, n_ent), \
            dtype: `torch.float`
            All entities projected in each of current batch's relation spaces.
        r: `torch.Tensor`, shape: (b_size, rel_emb_dim), dtype: `torch.float`
            Tensor containing current relations embeddings.

        """

        if not self.evaluated_projections:
            self.lp_evaluate_projections()

        r = self.rel_emb(r_idx)  # shape = (b_size, rel_emb_dim)
        proj_h = self.projected_entities[r_idx, h_idx]
        proj_t = self.projected_entities[r_idx, t_idx]
        proj_candidates = self.projected_entities[r_idx]

        return proj_h, proj_t, proj_candidates, r

    def lp_evaluate_projections(self):
        """Project all entities according to each relation.
                """
        if self.evaluated_projections:
            return

        for i in tqdm(range(self.n_ent), unit='entities',
                      desc='Projecting entities'):
            projection_matrices = self.proj_mat.weight.data
            projection_matrices = projection_matrices.view(self.n_rel,
                                                           self.rel_emb_dim,
                                                           self.ent_emb_dim)

            mask = tensor([i], device=projection_matrices.device).long()

            if projection_matrices.is_cuda:
                empty_cache()

            ent = self.ent_emb(mask)
            proj_ent = matmul(projection_matrices, ent.view(self.ent_emb_dim))
            proj_ent = proj_ent.view(self.n_rel, self.rel_emb_dim, 1)
            self.projected_entities[:, i, :] = proj_ent.view(self.n_rel,
                                                             self.rel_emb_dim)

            del proj_ent

        self.evaluated_projections = True


class TransDModel(TranslationModel):
    """Implementation of TransD model detailed in 2015 paper by Ji et al..
    This class inherits from the
    :class:`torchkge.models.interfaces.TranslationalModel` interface. It then
    has its attributes as well.

    References
    ----------
    * Guoliang Ji, Shizhu He, Liheng Xu, Kang Liu, and Jun Zhao.
      `Knowledge Graph Embedding via Dynamic Mapping Matrix.
      <https://aclweb.org/anthology/papers/P/P15/P15-1067/>`_
      In Proceedings of the 53rd Annual Meeting of the Association for
      Computational Linguistics and the 7th International Joint Conference on
      Natural Language Processing (Volume 1: Long Papers) pages 687–696,
      Beijing, China, July 2015. Association for Computational Linguistics.

    Parameters
    ----------
    ent_emb_dim: int
        Dimension of the embedding of entities.
    rel_emb_dim: int
        Dimension of the embedding of relations.
    n_entities: int
        Number of entities in the current data set.
    n_relations: int
        Number of relations in the current data set.

    Attributes
    ----------
    ent_emb_dim: int
        Dimension nof the embedding of entities.
    rel_emb_dim: int
        Dimension of the embedding of relations.
    ent_emb: `torch.nn.Embedding`, shape: (n_ent, ent_emb_dim)
        Embeddings of the entities, initialized with Xavier uniform
        distribution and then normalized.
    rel_emb: `torch.nn.Embedding`, shape: (n_rel, rel_emb_dim)
        Embeddings of the relations, initialized with Xavier uniform
        distribution and then normalized.
    ent_proj_vect: `torch.nn.Embedding`, shape: (n_ent, ent_emb_dim)
        Entity-specific vector used to build projection matrices. See paper for
        more details. Initialized with Xavier uniform distribution and then
        normalized.
    rel_proj_vect: `torch..nn.Embedding`, shape: (n_rel, rel_emb_dim)
        Relation-specific vector used to build projection matrices. See paper
        for more details. Initialized with Xavier uniform distribution and then
        normalized.
    projected_entities: `torch.nn.Parameter`, \
        shape: (n_rel, n_ent, rel_emb_dim)
        Contains the projection of each entity in each relation-specific
        sub-space.
    evaluated_projections: bool
        Indicates whether `projected_entities` has been computed. This should
        be set to true every time a backward pass is done in train mode.

    """

    def __init__(self, ent_emb_dim, rel_emb_dim, n_entities, n_relations):

        super().__init__(n_entities, n_relations, 2)

        self.ent_emb_dim = ent_emb_dim
        self.rel_emb_dim = rel_emb_dim

        self.ent_emb = Embedding(self.n_ent, self.ent_emb_dim)
        self.rel_emb = Embedding(self.n_rel, self.rel_emb_dim)
        self.ent_proj_vect = Embedding(self.n_ent, self.ent_emb_dim)
        self.rel_proj_vect = Embedding(self.n_rel, self.rel_emb_dim)

        xavier_uniform_(self.ent_emb.weight.data)
        xavier_uniform_(self.rel_emb.weight.data)
        xavier_uniform_(self.ent_proj_vect.weight.data)
        xavier_uniform_(self.rel_proj_vect.weight.data)

        self.normalize_parameters()

        self.evaluated_projections = False
        self.projected_entities = Parameter(empty(size=(self.n_rel,
                                                        self.n_ent,
                                                        self.rel_emb_dim)),
                                            requires_grad=False)

    def scoring_function(self, h_idx, t_idx, r_idx):
        """Compute the scoring function for the triplets given as argument.

        Parameters
        ----------
        h_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Indices of current head entities.
        t_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Indices of current tail entities.
        r_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Indices of current relations.

        Returns
        -------
        score: `torch.Tensor`, dtype: `torch.float`, shape: (batch_size)
            Score function: opposite of dissimilarities between h+r and t after
            projection.

        """
        self.evaluated_projections = False
        # TODO: project all pairs of entities and relations only once
        h = normalize(self.ent_emb(h_idx), p=2, dim=1)
        t = normalize(self.ent_emb(t_idx), p=2, dim=1)
        r = normalize(self.rel_emb(r_idx), p=2, dim=1)

        h_proj_v = normalize(self.ent_proj_vect(h_idx), p=2, dim=1)
        t_proj_v = normalize(self.ent_proj_vect(t_idx), p=2, dim=1)
        r_proj_v = normalize(self.rel_proj_vect(r_idx), p=2, dim=1)
        proj_h = self.project(h, h_proj_v, r_proj_v)
        proj_t = self.project(t, t_proj_v, r_proj_v)

        return - (proj_h + r - proj_t).norm(p=2, dim=1)

    def project(self, ent, e_proj_vect, r_proj_vect):
        b_size = ent.shape[0]

        proj_mat = matmul(r_proj_vect.view((b_size, self.rel_emb_dim, 1)),
                          e_proj_vect.view((b_size, 1, self.ent_emb_dim)))
        proj_mat += eye(n=self.rel_emb_dim, m=self.ent_emb_dim,
                        device=proj_mat.device)
        proj_e = matmul(proj_mat, ent.view(b_size, self.ent_emb_dim, 1))

        return proj_e.view(b_size, self.rel_emb_dim)

    def normalize_parameters(self):
        """Normalize the parameters of the model using the L2 norm.
        """
        self.ent_emb.weight.data = normalize(
            self.ent_emb.weight.data, p=2, dim=1)
        self.rel_emb.weight.data = normalize(
            self.rel_emb.weight.data, p=2, dim=1)
        self.ent_proj_vect.weight.data = normalize(
            self.ent_proj_vect.weight.data, p=2, dim=1)
        self.rel_proj_vect.weight.data = normalize(
            self.rel_proj_vect.weight.data, p=2, dim=1)

    def lp_get_emb_cand(self, h_idx, t_idx, r_idx):
        """Project current entities and candidates into relation-specific
        sub-spaces.

        Parameters
        ----------
        h_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Indices of current head entities.
        t_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Indices of current tail entities.
        r_idx: torch Tensor, shape: (b_size), dtype: `torch.long`
            Indices of current relations.
        Returns
        -------
        proj_h_emb: `torch.Tensor`, shape: (b_size, rel_emb_dim), \
            dtype: `torch.float`
            Embeddings of current head entities projected in relation space.
        proj_t_emb: `torch.Tensor`, shape: (b_size, rel_emb_dim), \
            dtype: `torch.float`
            Embeddings of current tail entities projected in relation space.
        proj_candidates: `torch.Tensor`, shape: (b_size, rel_emb_dim, n_ent), \
            dtype: `torch.float`
            All entities projected in each relation spaces (relations\
            corresponding to current batch's relations).
        r_emb: `torch.Tensor`, shape: (b_size, rel_emb_dim), \
            dtype: `torch.float`
            Embeddings of the current relations.

        """
        if not self.evaluated_projections:
            self.lp_evaluate_projections()

        r = self.rel_emb(r_idx)
        proj_h = self.projected_entities[r_idx, h_idx]
        proj_t = self.projected_entities[r_idx, t_idx]
        proj_candidates = self.projected_entities[r_idx]

        return proj_h, proj_t, proj_candidates, r

    def lp_evaluate_projections(self):
        """Project all entities according to each relation.
        """
        if self.evaluated_projections:
            return

        for i in tqdm(range(self.n_ent), unit='entities',
                      desc='Projecting entities'):
            ent_proj_vect = self.ent_proj_vect.weight[i]
            ent_proj_vect = ent_proj_vect.view(1, self.ent_emb_dim)
            rel_proj_vects = self.rel_proj_vect.weight.data
            rel_proj_vects = rel_proj_vects.view(self.n_rel,
                                                 self.rel_emb_dim,
                                                 1)

            projection_matrices = matmul(rel_proj_vects, ent_proj_vect)
            id_mat = eye(n=self.rel_emb_dim, m=self.ent_emb_dim,
                         device=projection_matrices.device)
            projection_matrices += id_mat.view(1,
                                               self.rel_emb_dim,
                                               self.ent_emb_dim)

            mask = tensor([i], device=projection_matrices.device).long()

            if projection_matrices.is_cuda:
                empty_cache()

            ent = self.ent_emb(mask)
            proj_ent = matmul(projection_matrices, ent.view(self.ent_emb_dim))
            proj_ent = proj_ent.view(self.n_rel, self.rel_emb_dim, 1)
            self.projected_entities[:, i, :] = proj_ent.view(self.n_rel,
                                                             self.rel_emb_dim)

            del proj_ent

        self.evaluated_projections = True


class TorusEModel(TransEModel):
    """Implementation of TorusE model detailed in 2018 paper by Ebisu and Ichise. This class inherits from the
    :class:`torchkge.models.TranslationalModels.TransEModel` class interpreted as an interface.
    It then has its attributes as well.

        References
        ----------
        * Takuma Ebisu and Ryutaro Ichise
        `TorusE: Knowledge Graph Embedding on a Lie Group.
        <https://arxiv.org/abs/1711.05435>`_
        In Proceedings of the 32nd AAAI Conference on Artificial Intelligence
        (New Orleans, LA, USA, Feb. 2018), AAAI Press, pp. 1819–1826.

        Parameters
        ----------
        ent_emb_dim: int
            Dimension of the embedding of entities.
        n_entities: int
            Number of entities in the current data set.
        n_relations: int
            Number of relations in the current data set.
        dissimilarity_type: function
            Used to compute dissimilarities (e.g. L1 or L2 dissimilarities).

        Attributes
        ----------
        dissimilarity_type: function
            Used to compute dissimilarities (e.g. L1 or L2 dissimilarities).
        relation_embeddings: torch Embedding, shape: (number_relations, emb_dim)
            Contains the embeddings of the relations. It is initialized with Xavier uniform and\
            then normalized.

        """

    def __init__(self, ent_emb_dim, n_entities, n_relations, dissimilarity_type):

        assert dissimilarity_type in ['L1', 'L2', 'eL2']
        self.dissimilarity_type = dissimilarity_type

        super().__init__(ent_emb_dim, n_entities, n_relations, dissimilarity_type=None)

        self.relation_embeddings = init_embedding(self.n_rel, self.ent_emb_dim)

        if self.dissimilarity_type == 'L1':
            self.dissimilarity = l1_torus_dissimilarity
        if self.dissimilarity_type == 'L2':
            self.dissimilarity = l2_torus_dissimilarity
        if self.dissimilarity_type == 'eL2':
            self.dissimilarity = el2_torus_dissimilarity

        self.normalize_parameters()

    def scoring_function(self, h_idx, t_idx, r_idx):
        """Compute the scoring function for the triplets given as argument.

        Parameters
        ----------
        h_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Indices of current head entities.
        t_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Indices of current tail entities.
        r_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Indices of current relations.

        Returns
        -------
        score: `torch.Tensor`, dtype: `torch.float`, shape: (batch_size)
            Score function: opposite of dissimilarities between h+r and t.

        """

        # recover relations embeddings
        rels_emb = self.relation_embeddings(r_idx)

        # recover, project and normalize entity embeddings
        h_emb = self.recover_project_normalize(h_idx, normalize_=False)
        t_emb = self.recover_project_normalize(t_idx, normalize_=False)

        if self.dissimilarity_type == 'L1':
            return 2 * self.dissimilarity(h_emb + rels_emb, t_emb)
        if self.dissimilarity_type == 'L2':
            return 4 * self.dissimilarity(h_emb + rels_emb, t_emb)**2
        else:
            assert self.dissimilarity_type == 'eL2'
            return self.dissimilarity(h_emb + rels_emb, t_emb)**2 / 4

    def recover_project_normalize(self, ent_idx, rel_idx=None, normalize_=False):
        """

        Parameters
        ----------
        ent_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Indices of current entities.
        rel_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Indices of relations.
        normalize_: bool
            Whether entities embeddings should be normalized or not.

        Returns
        -------
        projections: `torch.Tensor`, dtype: `torch.float`, shape: (batch_size, emb_dim)
            Embedded entities normalized.

        """
        # recover entity embeddings
        ent_emb = self.entity_embeddings(ent_idx)
        ent_emb.data.frac_()

        return ent_emb

    def normalize_parameters(self):
        """Project embeddings on torus.
        """
        self.entity_embeddings.weight.data.frac_()
        self.relation_embeddings.weight.data.frac_()
