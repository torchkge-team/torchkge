# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""

from torch import empty, matmul, tensor
from torch.cuda import empty_cache
from torch.nn import Parameter
from torch.nn.functional import normalize

from ..models.interfaces import TranslationModel
from ..utils import init_embedding

from tqdm.autonotebook import tqdm


class TransEModel(TranslationModel):
    """Implementation of TransE model detailed in 2013 paper by Bordes et al..
    This class inherits from the
    :class:`torchkge.models.interfaces.TranslationModel` interface. It then
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
    dissimilarity_type: str
        Either 'L1' or 'L2'.

    Attributes
    ----------
    emb_dim: int
        Dimension of the embedding.
    ent_emb: `torch.nn.Embedding`, shape: (n_ent, emb_dim)
        Embeddings of the entities, initialized with Xavier uniform
        distribution and then normalized.
    rel_emb: `torch.nn.Embedding`, shape: (n_rel, emb_dim)
        Embeddings of the relations, initialized with Xavier uniform
        distribution and then normalized.

    """

    def __init__(self, emb_dim, n_entities, n_relations, dissimilarity_type='L2'):

        super().__init__(n_entities, n_relations, dissimilarity_type)

        self.emb_dim = emb_dim
        self.ent_emb = init_embedding(self.n_ent, self.emb_dim)
        self.rel_emb = init_embedding(self.n_rel, self.emb_dim)

        self.normalize_parameters()
        self.rel_emb.weight.data = normalize(self.rel_emb.weight.data, p=2, dim=1)

    def scoring_function(self, h_idx, t_idx, r_idx):
        """Compute the scoring function for the triplets given as argument:
        :math:`||h + r - t||_p^p` with p being the `dissimilarity type (either
        1 or 2)`. See referenced paper for more details
        on the score. See torchkge.models.interfaces.Models for more details
        on the API.

        """
        h = normalize(self.ent_emb(h_idx), p=2, dim=1)
        t = normalize(self.ent_emb(t_idx), p=2, dim=1)
        r = self.rel_emb(r_idx)

        return - self.dissimilarity(h + r, t)

    def normalize_parameters(self):
        """Normalize the entity embeddings, as explained in original paper.
        This method should be called at the end of each training epoch and at
        the end of training as well.

        """
        self.ent_emb.weight.data = normalize(self.ent_emb.weight.data,
                                             p=2, dim=1)

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


class TransHModel(TranslationModel):
    """Implementation of TransH model detailed in 2014 paper by Wang et al..
    This class inherits from the
    :class:`torchkge.models.interfaces.TranslationModel` interface. It then
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
        super().__init__(n_entities, n_relations, dissimilarity_type='L2')
        self.emb_dim = emb_dim
        self.ent_emb = init_embedding(self.n_ent, self.emb_dim)
        self.rel_emb = init_embedding(self.n_rel, self.emb_dim)
        self.norm_vect = init_embedding(self.n_rel, self.emb_dim)

        self.normalize_parameters()

        self.evaluated_projections = False
        self.projected_entities = Parameter(empty(size=(self.n_rel,
                                                        self.n_ent,
                                                        self.emb_dim)),
                                            requires_grad=False)

    def scoring_function(self, h_idx, t_idx, r_idx):
        """Compute the scoring function for the triplets given as argument:
        :math:`||p_r(h) + r - p_r(t)||_2^2`. See referenced paper for
        more details on the score. See torchkge.models.interfaces.Models for
        more details on the API.

        """
        self.evaluated_projections = False

        h = normalize(self.ent_emb(h_idx), p=2, dim=1)
        t = normalize(self.ent_emb(t_idx), p=2, dim=1)
        r = self.rel_emb(r_idx)
        norm_vect = normalize(self.norm_vect(r_idx), p=2, dim=1)

        return - self.dissimilarity(self.project(h, norm_vect) + r,
                                    self.project(t, norm_vect))

    @staticmethod
    def project(ent, norm_vect):
        return ent - (ent * norm_vect).sum(dim=1).view(-1, 1) * norm_vect

    def normalize_parameters(self):
        """Normalize the entity embeddings and relations normal vectors, as
        explained in original paper. This methods should be called at the end
        of each training epoch and at the end of training as well.

        """
        self.ent_emb.weight.data = normalize(self.ent_emb.weight.data,
                                             p=2, dim=1)
        self.norm_vect.weight.data = normalize(self.norm_vect.weight.data,
                                               p=2, dim=1)
        self.rel_emb.weight.data = self.project(self.rel_emb.weight.data,
                                                self.norm_vect.weight.data)

    def get_embeddings(self):
        """Return the embeddings of entities and relations along with relation
        normal vectors.

        Returns
        -------
        ent_emb: torch.Tensor, shape: (n_ent, emb_dim), dtype: torch.float
            Embeddings of entities.
        rel_emb: torch.Tensor, shape: (n_rel, emb_dim), dtype: torch.float
            Embeddings of relations.
        norm_vect: torch.Tensor, shape: (n_rel, emb_dim), dtype: torch.float
            Normal vectors defining relation-specific hyperplanes.
        """
        self.normalize_parameters()
        return self.ent_emb.weight.data, self.rel_emb.weight.data, \
            self.norm_vect.weight.data

    def inference_prepare_candidates(self, h_idx, t_idx, r_idx, entities=True):
        """Link prediction evaluation helper function. Get entities embeddings
        and relations embeddings. The output will be fed to the
        `inference_scoring_function` method. See torchkge.models.interfaces.Models for
        more details on the API.

        """
        b_size = h_idx.shape[0]

        if not self.evaluated_projections:
            self.evaluate_projections()

        r = self.rel_emb(r_idx)

        if entities:
            proj_h = self.projected_entities[r_idx, h_idx]  # shape: (b_size, emb_dim)
            proj_t = self.projected_entities[r_idx, t_idx]  # shape: (b_size, emb_dim)
            candidates = self.projected_entities[r_idx]  # shape: (b_size, self.n_rel, self.emb_dim)
        else:
            proj_h = self.projected_entities[:, h_idx].transpose(0, 1)  # shape: (b_size, n_rel, emb_dim)
            proj_t = self.projected_entities[:, t_idx].transpose(0, 1)  # shape: (b_size, n_rel, emb_dim)
            candidates = self.rel_emb.weight.data.view(1, self.n_rel, self.emb_dim)
            candidates = candidates.expand(b_size, self.n_rel, self.emb_dim)

        return proj_h, proj_t, r, candidates

    def evaluate_projections(self):
        """Link prediction evaluation helper function. Project all entities
        according to each relation. Calling this method at the beginning of
        link prediction makes the process faster by computing projections only
        once.

        """
        if self.evaluated_projections:
            return

        for i in tqdm(range(self.n_ent), unit='entities', desc='Projecting entities'):

            norm_vect = self.norm_vect.weight.data.view(self.n_rel, self.emb_dim)
            mask = tensor([i], device=norm_vect.device).long()

            if norm_vect.is_cuda:
                empty_cache()

            ent = self.ent_emb(mask)
            norm_components = (ent.view(1, -1) * norm_vect).sum(dim=1)
            self.projected_entities[:, i, :] = (ent.view(1, -1) - norm_components.view(-1, 1) * norm_vect)

            del norm_components

        self.evaluated_projections = True


class TransRModel(TranslationModel):
    """Implementation of TransR model detailed in 2015 paper by Lin et al..
    This class inherits from the
    :class:`torchkge.models.interfaces.TranslationModel` interface. It then
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

        super().__init__(n_entities, n_relations, 'L2')
        self.ent_emb_dim = ent_emb_dim
        self.rel_emb_dim = rel_emb_dim

        self.ent_emb = init_embedding(self.n_ent, self.ent_emb_dim)
        self.rel_emb = init_embedding(self.n_rel, self.rel_emb_dim)
        self.proj_mat = init_embedding(self.n_rel,
                                       self.rel_emb_dim * self.ent_emb_dim)

        self.normalize_parameters()

        self.evaluated_projections = False
        self.projected_entities = Parameter(empty(size=(self.n_rel, self.n_ent,
                                                        self.rel_emb_dim)),
                                            requires_grad=False)

    def scoring_function(self, h_idx, t_idx, r_idx):
        """Compute the scoring function for the triplets given as argument:
        :math:`||p_r(h) + r - p_r(t)||_2^2`. See referenced paper for
        more details on the score. See torchkge.models.interfaces.Models for
        more details on the API.

        """
        self.evaluated_projections = False

        b_size = h_idx.shape[0]
        h = normalize(self.ent_emb(h_idx), p=2, dim=1)
        t = normalize(self.ent_emb(t_idx), p=2, dim=1)
        r = self.rel_emb(r_idx)
        proj_mat = self.proj_mat(r_idx).view(b_size,
                                             self.rel_emb_dim,
                                             self.ent_emb_dim)
        return - self.dissimilarity(self.project(h, proj_mat) + r,
                                    self.project(t, proj_mat))

    def project(self, ent, proj_mat):
        proj_e = matmul(proj_mat, ent.view(-1, self.ent_emb_dim, 1))
        return proj_e.view(-1, self.rel_emb_dim)

    def normalize_parameters(self):
        """Normalize the entity and relation embeddings, as explained in
        original paper. This methods should be called at the end of each
        training epoch and at the end of training as well.

        """
        self.ent_emb.weight.data = normalize(self.ent_emb.weight.data,
                                             p=2, dim=1)
        self.rel_emb.weight.data = normalize(self.rel_emb.weight.data,
                                             p=2, dim=1)

    def get_embeddings(self):
        """Return the embeddings of entities and relations along with their
        projection matrices.

        Returns
        -------
        ent_emb: torch.Tensor, shape: (n_ent, ent_emb_dim), dtype: torch.float
            Embeddings of entities.
        rel_emb: torch.Tensor, shape: (n_rel, rel_emb_dim), dtype: torch.float
            Embeddings of relations.
        proj_mat: torch.Tensor, shape: (n_rel, rel_emb_dim, ent_emb_dim),
        dtype: torch.float
            Relation-specific projection matrices.
        """
        self.normalize_parameters()
        return self.ent_emb.weight.data, self.rel_emb.weight.data, \
            self.proj_mat.weight.data.view(-1,
                                           self.rel_emb_dim,
                                           self.ent_emb_dim)

    def inference_prepare_candidates(self, h_idx, t_idx, r_idx, entities=True):
        """Link prediction evaluation helper function. Get entities embeddings
        and relations embeddings. The output will be fed to the
        `inference_scoring_function` method. See torchkge.models.interfaces.Models for
        more details on the API.

        """
        b_size = h_idx.shape[0]

        if not self.evaluated_projections:
            self.evaluate_projectionss()

        r = self.rel_emb(r_idx)  # shape = (b_size, self.rel_emb_dim)

        if entities:
            proj_h = self.projected_entities[r_idx, h_idx]  # shape: (b_size, rel_emb_dim)
            proj_t = self.projected_entities[r_idx, t_idx]  # shape: (b_size, rel_emb_dim)
            candidates = self.projected_entities[r_idx]  # shape: (b_size, n_rel, emb_dim)
        else:
            proj_h = self.projected_entities[:, h_idx].transpose(0, 1)  # shape: (b_size, n_rel, rel_emb_dim)
            proj_t = self.projected_entities[:, t_idx].transpose(0, 1)  # shape: (b_size, n_rel, rel_emb_dim)
            candidates = self.rel_emb.weight.data.view(1, self.n_rel, self.rel_emb_dim)
            candidates = candidates.expand(b_size, self.n_rel, self.rel_emb_dim)

        return proj_h, proj_t, r, candidates

    def evaluate_projectionss(self):
        """Link prediction evaluation helper function. Project all entities
        according to each relation. Calling this method at the beginning of
        link prediction makes the process faster by computing projections only
        once.

        """
        if self.evaluated_projections:
            return

        for i in tqdm(range(self.n_ent), unit='entities', desc='Projecting entities'):
            projection_matrices = self.proj_mat.weight.data
            projection_matrices = projection_matrices.view(self.n_rel, self.rel_emb_dim, self.ent_emb_dim)

            mask = tensor([i], device=projection_matrices.device).long()

            if projection_matrices.is_cuda:
                empty_cache()

            ent = self.ent_emb(mask)
            proj_ent = matmul(projection_matrices, ent.view(self.ent_emb_dim))
            proj_ent = proj_ent.view(self.n_rel, self.rel_emb_dim, 1)
            self.projected_entities[:, i, :] = proj_ent.view(self.n_rel, self.rel_emb_dim)

            del proj_ent

        self.evaluated_projections = True


class TransDModel(TranslationModel):
    """Implementation of TransD model detailed in 2015 paper by Ji et al..
    This class inherits from the
    :class:`torchkge.models.interfaces.TranslationModel` interface. It then
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
        Dimension of the embedding of entities.
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

        super().__init__(n_entities, n_relations, 'L2')

        self.ent_emb_dim = ent_emb_dim
        self.rel_emb_dim = rel_emb_dim

        self.ent_emb = init_embedding(self.n_ent, self.ent_emb_dim)
        self.rel_emb = init_embedding(self.n_rel, self.rel_emb_dim)
        self.ent_proj_vect = init_embedding(self.n_ent, self.ent_emb_dim)
        self.rel_proj_vect = init_embedding(self.n_rel, self.rel_emb_dim)

        self.normalize_parameters()

        self.evaluated_projections = False
        self.projected_entities = Parameter(empty(size=(self.n_rel,
                                                        self.n_ent,
                                                        self.rel_emb_dim)),
                                            requires_grad=False)

    def scoring_function(self, h_idx, t_idx, r_idx):
        """Compute the scoring function for the triplets given as argument:
        :math:`||p_r(h) + r - p_r(t)||_2^2`. See referenced paper for
        more details on the score. See torchkge.models.interfaces.Models for
        more details on the API.

        """
        self.evaluated_projections = False
        h = normalize(self.ent_emb(h_idx), p=2, dim=1)
        t = normalize(self.ent_emb(t_idx), p=2, dim=1)
        r = normalize(self.rel_emb(r_idx), p=2, dim=1)

        h_proj_v = normalize(self.ent_proj_vect(h_idx), p=2, dim=1)
        t_proj_v = normalize(self.ent_proj_vect(t_idx), p=2, dim=1)
        r_proj_v = normalize(self.rel_proj_vect(r_idx), p=2, dim=1)

        proj_h = self.project(h, h_proj_v, r_proj_v)
        proj_t = self.project(t, t_proj_v, r_proj_v)
        return - self.dissimilarity(proj_h + r, proj_t)

    def project(self, ent, e_proj_vect, r_proj_vect):
        """We note that :math:`p_r(e)_i = e^p^Te \\times r^p_i + e_i` which is
        more efficient to compute than the matrix formulation in the original
        paper.

        """
        b_size = ent.shape[0]

        scalar_product = (ent * e_proj_vect).sum(dim=1)
        proj_e = (r_proj_vect * scalar_product.view(b_size, 1))
        return proj_e + ent[:, :self.rel_emb_dim]

    def normalize_parameters(self):
        """Normalize the entity embeddings and relations normal vectors, as
        explained in original paper. This methods should be called at the end
        of each training epoch and at the end of training as well.

        """
        self.ent_emb.weight.data = normalize(self.ent_emb.weight.data, p=2, dim=1)
        self.rel_emb.weight.data = normalize(self.rel_emb.weight.data, p=2, dim=1)
        self.ent_proj_vect.weight.data = normalize(self.ent_proj_vect.weight.data, p=2, dim=1)
        self.rel_proj_vect.weight.data = normalize(self.rel_proj_vect.weight.data, p=2, dim=1)

    def get_embeddings(self):
        """Return the embeddings of entities and relations along with their
        projection vectors.

        Returns
        -------
        ent_emb: torch.Tensor, shape: (n_ent, ent_emb_dim), dtype: torch.float
            Embeddings of entities.
        rel_emb: torch.Tensor, shape: (n_rel, rel_emb_dim), dtype: torch.float
            Embeddings of relations.
        ent_proj_vect: torch.Tensor, shape: (n_ent, ent_emb_dim),
        dtype: torch.float
            Entity projection vectors.
        rel_proj_vect: torch.Tensor, shape: (n_ent, rel_emb_dim),
        dtype: torch.float
            Relation projection vectors.

        """
        self.normalize_parameters()
        return self.ent_emb.weight.data, self.rel_emb.weight.data, \
            self.ent_proj_vect.weight.data, self.rel_proj_vect.weight.data

    def inference_prepare_candidates(self, h_idx, t_idx, r_idx, entities=True):
        """Link prediction evaluation helper function. Get entities embeddings
        and relations embeddings. The output will be fed to the
        `inference_scoring_function` method. See torchkge.models.interfaces.Models for
        more details on the API.

        """
        b_size = h_idx.shape[0]

        if not self.evaluated_projections:
            self.evaluate_projectionss()

        r = self.rel_emb(r_idx)

        if entities:
            proj_h = self.projected_entities[r_idx, h_idx]  # shape: (b_size, emb_dim)
            proj_t = self.projected_entities[r_idx, t_idx]  # shape: (b_size, emb_dim)
            candidates = self.projected_entities[r_idx]  # shape: (b_size, self.n_rel, self.emb_dim)
        else:
            proj_h = self.projected_entities[:, h_idx].transpose(0, 1)  # shape: (b_size, n_rel, rel_emb_dim)
            proj_t = self.projected_entities[:, t_idx].transpose(0, 1)  # shape: (b_size, n_rel, rel_emb_dim)
            candidates = self.rel_emb.weight.data.view(1, self.n_rel, self.rel_emb_dim)
            candidates = candidates.expand(b_size, self.n_rel, self.rel_emb_dim)

        return proj_h, proj_t, r, candidates

    def evaluate_projectionss(self):
        """Link prediction evaluation helper function. Project all entities
        according to each relation. Calling this method at the beginning of
        link prediction makes the process faster by computing projections only
        once.

        """
        if self.evaluated_projections:
            return

        for i in tqdm(range(self.n_ent), unit='entities', desc='Projecting entities'):

            rel_proj_vects = self.rel_proj_vect.weight.data
            ent = self.ent_emb.weight[i]
            ent_proj_vect = self.ent_proj_vect.weight[i]

            sc_prod = (ent_proj_vect * ent).sum(dim=0)
            proj_e = sc_prod * rel_proj_vects + ent[:self.rel_emb_dim].view(1, -1)

            self.projected_entities[:, i, :] = proj_e

            del proj_e

        self.evaluated_projections = True


class TorusEModel(TranslationModel):
    """Implementation of TorusE model detailed in 2018 paper by Ebisu and
    Ichise. This class inherits from the
    :class:`torchkge.models.interfaces.TranslationModel` interface. It then
    has its attributes as well.

    References
    ----------
    * Takuma Ebisu and Ryutaro Ichise
      `TorusE: Knowledge Graph Embedding on a Lie Group.
      <https://arxiv.org/abs/1711.05435>`_
      In Proceedings of the 32nd AAAI Conference on Artificial Intelligence
      (New Orleans, LA, USA, Feb. 2018), AAAI Press, pp. 1819–1826.

    Parameters
    ----------
    emb_dim: int
        Embedding dimension.
    n_entities: int
        Number of entities in the current data set.
    n_relations: int
        Number of relations in the current data set.
    dissimilarity_type: str
        One of 'torus_L1', 'torus_L2', 'torus_eL2'.

    Attributes
    ----------
    emb_dim: int
        Embedding dimension.
    ent_emb: torch.nn.Embedding, shape: (n_ent, emb_dim)
        Embeddings of the entities, initialized with Xavier uniform
        distribution and then normalized.
    rel_emb: torch.nn.Embedding, shape: (n_rel, emb_dim)
        Embeddings of the relations, initialized with Xavier uniform
        distribution and then normalized.

    """

    def __init__(self, emb_dim, n_entities, n_relations, dissimilarity_type):

        assert dissimilarity_type in ['L1', 'torus_L1', 'torus_L2', 'torus_eL2']
        super().__init__(n_entities, n_relations, dissimilarity_type)

        self.emb_dim = emb_dim
        self.ent_emb = init_embedding(self.n_ent, self.emb_dim)
        self.rel_emb = init_embedding(self.n_rel, self.emb_dim)

        self.normalized = False
        self.normalize_parameters()

    def scoring_function(self, h_idx, t_idx, r_idx):
        """Compute the scoring function for the triplets given as argument:
        See referenced paper for more details on the score. See
        torchkge.models.interfaces.Models for more details on the API.

        """
        self.normalized = False

        h = self.ent_emb(h_idx)
        t = self.ent_emb(t_idx)
        r = self.rel_emb(r_idx)

        h.data.frac_()
        t.data.frac_()
        r.data.frac_()

        return - self.dissimilarity(h + r, t)

    def normalize_parameters(self):
        """Project embeddings on torus.
        """
        self.ent_emb.weight.data.frac_()
        self.rel_emb.weight.data.frac_()
        self.normalized = True

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

    def inference_prepare_candidates(self, h_idx, t_idx, r_idx, entities=True):
        """Link prediction evaluation helper function. Get entities embeddings
        and relations embeddings. The output will be fed to the
        `inference_scoring_function` method. See torchkge.models.interfaces.Models for
        more details on the API.

        """
        b_size = h_idx.shape[0]

        if not self.normalized:
            self.normalize_parameters()

        h = self.ent_emb(h_idx)
        t = self.ent_emb(t_idx)
        r = self.rel_emb(r_idx)

        if entities:
            candidates = self.ent_emb.weight.data.view(1, self.n_ent, self.emb_dim)
            candidates = candidates.expand(b_size, self.n_ent, self.emb_dim)
        else:
            candidates = self.rel_emb.weight.data.view(1, self.n_rel, self.emb_dim)
            candidates = candidates.expand(b_size, self.n_rel, self.rel_emb_dim)

        return h, t, r, candidates
