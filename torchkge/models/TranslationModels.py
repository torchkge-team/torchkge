# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
aboschin@enst.fr
"""

from torch import empty, matmul, eye, arange, tensor
from torch.nn import Parameter
from torch.nn.functional import normalize
from torch.nn.init import xavier_uniform_
from torch.cuda import empty_cache

from torchkge.models import TranslationalModel
from torchkge.utils import init_embedding
from torchkge.utils import l1_torus_dissimilarity, l2_torus_dissimilarity, el2_torus_dissimilarity

from tqdm import tqdm


class TransEModel(TranslationalModel):
    """Implementation of TransE model detailed in 2013 paper by Bordes et al..

    References
    ----------
    * Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko.
      Translating Embeddings for Modeling Multi-relational Data.
      In Advances in Neural Information Processing Systems 26, pages 2787–2795, 2013.
      https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data

    Parameters
    ----------
    ent_emb_dim: int
        Dimension of the embedding of entities.
    n_entities: int
        Number of entities in the current data set.
    n_relations: int
        Number of relations in the current data set.
    dissimilarity: String
        Either 'L1' or 'L2'.

    Attributes
    ----------
    ent_emb_dim: int
        Dimension of the embedding of entities.
    number_entities: int
        Number of entities in the current data set.
    number_relations: int
        Number of relations in the current data set.
    dissimilarity: function
        Used to compute dissimilarities (e.g. L1 or L2 dissimilarities).
    entity_embeddings: torch Embedding, shape = (number_entities, ent_emb_dim)
        Contains the embeddings of the entities. It is initialized with Xavier uniform and then\
         normalized.
    relation_embeddings: torch Embedding, shape = (number_relations, ent_emb_dim)
        Contains the embeddings of the relations. It is initialized with Xavier uniform and then\
         normalized.

    """

    def __init__(self, ent_emb_dim, n_entities, n_relations, dissimilarity):
        try:
            assert dissimilarity in ['L1', 'L2', None]
        except AssertionError:
            raise AssertionError("Dissimilarity variable can either be 'L1' or 'L2'.")

        super().__init__(ent_emb_dim, n_entities, n_relations, dissimilarity)

        # initialize embeddings
        self.relation_embeddings = init_embedding(self.number_relations, self.ent_emb_dim)

        # normalize parameters
        self.relation_embeddings.weight.data = normalize(self.relation_embeddings.weight.data,
                                                         p=2, dim=1)
        self.entity_embeddings.weight.data = normalize(self.entity_embeddings.weight.data,
                                                       p=2, dim=1)

    def scoring_function(self, heads_idx, tails_idx, rels_idx):
        """Compute the scoring function for the triplets given as argument.

        Parameters
        ----------
        heads_idx: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's heads
        tails_idx: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's tails.
        rels_idx: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's relations.

        Returns
        -------
        score: torch.Tensor, dtype = float, shape = (batch_size)
            Score function: opposite of dissimilarities between h+r and t.

        """
        # recover relations embeddings
        rels_emb = self.relation_embeddings(rels_idx)

        # recover, project and normalize entity embeddings
        h_emb = self.recover_project_normalize(heads_idx, rels_idx, normalize_=True)
        t_emb = self.recover_project_normalize(tails_idx, rels_idx, normalize_=True)

        return - self.dissimilarity(h_emb + rels_emb, t_emb)

    def recover_project_normalize(self, ent_idx, rel_idx, normalize_=True):
        """

        Parameters
        ----------
        ent_idx: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of entities
        rel_idx: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of relations
        normalize_: bool
            Whether entities embeddings should be normalized or not.

        Returns
        -------
        projections: torch.Tensor, dtype = float, shape = (batch_size, ent_emb_dim)
            Embedded entities normalized.

        """
        # recover entity embeddings
        ent_emb = self.entity_embeddings(ent_idx)

        # normalize entity embeddings
        if normalize_:
            ent_emb = normalize(ent_emb, p=2, dim=1)

        return ent_emb

    def normalize_parameters(self):
        """Normalize the parameters of the model using the L2 norm.
        """
        self.entity_embeddings.weight.data = normalize(self.entity_embeddings.weight.data,
                                                       p=2, dim=1)

    def evaluation_helper(self, h_idx, t_idx, r_idx):
        """Project current entities and candidates into relation-specific sub-spaces.

        Parameters
        ----------
        h_idx: torch.Tensor, shape = (b_size), dtype = long
            Tensor containing indices of current head entities.
        t_idx: torch.Tensor, shape = (b_size), dtype = long
            Tensor containing indices of current tail entities.
        r_idx: torch.Tensor, shape = (b_size), dtype = long
            Tensor containing indices of current relations.

        Returns
        -------
        proj_h_emb: torch.Tensor, shape = (b_size, rel_emb_dim), dtype = float
            Tensor containing embeddings of current head entities projected in relation space.
        proj_t_emb: torch.Tensor, shape = (b_size, rel_emb_dim), dtype = float
            Tensor containing embeddings of current tail entities projected in relation space.
        proj_candidates: torch.Tensor, shape = (b_size, rel_emb_dim, n_entities), dtype = float
            Tensor containing all entities projected in each relation spaces (relations
            corresponding to current batch's relations).
        r_emb: torch.Tensor, shape = (b_size, rel_emb_dim), dtype = float
            Tensor containing current relations embeddings.

        """
        # recover, project and normalize entity embeddings
        all_idx = arange(0, self.number_entities).long()

        if h_idx.is_cuda:
            all_idx = all_idx.cuda()
        proj_candidates = self.recover_project_normalize(all_idx, None, normalize_=False)

        proj_h_emb = proj_candidates[h_idx]
        proj_t_emb = proj_candidates[t_idx]
        r_emb = self.relation_embeddings(r_idx)

        b_size, emb_dim = proj_h_emb.shape
        proj_candidates = proj_candidates.transpose(0, 1)
        proj_candidates = proj_candidates.view(1, emb_dim, self.number_entities)
        proj_candidates = proj_candidates.expand(b_size, emb_dim, self.number_entities)

        return proj_h_emb, proj_t_emb, proj_candidates, r_emb


class TransHModel(TransEModel):
    """Implementation of TransH model detailed in 2014 paper by Wang et al..

    References
    ----------
    * Zhen Wang, Jianwen Zhang, Jianlin Feng, and Zheng Chen.
      Knowledge Graph Embedding by Translating on Hyperplanes.
      In Twenty-Eighth AAAI Conference on Artificial Intelligence, June 2014.
      https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8531

    Parameters
    ----------
    ent_emb_dim: int
        Dimension of the embedding of entities.
    n_entities: int
        Number of entities in the current data set.
    n_relations: int
        Number of relations in the current data set.

    Attributes
    ----------
    ent_emb_dim: int
        Dimension of the embedding of entities.
    number_entities: int
        Number of entities in the current data set.
    number_relations: int
        Number of relations in the current data set.
    entity_embeddings: torch.nn.Embedding, shape = (number_entities, ent_emb_dim)
        Contains the embeddings of the entities. It is initialized with Xavier uniform and then\
         normalized.
    relation_embeddings: torch.nn.Embedding, shape = (number_relations, ent_emb_dim)
        Contains the embeddings of the relations. It is initialized with Xavier uniform and then\
        normalized.
    dissimilarity: function
        `torchkge.utils.dissimilarities.l2_dissimilarity`
    normal_vectors: torch.Tensor, shape = (number_relations, ent_emb_dim)
        Normal vectors associated to each relation and used to compute the relation-specific\
        hyperplanes entities are projected on. See paper for more details.

    """

    def __init__(self, ent_emb_dim, n_entities, n_relations):
        super().__init__(ent_emb_dim, n_entities, n_relations, dissimilarity='L2')
        self.normal_vectors = Parameter(xavier_uniform_(empty(size=(n_relations, ent_emb_dim))),
                                        requires_grad=True)

    def scoring_function(self, heads_idx, tails_idx, rels_idx):
        """Compute the scoring function for the triplets given as argument.

        Parameters
        ----------
        heads_idx: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's heads
        tails_idx: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's tails.
        rels_idx: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's relations.

        Returns
        -------
        score: torch.Tensor, dtype = float, shape = (batch_size)
            Score function: opposite of dissimilarities between h+r and t after projection.

        """
        # recover relations embeddings and normal projection vectors
        relations_embeddings = self.relation_embeddings(rels_idx)
        self.normal_vectors[rels_idx] = normalize(self.normal_vectors[rels_idx], p=2, dim=1)

        # project entities in relation specific hyperplane
        projected_heads = self.recover_project_normalize(heads_idx, rels_idx, normalize_=True)
        projected_tails = self.recover_project_normalize(tails_idx, rels_idx, normalize_=True)

        return - self.dissimilarity(projected_heads + relations_embeddings, projected_tails)

    def recover_project_normalize(self, ent_idx, rel_idx, normalize_=True):
        """Recover entity (either head or tail) embeddings and project on hyperplane defined by\
        provided normal vectors.

        Parameters
        ----------
        ent_idx: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of entities
        rel_idx: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of relations
        normalize_: bool
            Whether entities embeddings should be normalized or not.

        Returns
        -------
        projections: torch.Tensor, dtype = float, shape = (batch_size, ent_emb_dim)
            Projection of the embedded entities on the hyperplanes defined by the provided normal\
            vectors.

        """
        # recover entity embeddings
        ent_emb = self.entity_embeddings(ent_idx)

        # normalize entity embeddings
        if normalize_:
            ent_emb = normalize(ent_emb, p=2, dim=1)

        # project entities into relation space
        normal_vectors = self.normal_vectors[rel_idx]
        normal_component = (ent_emb * normal_vectors).sum(dim=1).view((-1, 1))

        return ent_emb - normal_component * normal_vectors

    def normalize_parameters(self):
        """Normalize the parameters of the model using the L2 norm.
        """
        self.normal_vectors.data = normalize(self.normal_vectors, p=2, dim=1)
        self.entity_embeddings.weight.data = normalize(self.entity_embeddings.weight.data,
                                                       p=2, dim=1)

    def evaluation_helper(self, h_idx, t_idx, r_idx):
        """Project current entities and candidates into relation-specific sub-spaces.

        Parameters
        ----------
        h_idx: torch.Tensor, shape = (b_size), dtype = long
            Tensor containing indices of current head entities.
        t_idx: torch.Tensor, shape = (b_size), dtype = long
            Tensor containing indices of current tail entities.
        r_idx: torch.Tensor, shape = (b_size), dtype = long
            Tensor containing indices of current relations.

        Returns
        -------
        proj_h_emb: torch.Tensor, shape = (b_size, rel_emb_dim), dtype = float
            Tensor containing embeddings of current head entities projected in relation space.
        proj_t_emb: torch.Tensor, shape = (b_size, rel_emb_dim), dtype = float
            Tensor containing embeddings of current tail entities projected in relation space.
        proj_candidates: torch.Tensor, shape = (b_size, rel_emb_dim, n_entities), dtype = float
            Tensor containing all entities projected in each relation spaces (relations
            corresponding to current batch's relations).
        r_emb: torch.Tensor, shape = (b_size, rel_emb_dim), dtype = float
            Tensor containing current relations embeddings.

        """
        # recover relations embeddings and normal projection vectors
        r_emb = self.relation_embeddings(r_idx)
        normal_vectors = normalize(self.normal_vectors[r_idx], p=2, dim=1)
        b_size, _ = normal_vectors.shape

        # recover candidates
        candidates = self.recover_candidates(h_idx, b_size)

        # project each candidates with each normal vector
        normal_components = candidates * normal_vectors.view((b_size, self.ent_emb_dim, 1))
        normal_components = normal_components.sum(dim=1).view(b_size, 1, self.number_entities)
        normal_components = normal_components * normal_vectors.view(b_size, self.ent_emb_dim, 1)
        proj_candidates = candidates - normal_components

        assert proj_candidates.shape == (b_size, self.ent_emb_dim, self.number_entities)

        # recover, project and normalize entity embeddings
        proj_h_emb, proj_t_emb = self.projection_helper(h_idx, t_idx, b_size, proj_candidates,
                                                        self.ent_emb_dim)

        return proj_h_emb, proj_t_emb, proj_candidates, r_emb


class TransRModel(TranslationalModel):
    """Implementation of TransR model detailed in 2015 paper by Lin et al..

    References
    ----------
    * Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, and Xuan Zhu.
      Learning Entity and Relation Embeddings for Knowledge Graph Completion.
      In Twenty-Ninth AAAI Conference on Artificial Intelligence, February 2015
      https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9571/9523

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
    number_entities: int
        Number of entities in the current data set.
    number_relations: int
        Number of relations in the current data set.
    entity_embeddings: torch.nn.Embedding, shape = (number_entities, ent_emb_dim)
        Contains the embeddings of the entities. It is initialized with Xavier uniform and then\
         normalized.
    relation_embeddings: torch.nn.Embedding, shape = (number_relations, rel_emb_dim)
        Contains the embeddings of the relations. It is initialized with Xavier uniform and then\
        normalized.
    dissimilarity: function
        `torchkge.utils.dissimilarities.l2_dissimilarity`
    projection_matrices: torch.Tensor, shape = (number_relations, rel_emb_dim, ent_emb_dim)
        Relation-specific projection matrices. See paper for more details.

    """

    def __init__(self, ent_emb_dim, rel_emb_dim, n_entities, n_relations):

        super().__init__(ent_emb_dim, n_entities, n_relations, dissimilarity='L2')

        self.rel_emb_dim = rel_emb_dim
        self.relation_embeddings = init_embedding(self.number_relations, self.rel_emb_dim)

        self.projection_matrices = Parameter(xavier_uniform_(empty(size=(n_relations, rel_emb_dim,
                                                                         ent_emb_dim))),
                                             requires_grad=True)
        self.normalize_parameters()

    def scoring_function(self, heads_idx, tails_idx, rels_idx):
        """Compute the scoring function for the triplets given as argument.

        Parameters
        ----------
        heads_idx: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's heads
        tails_idx: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's tails.
        rels_idx: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's relations.

        Returns
        -------
        score: torch.Tensor, dtype = float, shape = (batch_size)
            Score function: opposite of dissimilarities between h+r and t after projection.

        """
        # recover relations embeddings and normal projection matrices
        relations_embeddings = normalize(self.relation_embeddings(rels_idx), p=2, dim=1)

        # project entities in relation specific hyperplane
        projected_heads = self.recover_project_normalize(heads_idx, rels_idx, normalize_=True)
        projected_tails = self.recover_project_normalize(tails_idx, rels_idx, normalize_=True)

        return - self.dissimilarity(projected_heads + relations_embeddings, projected_tails)

    def recover_project_normalize(self, ent_idx, rel_idx, normalize_=True):
        """Recover entity (either head or tail) embeddings and project on hyperplane defined by\
        provided projection matrices.

        Parameters
        ----------
        ent_idx: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of entities
        rel_idx: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of relations
        normalize_: bool
            Whether entities embeddings should be normalized or not.

        Returns
        -------
        projections: torch.Tensor, dtype = float, shape = (batch_size, rel_emb_dim)
            Projection of the entities into relation-specific subspaces.
        """
        b_size = ent_idx.shape[0]
        # recover and normalize embeddings
        ent_emb = self.entity_embeddings(ent_idx)
        if normalize_:
            ent_emb = normalize(ent_emb, p=2, dim=1)

        # project entities into relation space
        new_shape = (b_size, self.ent_emb_dim, 1)
        projection_matrices = self.projection_matrices[rel_idx]
        projection = matmul(projection_matrices, ent_emb.view(new_shape))

        return normalize(projection.view(b_size, self.rel_emb_dim), p=2, dim=1)

    def normalize_parameters(self):
        """Normalize the parameters of the model using the L2 norm.
        """
        self.entity_embeddings.weight.data = normalize(self.entity_embeddings.weight.data,
                                                       p=2, dim=1)
        self.relation_embeddings.weight.data = normalize(self.relation_embeddings.weight.data,
                                                         p=2, dim=1)
        self.projection_matrices.data = normalize(self.projection_matrices.data, p=2, dim=2)

    def evaluation_helper(self, h_idx, t_idx, r_idx):
        """Project current entities and candidates into relation-specific sub-spaces.

        Parameters
        ----------
        h_idx: torch.Tensor, shape = (b_size), dtype = long
            Tensor containing indices of current head entities.
        t_idx: torch.Tensor, shape = (b_size), dtype = long
            Tensor containing indices of current tail entities.
        r_idx: torch.Tensor, shape = (b_size), dtype = long
            Tensor containing indices of current relations.

        Returns
        -------
        proj_h_emb: torch.Tensor, shape = (b_size, rel_emb_dim), dtype = float
            Tensor containing embeddings of current head entities projected in relation space.
        proj_t_emb: torch.Tensor, shape = (b_size, rel_emb_dim), dtype = float
            Tensor containing embeddings of current tail entities projected in relation space.
        proj_candidates: torch.Tensor, shape = (b_size, rel_emb_dim, n_entities), dtype = float
            Tensor containing all entities projected in each relation spaces (relations
            corresponding to current batch's relations).
        r_emb: torch.Tensor, shape = (b_size, rel_emb_dim), dtype = float
            Tensor containing current relations embeddings.

        """
        # recover relations embeddings and normal projection matrices
        r_emb = normalize(self.relation_embeddings(r_idx), p=2, dim=1)
        projection_matrices = normalize(self.projection_matrices[r_idx], p=2, dim=2)
        b_size, _, _ = projection_matrices.shape

        # recover candidates
        candidates = self.recover_candidates(h_idx, b_size)

        # project each candidates with each projection matrix
        proj_candidates = matmul(projection_matrices, candidates)

        proj_h_emb, proj_t_emb = self.projection_helper(h_idx, t_idx, b_size, candidates,
                                                        self.rel_emb_dim)

        return proj_h_emb, proj_t_emb, proj_candidates, r_emb


class TransDModel(TranslationalModel):
    """Implementation of TransD model detailed in 2015 paper by Ji et al..

    References
    ----------
    * Guoliang Ji, Shizhu He, Liheng Xu, Kang Liu, and Jun Zhao.
      Knowledge Graph Embedding via Dynamic Mapping Matrix.
      In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and
      the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)
      pages 687–696, Beijing, China, July 2015. Association for Computational Linguistics.
      https://aclweb.org/anthology/papers/P/P15/P15-1067/

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
    number_entities: int
        Number of entities in the current data set.
    number_relations: int
        Number of relations in the current data set.
    entity_embeddings: torch.nn.Embedding, shape = (number_entities, ent_emb_dim)
        Contains the embeddings of the entities. It is initialized with Xavier uniform and then\
         normalized.
    relation_embeddings: torch.nn.Embedding, shape = (number_relations, rel_emb_dim)
        Contains the embeddings of the relations. It is initialized with Xavier uniform and then\
        normalized.
    dissimilarity: function
        `torchkge.utils.dissimilarities.l2_dissimilarity`
    ent_proj_vects: torch.Tensor, shape = (number_entities, ent_emb_dim)
        Entity-specific vector used to build projection matrices. See paper for more details.
    rel_proj_vects: torch.Tensor, shape = (number_relations, rel_emb_dim)
        Relation-specific vector used to build projection matrices. See paper for more details.

    """

    def __init__(self, ent_emb_dim, rel_emb_dim, n_entities, n_relations):

        super().__init__(ent_emb_dim, n_entities, n_relations, dissimilarity='L2')

        self.rel_emb_dim = rel_emb_dim
        self.relation_embeddings = init_embedding(self.number_relations, self.rel_emb_dim)

        self.ent_proj_vects = Parameter(xavier_uniform_(empty(size=(n_entities, ent_emb_dim))),
                                        requires_grad=True)
        self.rel_proj_vects = Parameter(xavier_uniform_(empty(size=(n_relations, rel_emb_dim))),
                                        requires_grad=True)
        self.normalize_parameters()

        self.evaluated_projections = False
        self.projected_entities = Parameter(empty(size=(self.number_relations,
                                                        self.rel_emb_dim,
                                                        self.number_entities)), requires_grad=False)

    def scoring_function(self, heads_idx, tails_idx, rels_idx):
        """Compute the scoring function for the triplets given as argument.

        Parameters
        ----------
        heads_idx: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's heads
        tails_idx: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's tails.
        rels_idx: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's relations.

        Returns
        -------
        score: torch.Tensor, dtype = float, shape = (batch_size)
            Score function: opposite of dissimilarities between h+r and t after projection.

        """
        self.evaluated_projections = False

        # recover relations projection vectors and relations embeddings
        self.rel_proj_vects[rels_idx] = normalize(self.rel_proj_vects[rels_idx], p=2, dim=1)
        relations_embeddings = normalize(self.relation_embeddings(rels_idx), p=2, dim=1)

        # project
        projected_heads = self.recover_project_normalize(heads_idx, rels_idx, normalize_=True)
        projected_tails = self.recover_project_normalize(tails_idx, rels_idx, normalize_=True)

        return - self.dissimilarity(projected_heads + relations_embeddings, projected_tails)

    def recover_project_normalize(self, ent_idx, rel_idx, normalize_=True):
        """Recover entity (either head or tail) embeddings and project on hyperplane defined by\
        provided normal vectors.

        Parameters
        ----------
        ent_idx: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of entities
        rel_idx: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of relations
        normalize_: bool
            Whether entities embeddings should be normalized or not.

        Returns
        -------
        projections: torch.Tensor, dtype = float, shape = (batch_size, rel_emb_dim)
            Projection of the entities into relation-specific subspaces.

        """
        b_size = ent_idx.shape[0]

        # recover entities embeddings and projection vectors
        ent_emb = self.entity_embeddings(ent_idx)
        ent_proj = self.ent_proj_vects[ent_idx]

        if normalize_:
            ent_emb = normalize(ent_emb, p=2, dim=1)
            ent_proj = normalize(ent_proj, p=2, dim=1)

        # project entities into relation space
        rel_proj = self.rel_proj_vects[rel_idx]
        proj_mat = matmul(rel_proj.view((b_size, self.rel_emb_dim, 1)),
                          ent_proj.view((b_size, 1, self.ent_emb_dim)))

        if proj_mat.is_cuda:
            proj_mat += eye(n=self.rel_emb_dim, m=self.ent_emb_dim, device='cuda')
        else:
            proj_mat += eye(n=self.rel_emb_dim, m=self.ent_emb_dim)

        projection = matmul(proj_mat, ent_emb.view((b_size, self.ent_emb_dim, 1)))

        return projection.view((b_size, self.rel_emb_dim))

    def normalize_parameters(self):
        """Normalize the parameters of the model using the L2 norm.
        """
        self.entity_embeddings.weight.data = normalize(self.entity_embeddings.weight.data,
                                                       p=2, dim=1)
        self.relation_embeddings.weight.data = normalize(self.relation_embeddings.weight.data,
                                                         p=2, dim=1)
        self.ent_proj_vects.data = normalize(self.ent_proj_vects.data, p=2, dim=1)
        self.rel_proj_vects.data = normalize(self.rel_proj_vects.data, p=2, dim=1)

    def evaluate_projections(self):
        """Project all entities according to each relation.
        """
        # TODO turn this to batch computation

        if self.evaluated_projections:
            return

        print('Projecting entities in relations spaces.')

        for i in tqdm(range(self.number_entities)):
            ent_proj_vect = self.ent_proj_vects.data[i].view(1, -1)
            rel_proj_vects = self.rel_proj_vects.data.view(self.number_relations,
                                                           self.rel_emb_dim, 1)

            projection_matrices = matmul(rel_proj_vects, ent_proj_vect)

            if projection_matrices.is_cuda:
                id_mat = eye(n=self.rel_emb_dim, m=self.ent_emb_dim, device='cuda')
            else:
                id_mat = eye(n=self.rel_emb_dim, m=self.ent_emb_dim)

            id_mat = id_mat.view(1, self.rel_emb_dim, self.ent_emb_dim)

            projection_matrices += id_mat.expand(self.number_relations, self.rel_emb_dim,
                                                 self.ent_emb_dim)

            empty_cache()

            mask = tensor([i]).long()

            if self.entity_embeddings.weight.is_cuda:
                assert self.projected_entities.is_cuda
                empty_cache()
                mask = mask.cuda()

            entity = self.entity_embeddings(mask)
            projected_entity = matmul(projection_matrices, entity.view(-1)).detach()
            projected_entity = projected_entity.view(self.number_relations, self.rel_emb_dim, 1)
            self.projected_entities[:, :, i] = projected_entity.view(self.number_relations,
                                                                     self.rel_emb_dim)

            del projected_entity

        self.evaluated_projections = True

    def evaluation_helper(self, h_idx, t_idx, r_idx):
        """Project current entities and candidates into relation-specific sub-spaces.

        Parameters
        ----------
        h_idx: torch.Tensor, shape = (b_size), dtype = long
            Tensor containing indices of current head entities.
        t_idx: torch.Tensor, shape = (b_size), dtype = long
            Tensor containing indices of current tail entities.
        r_idx: torch Tensor, shape = (b_size), dtype = long
            Tensor containing indices of current relations.
        Returns
        -------
        proj_h_emb: torch Tensor, shape = (b_size, rel_emb_dim), dtype = float
            Tensor containing embeddings of current head entities projected in relation space.
        proj_t_emb: torch Tensor, shape = (b_size, rel_emb_dim), dtype = float
            Tensor containing embeddings of current tail entities projected in relation space.
        proj_candidates: torch Tensor, shape = (b_size, rel_emb_dim, n_entities), dtype = float
            Tensor containing all entities projected in each relation spaces (relations\
            corresponding to current batch's relations).
        r_emb: torch Tensor, shape = (b_size, rel_emb_dim), dtype = float
            Tensor containing current relations embeddings.

        """
        b_size = h_idx.shape[0]
        if not self.evaluated_projections:
            self.evaluate_projections()

        # recover relations embeddings and projected candidates
        r_emb = normalize(self.relation_embeddings(r_idx), p=2, dim=1)
        proj_candidates = self.projected_entities[r_idx]

        proj_h_emb, proj_t_emb = self.projection_helper(h_idx, t_idx, b_size, proj_candidates,
                                                        self.rel_emb_dim)

        return proj_h_emb, proj_t_emb, proj_candidates, r_emb


class TorusEModel(TransEModel):
    """Implementation of TorusE model detailed in 2018 paper by Ebisu and Ichise.

        References
        ----------
        * Takuma Ebisu and Ryutaro Ichise
        TorusE: Knowledge Graph Embedding on a Lie Group.
        In Proceedings of the 32nd AAAI Conference on Artificial Intelligence
        (New Orleans, LA, USA, Feb. 2018), AAAI Press, pp. 1819–1826.
        https://arxiv.org/abs/1711.05435

        Parameters
        ----------
        ent_emb_dim: int
            Dimension of the embedding of entities.
        n_entities: int
            Number of entities in the current data set.
        n_relations: int
            Number of relations in the current data set.
        dissimilarity: function
            Used to compute dissimilarities (e.g. L1 or L2 dissimilarities).

        Attributes
        ----------
        ent_emb_dim: int
            Dimension of the embedding of entities.
        number_entities: int
            Number of entities in the current data set.
        number_relations: int
            Number of relations in the current data set.
        dissimilarity: function
            Used to compute dissimilarities (e.g. L1 or L2 dissimilarities).
        entity_embeddings: torch Embedding, shape = (number_entities, ent_emb_dim)
            Contains the embeddings of the entities. It is initialized with Xavier uniform and then\
             normalized.
        relation_embeddings: torch Embedding, shape = (number_relations, ent_emb_dim)
            Contains the embeddings of the relations. It is initialized with Xavier uniform and\
            then normalized.

        """

    def __init__(self, ent_emb_dim, n_entities, n_relations, dissimilarity):

        assert dissimilarity in ['L1', 'L2', 'eL2']
        self.dissimilarity_type = dissimilarity

        super().__init__(ent_emb_dim, n_entities, n_relations, dissimilarity=None)

        self.relation_embeddings = init_embedding(self.number_relations, self.ent_emb_dim)

        if self.dissimilarity_type == 'L1':
            self.dissimilarity = l1_torus_dissimilarity
        if self.dissimilarity_type == 'L2':
            self.dissimilarity = l2_torus_dissimilarity
        if self.dissimilarity_type == 'eL2':
            self.dissimilarity = el2_torus_dissimilarity

        self.normalize_parameters()

    def scoring_function(self, heads_idx, tails_idx, rels_idx):
        """Compute the scoring function for the triplets given as argument.

        Parameters
        ----------
        heads_idx: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's heads
        tails_idx: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's tails.
        rels_idx: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's relations.

        Returns
        -------
        score: torch.Tensor, dtype = float, shape = (batch_size)
            Score function: opposite of dissimilarities between h+r and t.

        """

        # recover relations embeddings
        rels_emb = self.relation_embeddings(rels_idx)

        # recover, project and normalize entity embeddings
        h_emb = self.recover_project_normalize(heads_idx, normalize_=False)
        t_emb = self.recover_project_normalize(tails_idx, normalize_=False)

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
        ent_idx: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of entities
        rel_idx: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of relations
        normalize_: bool
            Whether entities embeddings should be normalized or not.

        Returns
        -------
        projections: torch.Tensor, dtype = float, shape = (batch_size, ent_emb_dim)
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
