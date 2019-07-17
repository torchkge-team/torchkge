# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
armand.boschin@telecom-paristech.fr
"""

from torch import empty, matmul, eye, arange, tensor
from torch.nn import Module, Parameter
from torch.nn.functional import normalize
from torch.nn.init import xavier_uniform_
from torch.cuda import empty_cache
from torchkge.utils import get_rank, init_embedding, l2_dissimilarity

from tqdm import tqdm


class TransEModel(Module):
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
    norm_type: int
        1 or 2 indicates the type of the norm to be used when normalizing.
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
    norm_type: int
        1 or 2 indicates the type of the norm to be used when normalizing.
    dissimilarity: function
        Used to compute dissimilarities (e.g. L1 or L2 dissimilarities).
    entity_embeddings: torch Embedding, shape = (number_entities, ent_emb_dim)
        Contains the embeddings of the entities. It is initialized with Xavier uniform and then\
         normalized.
    relation_embeddings: torch Embedding, shape = (number_relations, ent_emb_dim)
        Contains the embeddings of the relations. It is initialized with Xavier uniform and then\
         normalized.

    """

    def __init__(self, ent_emb_dim, n_entities, n_relations, norm_type, dissimilarity,
                 rel_emb_dim=None):
        super().__init__()

        self.ent_emb_dim = ent_emb_dim
        self.number_entities = n_entities
        self.number_relations = n_relations
        self.norm_type = norm_type
        self.dissimilarity = dissimilarity

        if rel_emb_dim is None:
            rel_emb_dim = ent_emb_dim
        else:
            self.rel_emb_dim = rel_emb_dim

        # initialize embeddings
        self.entity_embeddings = init_embedding(self.number_entities, self.ent_emb_dim)
        self.relation_embeddings = init_embedding(self.number_relations, rel_emb_dim)

        # normalize parameters
        self.entity_embeddings.weight.data = normalize(self.entity_embeddings.weight.data,
                                                       p=self.norm_type, dim=1)
        self.relation_embeddings.weight.data = normalize(self.relation_embeddings.weight.data,
                                                         p=self.norm_type, dim=1)

    def forward(self, heads, tails, negative_heads, negative_tails, relations):
        """Forward pass on the current batch.

        Parameters
        ----------
        heads: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's heads
        tails: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's tails.
        negative_heads: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's negatively sampled heads.
        negative_tails: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's negatively sampled tails.
        relations: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's relations.

        Returns
        -------
        golden_triplets: torch.Tensor, dtype = float, shape = (batch_size, rel_emb_dim)
            Score function: opposite of dissimilarities between h+r and t for golden triplets.
        negative_triplets: torch.Tensor, dtype = float, shape = (batch_size, rel_emb_dim)
            Score function: opposite of dissimilarities between h+r and t for negatively
            sampled triplets.

        """
        # recover, project and normalize entity embeddings
        h_emb = self.recover_project_normalize(heads, normalize_=True)
        t_emb = self.recover_project_normalize(tails, normalize_=True)
        n_h_emb = self.recover_project_normalize(negative_heads, normalize_=True)
        n_t_emb = self.recover_project_normalize(negative_tails, normalize_=True)

        # recover relations embeddings
        r_emb = self.relation_embeddings(relations)

        # compute dissimilarity
        golden_triplets = self.dissimilarity(h_emb + r_emb, t_emb)
        negative_triplets = self.dissimilarity(n_h_emb + r_emb, n_t_emb)

        return -golden_triplets, -negative_triplets

    def recover_project_normalize(self, ent_idx, normalize_=True, **kwargs):
        """

        Parameters
        ----------
        ent_idx: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of entities
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
            ent_emb = normalize(ent_emb, p=self.norm_type, dim=1)

        return ent_emb

    def normalize_parameters(self):
        """Normalize the parameters of the model using the model-specified norm.
        """
        self.entity_embeddings.weight.data = normalize(self.entity_embeddings.weight.data,
                                                       p=self.norm_type, dim=1)

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
        proj_candidates = self.recover_project_normalize(all_idx, normalize_=False)

        proj_h_emb = proj_candidates[h_idx]
        proj_t_emb = proj_candidates[t_idx]
        r_emb = self.relation_embeddings(r_idx)

        b_size, emb_dim = proj_h_emb.shape
        proj_candidates = proj_candidates.transpose(0, 1)
        proj_candidates = proj_candidates.view(1, emb_dim, self.number_entities)
        proj_candidates = proj_candidates.expand(b_size, emb_dim, self.number_entities)

        return proj_h_emb, proj_t_emb, proj_candidates, r_emb

    def compute_ranks(self, proj_e_emb, proj_candidates,
                      r_emb, e_idx, r_idx, true_idx, dictionary, heads=1):
        """

        Parameters
        ----------
        proj_e_emb: torch.Tensor, shape = (batch_size, rel_emb_dim), dtype = float
            Tensor containing current projected embeddings of entities.
        proj_candidates: torch.Tensor, shape = (b_size, rel_emb_dim, n_entities), dtype = float
            Tensor containing projected embeddings of all entities.
        r_emb: torch.Tensor, shape = (batch_size, ent_emb_dim), dtype = float
            Tensor containing current embeddings of relations.
        e_idx: torch.Tensor, shape = (batch_size), dtype = long
            Tensor containing the indices of entities.
        r_idx: torch.Tensor, shape = (batch_size), dtype = long
            Tensor containing the indices of relations.
        true_idx: torch.Tensor, shape = (batch_size), dtype = long
            Tensor containing the true entity for each sample.
        dictionary: default dict
            Dictionary of keys (int, int) and values list of ints giving all possible entities for
            the (entity, relation) pair.
        heads: integer
            1 ou -1 (must be 1 if entities are heads and -1 if entities are tails). \
            We test dissimilarity between heads * entities + relations and heads * targets.


        Returns
        -------
        rank_true_entities: torch.Tensor, shape = (b_size), dtype = int
            Tensor containing the rank of the true entities when ranking any entity based on \
            computation of d(hear+relation, tail).
        filtered_rank_true_entities: torch.Tensor, shape = (b_size), dtype = int
            Tensor containing the rank of the true entities when ranking only true false entities \
            based on computation of d(hear+relation, tail).

        """
        current_batch_size, embedding_dimension = proj_e_emb.shape

        # tmp_sum is either heads + r_emb or r_emb - tails (expand does not use extra memory)
        tmp_sum = (heads * proj_e_emb + r_emb).view((current_batch_size, embedding_dimension, 1))
        tmp_sum = tmp_sum.expand((current_batch_size, embedding_dimension, self.number_entities))

        # compute either dissimilarity(heads + relation, proj_candidates) or
        # dissimilarity(-proj_candidates, relation - tails)
        dissimilarities = self.dissimilarity(tmp_sum, heads * proj_candidates)

        # filter out the true negative samples by assigning infinite dissimilarity
        filt_dissimilarities = dissimilarities.clone()
        for i in range(current_batch_size):
            true_targets = dictionary[e_idx[i].item(), r_idx[i].item()].copy()
            if len(true_targets) == 1:
                continue
            true_targets.remove(true_idx[i].item())
            true_targets = tensor(true_targets).long()
            filt_dissimilarities[i][true_targets] = float('Inf')

        # from dissimilarities, extract the rank of the true entity.
        rank_true_entities = get_rank(-dissimilarities, true_idx)
        filtered_rank_true_entities = get_rank(-filt_dissimilarities, true_idx)

        return rank_true_entities, filtered_rank_true_entities

    def evaluate_candidates(self, h_idx, t_idx, r_idx, kg):
        proj_h_emb, proj_t_emb, proj_candidates, r_emb = self.evaluation_helper(h_idx, t_idx, r_idx)

        # evaluation_helper both ways (head, rel) -> tail and (rel, tail) -> head
        rank_true_tails, filt_rank_true_tails = self.compute_ranks(proj_h_emb,
                                                                   proj_candidates,
                                                                   r_emb, h_idx, r_idx,
                                                                   t_idx,
                                                                   kg.dict_of_tails,
                                                                   heads=1)
        rank_true_heads, filt_rank_true_heads = self.compute_ranks(proj_t_emb,
                                                                   proj_candidates,
                                                                   r_emb, t_idx, r_idx,
                                                                   h_idx,
                                                                   kg.dict_of_heads,
                                                                   heads=-1)

        return rank_true_tails, filt_rank_true_tails, rank_true_heads, filt_rank_true_heads


class TransHModel(TransEModel):
    """Implementation of TransH model detailed in 2014 paper by Wang et al.. According to this\
    paper, both normalization and dissimilarity measure are by default L2.

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
    norm_type: int (default=2)
        1 or 2 indicates the type of the norm to be used when normalizing. Default is 2.
    dissimilarity: function (default=torchkge.utils.dissimilarities.l2_dissimilarity)
        Used to compute dissimilarities (e.g. L1 or L2 dissimilarities). \
        Default is l2_dissimilarity.

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
    normal_vectors: torch.Tensor, shape = (number_relations, ent_emb_dim)
        Normal vectors associated to each relation and used to compute the relation-specific\
        hyperplanes entities are projected on. See paper for more details.

    """

    def __init__(self, ent_emb_dim, n_entities, n_relations, norm_type=2,
                 dissimilarity=l2_dissimilarity):

        super().__init__(ent_emb_dim, n_entities, n_relations, norm_type, dissimilarity)

        self.normal_vectors = Parameter(xavier_uniform_(empty(size=(n_relations, ent_emb_dim))))
        self.normalize_parameters()

    def forward(self, heads, tails, negative_heads, negative_tails, relations):
        """Forward pass on the current batch.

        Parameters
        ----------
        heads: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's heads
        tails: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's tails.
        negative_heads: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's negatively sampled heads.
        negative_tails: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's negatively sampled tails.
        relations: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's relations.

        Returns
        -------
        golden_triplets: torch.Tensor, dtype = float, shape = (batch_size, rel_emb_dim)
            Score function: opposite of dissimilarities between h+r and t for golden triplets.
        negative_triplets: torch.Tensor, dtype = float, shape = (batch_size, rel_emb_dim)
            Score function: opposite of dissimilarities between h+r and t for negatively
            sampled triplets.

        """
        # recover relations embeddings and normal projection vectors
        relations_embeddings = self.relation_embeddings(relations)
        normal_vectors = normalize(self.normal_vectors[relations], p=2, dim=1)

        # project entities in relation specific hyperplane

        projected_heads = self.recover_project_normalize(heads, normalize_=True,
                                                         normal_vectors=normal_vectors)
        projected_tails = self.recover_project_normalize(tails, normalize_=True,
                                                         normal_vectors=normal_vectors)
        projected_neg_heads = self.recover_project_normalize(negative_heads, normalize_=True,
                                                             normal_vectors=normal_vectors)
        projected_neg_tails = self.recover_project_normalize(negative_tails, normalize_=True,
                                                             normal_vectors=normal_vectors)

        # compute dissimilarities
        golden_triplets = self.dissimilarity(projected_heads + relations_embeddings,
                                             projected_tails)
        negative_triplets = self.dissimilarity(projected_neg_heads + relations_embeddings,
                                               projected_neg_tails)

        return -golden_triplets, -negative_triplets

    def recover_project_normalize(self, ent_idx, normalize_=True, **kwargs):
        """Recover entity (either head or tail) embeddings and project on hyperplane defined by\
        provided normal vectors.

        Parameters
        ----------
        ent_idx: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of entities
        normalize_: bool
            Whether entities embeddings should be normalized or not.
        normal_vectors: torch.Tensor, dtype = float, shape = (batch_size, ent_emb_dim)
            Normal vectors relative to the current relations.

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
            ent_emb = normalize(ent_emb, p=self.norm_type, dim=1)

        # project entities into relation space
        normal_vectors = kwargs['normal_vectors']
        normal_component = (ent_emb * normal_vectors).sum(dim=1).view((-1, 1))

        return ent_emb - normal_component * normal_vectors

    def normalize_parameters(self):
        """Normalize the parameters of the model using norm defined in self.norm_type. Default\
        is L2.
        """
        self.normal_vectors.data = normalize(self.normal_vectors, p=self.norm_type, dim=1)
        self.entity_embeddings.weight.data = normalize(self.entity_embeddings.weight.data,
                                                       p=self.norm_type, dim=1)

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
        all_idx = arange(0, self.number_entities).long()
        if h_idx.is_cuda:
            all_idx = all_idx.cuda()
        candidates = self.entity_embeddings(all_idx).transpose(0, 1)
        candidates = candidates.view((1,
                                      self.ent_emb_dim,
                                      self.number_entities)).expand((b_size,
                                                                     self.ent_emb_dim,
                                                                     self.number_entities))

        # project each candidates with each normal vector
        normal_components = candidates * normal_vectors.view((b_size, self.ent_emb_dim, 1))
        normal_components = normal_components.sum(dim=1).view(b_size, 1, self.number_entities)
        normal_components = normal_components * normal_vectors.view(b_size, self.ent_emb_dim, 1)
        proj_candidates = candidates - normal_components

        assert proj_candidates.shape == (b_size, self.ent_emb_dim, self.number_entities)

        # recover, project and normalize entity embeddings
        mask = h_idx.view(b_size, 1, 1).expand(b_size, self.ent_emb_dim, 1)
        proj_h_emb = proj_candidates.gather(dim=2, index=mask).view(b_size, self.ent_emb_dim)

        mask = t_idx.view(b_size, 1, 1).expand(b_size, self.ent_emb_dim, 1)
        proj_t_emb = proj_candidates.gather(dim=2, index=mask).view(b_size, self.ent_emb_dim)

        return proj_h_emb, proj_t_emb, proj_candidates, r_emb


class TransRModel(TransEModel):
    """Implementation of TransR model detailed in 2015 paper by Lin et al.. According to this\
    paper, both normalization and dissimilarity measure are by default L2.

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
    norm_type: int (default=2)
        1 or 2 indicates the type of the norm to be used when normalizing. Default is 2.
    dissimilarity: function (default=torchkge.utils.dissimilarities.l2_dissimilarity)
        Used to compute dissimilarities (e.g. L1 or L2 dissimilarities). \
        Default is l2_dissimilarity.

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
    projection_matrices: torch.Tensor, shape = (number_relations, rel_emb_dim, ent_emb_dim)
        Relation-specific projection matrices. See paper for more details.

    """

    def __init__(self, ent_emb_dim, rel_emb_dim, n_entities, n_relations, norm_type=2,
                 dissimilarity=l2_dissimilarity):

        super().__init__(ent_emb_dim, n_entities, n_relations, norm_type, dissimilarity,
                         rel_emb_dim=rel_emb_dim)
        self.projection_matrices = Parameter(xavier_uniform_(empty(size=(n_relations, rel_emb_dim,
                                                                         ent_emb_dim))))
        self.normalize_parameters()

    def forward(self, heads, tails, negative_heads, negative_tails, relations):
        """Forward pass on the current batch.

        Parameters
        ----------
        heads: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's heads
        tails: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's tails.
        negative_heads: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's negatively sampled heads.
        negative_tails: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's negatively sampled tails.
        relations: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's relations.

        Returns
        -------
        golden_triplets: torch.Tensor, dtype = float, shape = (batch_size, rel_emb_dim)
            Score function: opposite of dissimilarities between h+r and t for golden triplets.
        negative_triplets: torch.Tensor, dtype = float, shape = (batch_size, rel_emb_dim)
            Score function: opposite of dissimilarities between h+r and t for negatively
            sampled triplets.

        """
        # recover relations embeddings and normal projection matrices
        relations_embeddings = normalize(self.relation_embeddings(relations), p=2, dim=1)
        projection_matrices = normalize(self.projection_matrices[relations], p=2, dim=2)

        # project entities in relation specific hyperplane
        projected_heads = self.recover_project_normalize(heads, normalize_=True,
                                                         projection_matrices=projection_matrices)
        projected_tails = self.recover_project_normalize(tails, normalize_=True,
                                                         projection_matrices=projection_matrices)
        projected_neg_heads = self.recover_project_normalize(negative_heads, normalize_=True,
                                                             projection_matrices=projection_matrices
                                                             )
        projected_neg_tails = self.recover_project_normalize(negative_tails, normalize_=True,
                                                             projection_matrices=projection_matrices
                                                             )

        # compute dissimilarities
        golden_triplets = self.dissimilarity(projected_heads + relations_embeddings,
                                             projected_tails)
        negative_triplets = self.dissimilarity(projected_neg_heads + relations_embeddings,
                                               projected_neg_tails)
        return -golden_triplets, -negative_triplets

    def recover_project_normalize(self, ent_idx, normalize_=True, **kwargs):
        """Recover entity (either head or tail) embeddings and project on hyperplane defined by\
        provided projection matrices.

        Parameters
        ----------
        ent_idx: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of entities
        normalize_: bool
            Whether entities embeddings should be normalized or not.
        projection_matrices: torch.Tensor, dtype = float, shape = (b_size, r_emb_dim, e_emb_dim)
            Projection matrices for the current relations.

        Returns
        -------
        projections: torch.Tensor, dtype = float, shape = (batch_size, rel_emb_dim)
            Projection of the entities into relation-specific subspaces.
        """
        b_size = len(ent_idx)
        # recover and normalize embeddings
        ent_emb = self.entity_embeddings(ent_idx)
        if normalize_:
            ent_emb = normalize(ent_emb, p=2, dim=1)

        # project entities into relation space
        new_shape = (b_size, self.ent_emb_dim, 1)
        projection_matrices = kwargs['projection_matrices']
        projection = matmul(projection_matrices, ent_emb.view(new_shape))

        return projection.view(b_size, self.rel_emb_dim)

    def normalize_parameters(self):
        """Normalize the parameters of the model using norm defined in self.norm_type. Default\
        is L2.
        """
        self.entity_embeddings.weight.data = normalize(self.entity_embeddings.weight.data,
                                                       p=self.norm_type, dim=1)
        self.relation_embeddings.weight.data = normalize(self.relation_embeddings.weight.data,
                                                         p=self.norm_type, dim=1)
        self.projection_matrices.data = normalize(self.projection_matrices.data, p=self.norm_type,
                                                  dim=2)

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
        all_idx = arange(0, self.number_entities).long()
        if h_idx.is_cuda:
            all_idx = all_idx.cuda()
        candidates = self.entity_embeddings(all_idx).transpose(0, 1)
        candidates = candidates.view((1,
                                      self.ent_emb_dim,
                                      self.number_entities)).expand((b_size,
                                                                     self.ent_emb_dim,
                                                                     self.number_entities))

        # project each candidates with each projection matrix
        proj_candidates = matmul(projection_matrices, candidates)

        mask = h_idx.view(b_size, 1, 1).expand(b_size, self.rel_emb_dim, 1)
        proj_h_emb = proj_candidates.gather(dim=2, index=mask).view(b_size, self.rel_emb_dim)

        mask = t_idx.view(b_size, 1, 1).expand(b_size, self.rel_emb_dim, 1)
        proj_t_emb = proj_candidates.gather(dim=2, index=mask).view(b_size, self.rel_emb_dim)

        return proj_h_emb, proj_t_emb, proj_candidates, r_emb


class TransDModel(TransEModel):
    """Implementation of TransD model detailed in 2015 paper by Ji et al.. According to this\
    paper, both normalization and dissimilarity measure are by default L2.

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
    norm_type: int (default=2)
        1 or 2 indicates the type of the norm to be used when normalizing. Default is 2.
    dissimilarity: function (default=torchkge.utils.dissimilarities.l2_dissimilarity)
        Used to compute dissimilarities (e.g. L1 or L2 dissimilarities). \
        Default is l2_dissimilarity.

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
    ent_proj_vects: torch.Tensor, shape = (number_entities, ent_emb_dim)
        Entity-specific vector used to build projection matrices. See paper for more details.
    rel_proj_vects: torch.Tensor, shape = (number_relations, rel_emb_dim)
        Relation-specific vector used to build projection matrices. See paper for more details.

    """

    def __init__(self, ent_emb_dim, rel_emb_dim, n_entities, n_relations, norm_type=2,
                 dissimilarity=l2_dissimilarity):


        super().__init__(ent_emb_dim, n_entities, n_relations, norm_type, dissimilarity,
                         rel_emb_dim=rel_emb_dim)

        self.ent_proj_vects = Parameter(xavier_uniform_(empty(size=(n_entities, ent_emb_dim))))
        self.rel_proj_vects = Parameter(xavier_uniform_(empty(size=(n_relations, rel_emb_dim))))
        self.normalize_parameters()

        self.evaluated_projections = False
        self.projected_entities = Parameter(empty(size=(self.number_relations,
                                                        self.rel_emb_dim,
                                                        self.number_entities)), requires_grad=False)

    def forward(self, heads, tails, negative_heads, negative_tails, relations):
        """Forward pass on the current batch.

        Parameters
        ----------
        heads: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's heads
        tails: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's tails.
        negative_heads: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's negatively sampled heads.
        negative_tails: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's negatively sampled tails.
        relations: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's relations.

        Returns
        -------
        golden_triplets: torch.Tensor, dtype = float, shape = (batch_size, rel_emb_dim)
            Score function: opposite of dissimilarities between h+r and t for golden triplets.
        negative_triplets: torch.Tensor, dtype = float, shape = (batch_size, rel_emb_dim)
            Score function: opposite of dissimilarities between h+r and t for negatively
            sampled triplets.

        """
        self.evaluated_projections = False

        # recover relations projection vectors and relations embeddings
        rel_proj = normalize(self.rel_proj_vects[relations], p=2, dim=1)
        relations_embeddings = normalize(self.relation_embeddings(relations), p=2, dim=1)

        # project
        projected_heads = self.recover_project_normalize(heads, normalize_=True, rel_proj=rel_proj)
        projected_tails = self.recover_project_normalize(tails, normalize_=True, rel_proj=rel_proj)
        projected_neg_heads = self.recover_project_normalize(negative_heads,
                                                             normalize_=True, rel_proj=rel_proj)
        projected_neg_tails = self.recover_project_normalize(negative_tails,
                                                             normalize_=True, rel_proj=rel_proj)

        # compute dissimilarities
        golden_triplets = self.dissimilarity(projected_heads + relations_embeddings,
                                             projected_tails)
        negative_triplets = self.dissimilarity(projected_neg_heads + relations_embeddings,
                                               projected_neg_tails)

        return - golden_triplets, - negative_triplets

    def recover_project_normalize(self, ent_idx, normalize_=True, **kwargs):
        """Recover entity (either head or tail) embeddings and project on hyperplane defined by\
        provided normal vectors.

        Parameters
        ----------
        ent_idx: torch.Tensor, dtype = long, shape = (batch_size)
            Integer keys of entities
        normalize_: bool
            Whether entities embeddings should be normalized or not.
        rel_proj: torch.Tensor, dtype = float, shape = (batch_size, rel_emb_dim)
            Projection vectors for the current relations.

        Returns
        -------
        projections: torch.Tensor, dtype = float, shape = (batch_size, rel_emb_dim)
            Projection of the entities into relation-specific subspaces.

        """
        b_size = len(ent_idx)

        # recover entities embeddings and projection vectors
        ent_emb = self.entity_embeddings(ent_idx)
        ent_proj = self.ent_proj_vects[ent_idx]

        if normalize_:
            ent_emb = normalize(ent_emb, p=2, dim=1)
            ent_proj = normalize(ent_proj, p=2, dim=1)

        # project entities into relation space
        rel_proj = kwargs['rel_proj']
        proj_mat = matmul(rel_proj.view((b_size, self.rel_emb_dim, 1)),
                          ent_proj.view((b_size, 1, self.ent_emb_dim)))

        if proj_mat.is_cuda:
            proj_mat += eye(n=self.rel_emb_dim, m=self.ent_emb_dim, device='cuda')
        else:
            proj_mat += eye(n=self.rel_emb_dim, m=self.ent_emb_dim)

        projection = matmul(proj_mat, ent_emb.view((b_size, self.ent_emb_dim, 1)))

        return projection.view((b_size, self.rel_emb_dim))

    def normalize_parameters(self):
        """Normalize the parameters of the model using norm defined in self.norm_type. Default\
        is L2.
        """
        self.entity_embeddings.weight.data = normalize(self.entity_embeddings.weight.data,
                                                       p=self.norm_type, dim=1)
        self.relation_embeddings.weight.data = normalize(self.relation_embeddings.weight.data,
                                                         p=self.norm_type, dim=1)
        self.ent_proj_vects.data = normalize(self.ent_proj_vects.data, p=self.norm_type, dim=1)
        self.rel_proj_vects.data = normalize(self.rel_proj_vects.data, p=self.norm_type, dim=1)

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

            entity = self.entity_embeddings(mask.cuda())
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
        b_size = len(h_idx)
        if not self.evaluated_projections:
            self.evaluate_projections()

        # recover relations embeddings and projected candidates
        r_emb = normalize(self.relation_embeddings(r_idx), p=2, dim=1)
        proj_candidates = self.projected_entities[r_idx]

        mask = h_idx.view(b_size, 1, 1).expand(b_size, self.rel_emb_dim, 1)
        proj_h_emb = proj_candidates.gather(dim=2, index=mask).view(b_size, self.rel_emb_dim)

        mask = t_idx.view(b_size, 1, 1).expand(b_size, self.rel_emb_dim, 1)
        proj_t_emb = proj_candidates.gather(dim=2, index=mask).view(b_size, self.rel_emb_dim)

        return proj_h_emb, proj_t_emb, proj_candidates, r_emb
