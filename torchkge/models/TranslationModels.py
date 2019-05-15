# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
armand.boschin@telecom-paristech.fr
"""

from torch import empty, matmul, eye, arange, tensor
from torch.nn import Module, Parameter, Embedding
from torch.nn.functional import normalize
from torch.nn.init import xavier_uniform_
from torch.cuda import empty_cache

from tqdm import tqdm


class TransEModel(Module):
    """Implement torch.nn.Module interface.

    Parameters
    ----------

    config : Config object
        Contains all configuration parameters.
    dissimilarity : function
        Used to compute dissimilarities.

    Attributes
    ----------

    ent_emb_dim : int
        Dimension of the embedding of entities
    rel_emb_dim : int
        Dimension of the embedding of relations
    number_entities : int
        Number of entities in the current data set.
    norm_type : int
        1 or 2 indicates the type of the norm to be used when normalizing.
    dissimilarity : function
        Used to compute dissimilarities.
    entity_embeddings : torch Embedding, shape = (number_entities, ent_emb_dim)
        Contains the embeddings of the entities. It is initialized with Xavier uniform and then\
         normalized.
    relation_embeddings : torch Embedding, shape = (number_relations, ent_emb_dim)
        Contains the embeddings of the relations. It is initialized with Xavier uniform and then\
         normalized.

    """

    def __init__(self, config, dissimilarity):
        super().__init__()

        self.ent_emb_dim = config.entities_embedding_dimension
        self.rel_emb_dim = config.relations_embedding_dimension
        self.number_entities = config.number_entities
        self.number_relations = config.number_relations
        self.norm_type = config.norm_type
        self.dissimilarity = dissimilarity

        # initialize embedding objects
        self.entity_embeddings = Embedding(self.number_entities, self.ent_emb_dim)
        self.relation_embeddings = Embedding(self.number_relations, self.rel_emb_dim)

        # fill the embedding weights with Xavier initialized values
        self.entity_embeddings.weight = Parameter(xavier_uniform_(
            empty(size=(self.number_entities, self.ent_emb_dim))))
        self.relation_embeddings.weight = Parameter(xavier_uniform_(
            empty(size=(self.number_relations, self.rel_emb_dim))))

        # normalize the embeddings
        self.entity_embeddings.weight.data = normalize(self.entity_embeddings.weight.data,
                                                       p=self.norm_type, dim=1)
        self.relation_embeddings.weight.data = normalize(self.relation_embeddings.weight.data,
                                                         p=self.norm_type, dim=1)

    def forward(self, heads, tails, negative_heads, negative_tails, relations):
        """Forward pass on the current batch.

        Parameters
        ----------

        heads : torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's heads
        tails : torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's tails.
        negative_heads : torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's negatively sampled heads.
        negative_tails : torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's negatively sampled tails.
        relations : torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's relations.

        Returns
        -------

        golden_triplets : torch tensor, dtype = float, shape = (batch_size, rel_emb_dim)
            Dissimilarities between h+r and t for golden triplets.
        negative_triplets : torch tensor, dtype = float, shape = (batch_size, rel_emb_dim)
            Dissimilarities between h+r and t for negatively sampled triplets.
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

        return golden_triplets, negative_triplets

    def recover_project_normalize(self, ent_idx, normalize_=True, **kwargs):
        """

        Parameters
        ----------
        ent_idx : torch tensor, dtype = long, shape = (batch_size)
            Integer keys of entities
        normalize_ : bool
            Whether entities embeddings should be normalized or not.

        Returns
        -------
        projections : torch tensor, dtype = float, shape = (batch_size, ent_emb_dim)
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

    def evaluate(self, h_idx, t_idx, r_idx):
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


class TransHModel(TransEModel):
    """Implement torch.nn.Module interface and inherits torchkge.models.translational_models.TransE.

    Parameters
    ----------

    config : Config object
        Contains all configuration parameters.
    dissimilarity : function
        Used to compute dissimilarities.

    Attributes
    ----------

    ent_emb_dim : int
        Dimension of the embedding of entities
    rel_emb_dim : int
        Dimension of the embedding of relations
    number_entities : int
        Number of entities in the current data set.
    norm_type : int
        1 or 2 indicates the type of the norm to be used when normalizing.
    dissimilarity : function
        Used to compute dissimilarities.
    entity_embeddings : torch Embedding, shape = (number_entities, ent_emb_dim)
        Contains the embeddings of the entities. It is initialized with Xavier uniform and then\
         normalized.
    relation_embeddings : torch Embedding, shape = (number_relations, ent_emb_dim)
        Contains the embeddings of the relations. It is initialized with Xavier uniform and then\
         normalized.

    """
    def __init__(self, config, dissimilarity):
        # initialize and normalize embeddings
        super().__init__(config, dissimilarity)

        # initialize and normalize normal vector
        self.normal_vectors = Parameter(xavier_uniform_(empty(size=(self.number_relations,
                                                                    self.ent_emb_dim))))
        self.normal_vectors.data = normalize(self.normal_vectors.data, p=2, dim=1)

    def forward(self, heads, tails, negative_heads, negative_tails, relations):
        """Forward pass on the current batch.

        Parameters
        ----------

        heads : torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's heads
        tails : torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's tails.
        negative_heads : torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's negatively sampled heads.
        negative_tails : torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's negatively sampled tails.
        relations : torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's relations.

        Returns
        -------

        golden_triplets : torch tensor, dtype = float, shape = (batch_size, rel_emb_dim)
            Dissimilarities between h+r and t for golden triplets.
        negative_triplets : torch tensor, dtype = float, shape = (batch_size, rel_emb_dim)
            Dissimilarities between h+r and t for negatively sampled triplets.
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

        return golden_triplets, negative_triplets

    def recover_project_normalize(self, ent_idx, normalize_=True, **kwargs):
        """Recover entity (either head or tail) embeddings and project on hyperplane defined by\
        provided normal vectors.

        Parameters
        ----------

        ent_idx : torch tensor, dtype = long, shape = (batch_size)
            Integer keys of entities
        normalize_ : bool
            Whether entities embeddings should be normalized or not.
        normal_vectors : torch tensor, dtype = float, shape = (batch_size, ent_emb_dim)
            Normal vectors relative to the current relations.

        Returns
        -------

        projections : torch tensor, dtype = float, shape = (batch_size, ent_emb_dim)
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
        """Normalize the embeddings of the entities using the model-specified norm and the normal \
        vectors using the L2 norm.
        """
        self.entity_embeddings.weight.data = normalize(self.entity_embeddings.weight.data,
                                                       p=self.norm_type, dim=1)
        self.normal_vectors.data = normalize(self.normal_vectors, p=2, dim=1)

    def evaluate(self, h_idx, t_idx, r_idx):
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
    """Implement torch.nn.Module interface and inherits torchkge.models.translational_models.TransE.

    Parameters
    ----------

    config : Config object
        Contains all configuration parameters.
    dissimilarity : function
        Used to compute dissimilarities.

    Attributes
    ----------

    ent_emb_dim : int
        Dimension of the embedding of entities
    rel_emb_dim : int
        Dimension of the embedding of relations
    number_entities : int
        Number of entities in the current data set.
    norm_type : int
        1 or 2 indicates the type of the norm to be used when normalizing.
    dissimilarity : function
        Used to compute dissimilarities.
    entity_embeddings : torch Embedding, shape = (number_entities, ent_emb_dim)
        Contains the embeddings of the entities. It is initialized with Xavier uniform and then\
         normalized.
    relation_embeddings : torch Embedding, shape = (number_relations, ent_emb_dim)
        Contains the embeddings of the relations. It is initialized with Xavier uniform and then\
         normalized.

        """
    def __init__(self, config, dissimilarity):
        super().__init__(config, dissimilarity)

        # initialize and normalize projection matrices
        self.projection_matrices = Parameter(xavier_uniform_(empty(size=(self.number_relations,
                                                                         self.rel_emb_dim,
                                                                         self.ent_emb_dim))))

    def forward(self, heads, tails, negative_heads, negative_tails, relations):
        """Forward pass on the current batch.

        Parameters
        ----------

        heads : torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's heads
        tails : torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's tails.
        negative_heads : torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's negatively sampled heads.
        negative_tails : torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's negatively sampled tails.
        relations : torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's relations.

        Returns
        -------

        golden_triplets : torch tensor, dtype = float, shape = (batch_size, rel_emb_dim)
            Dissimilarities between h+r and t for golden triplets.
        negative_triplets : torch tensor, dtype = float, shape = (batch_size, rel_emb_dim)
            Dissimilarities between h+r and t for negatively sampled triplets.
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
        return golden_triplets, negative_triplets

    def recover_project_normalize(self, ent_idx, normalize_=True, **kwargs):
        """Recover entity (either head or tail) embeddings and project on hyperplane defined by\
        provided projection matrices.

        Parameters
        ----------

        ent_idx : torch tensor, dtype = long, shape = (batch_size)
            Integer keys of entities
        normalize_ : bool
            Whether entities embeddings should be normalized or not.
        projection_matrices : torch tensor, dtype = float, shape = (b_size, r_emb_dim, e_emb_dim)
            Projection matrices for the current relations.

        Returns
        -------

        projections : torch tensor, dtype = float, shape = (batch_size, rel_emb_dim)
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
        """Normalize the parameters of the model using only L2 norm.
        """
        self.entity_embeddings.weight.data = normalize(self.entity_embeddings.weight.data,
                                                       p=2, dim=1)
        self.relation_embeddings.weight.data = normalize(self.relation_embeddings.weight.data,
                                                         p=2, dim=1)
        self.projection_matrices.data = normalize(self.projection_matrices.data, p=2, dim=2)

    def evaluate(self, h_idx, t_idx, r_idx):
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
    """Implement torch.nn.Module interface and inherits torchkge.models.translational_models.TransE.

    Parameters
    ----------

    config : Config object
        Contains all configuration parameters.
    dissimilarity : function
        Used to compute dissimilarities.

    Attributes
    ----------

    ent_emb_dim : int
        Dimension of the embedding of entities
    rel_emb_dim : int
        Dimension of the embedding of relations
    number_entities : int
        Number of entities in the current data set.
    norm_type : int
        1 or 2 indicates the type of the norm to be used when normalizing.
    dissimilarity : function
        Used to compute dissimilarities.
    entity_embeddings : torch Embedding, shape = (number_entities, ent_emb_dim)
        Contains the embeddings of the entities. It is initialized with Xavier uniform and then\
         normalized.
    relation_embeddings : torch Embedding, shape = (number_relations, ent_emb_dim)
        Contains the embeddings of the relations. It is initialized with Xavier uniform and then\
         normalized.

    """
    def __init__(self, config, dissimilarity):
        super().__init__(config, dissimilarity)

        # initialize and normalize projection vectors
        self.ent_proj_vects = Parameter(xavier_uniform_(empty(size=(self.number_entities,
                                                                    self.ent_emb_dim))))
        self.rel_proj_vects = Parameter(xavier_uniform_(empty(size=(self.number_relations,
                                                                    self.rel_emb_dim))))

        self.ent_proj_vects.data = normalize(self.ent_proj_vects.data, p=2, dim=1)
        self.rel_proj_vects.data = normalize(self.rel_proj_vects.data, p=2, dim=1)

        self.evaluated_projections = False
        self.projected_entities = Parameter(empty(size=(self.number_relations,
                                                        self.rel_emb_dim,
                                                        self.number_entities)), requires_grad=False)

    def forward(self, heads, tails, negative_heads, negative_tails, relations):
        """Forward pass on the current batch.

        Parameters
        ----------

        heads : torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's heads
        tails : torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's tails.
        negative_heads : torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's negatively sampled heads.
        negative_tails : torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's negatively sampled tails.
        relations : torch tensor, dtype = long, shape = (batch_size)
            Integer keys of the current batch's relations.

        Returns
        -------

        golden_triplets : torch tensor, dtype = float, shape = (batch_size, rel_emb_dim)
            Dissimilarities between h+r and t for golden triplets.
        negative_triplets : torch tensor, dtype = float, shape = (batch_size, rel_emb_dim)
            Dissimilarities between h+r and t for negatively sampled triplets.
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

        return golden_triplets, negative_triplets

    def recover_project_normalize(self, ent_idx, normalize_=True, **kwargs):
        """Recover entity (either head or tail) embeddings and project on hyperplane defined by\
        provided normal vectors.

        Parameters
        ----------

        ent_idx : torch tensor, dtype = long, shape = (batch_size)
            Integer keys of entities
        normalize_ : bool
            Whether entities embeddings should be normalized or not.
        rel_proj : torch tensor, dtype = float, shape = (batch_size, rel_emb_dim)
            Projection vectors for the current relations.

        Returns
        -------

        projections : torch tensor, dtype = float, shape = (batch_size, rel_emb_dim)
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
        """Normalize the parameters of the model using only L2 norm.
        """
        self.entity_embeddings.weight.data = normalize(self.entity_embeddings.weight.data,
                                                       p=2, dim=1)
        self.relation_embeddings.weight.data = normalize(self.relation_embeddings.weight.data,
                                                         p=2, dim=1)
        self.ent_proj_vects.data = normalize(self.ent_proj_vects.data, p=2, dim=1)
        self.rel_proj_vects.data = normalize(self.rel_proj_vects.data, p=2, dim=1)

    def evaluate_projections(self):
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

            projection_matrices += id_mat.expand(self.number_relations, self.rel_emb_dim, self.ent_emb_dim)

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

    def evaluate(self, h_idx, t_idx, r_idx):
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
