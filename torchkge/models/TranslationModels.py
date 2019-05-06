# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
armand.boschin@telecom-paristech.fr
"""

from torch import Tensor, matmul, eye
from torch.nn import Module, Parameter, Embedding
from torch.nn.functional import normalize
from torch.nn.init import xavier_uniform_


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
        self.entity_embeddings.weight = Parameter(xavier_uniform_(Tensor(self.number_entities,
                                                                         self.ent_emb_dim)))
        self.relation_embeddings.weight = Parameter(xavier_uniform_(Tensor(self.number_relations,
                                                                           self.rel_emb_dim)))

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
        # recover and normalize entities embeddings
        heads_embeddings = normalize(self.entity_embeddings(heads), p=self.norm_type, dim=1)
        tails_embeddings = normalize(self.entity_embeddings(tails), p=self.norm_type, dim=1)
        neg_heads_embeddings = normalize(self.entity_embeddings(negative_heads),
                                         p=self.norm_type, dim=1)
        neg_tails_embeddings = normalize(self.entity_embeddings(negative_tails),
                                         p=self.norm_type, dim=1)

        # recover relations embeddings
        relations_embeddings = self.relation_embeddings(relations)

        # compute dissimilarities
        golden_triplets = self.dissimilarity(heads_embeddings + relations_embeddings,
                                             tails_embeddings)
        negative_triplets = self.dissimilarity(neg_heads_embeddings + relations_embeddings,
                                               neg_tails_embeddings)

        return golden_triplets, negative_triplets

    def normalize_parameters(self):
        """Normalize the parameters of the model using the model-specified norm.
        """
        self.entity_embeddings.weight.data = normalize(self.entity_embeddings.weight.data,
                                                       p=self.norm_type, dim=1)


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
        self.normal_vectors = Parameter(xavier_uniform_(Tensor(self.number_relations,
                                                               self.ent_emb_dim)))
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
        projected_heads = self.recover_and_project(heads, normal_vectors)
        projected_tails = self.recover_and_project(tails, normal_vectors)
        projected_neg_heads = self.recover_and_project(negative_heads, normal_vectors)
        projected_neg_tails = self.recover_and_project(negative_tails, normal_vectors)

        # compute dissimilarities
        golden_triplets = self.dissimilarity(projected_heads + relations_embeddings,
                                             projected_tails)
        negative_triplets = self.dissimilarity(projected_neg_heads + relations_embeddings,
                                               projected_neg_tails)

        return golden_triplets, negative_triplets

    def recover_and_project(self, entities, normal_vectors):
        """Recover entity (either head or tail) embeddings and project on hyperplane defined by\
        provided normal vectors.

        Parameters
        ----------

        entities : torch tensor, dtype = long, shape = (batch_size)
            Integer keys of entities
        normal_vectors : torch tensor, dtype = float, shape = (batch_size, ent_emb_dim)
            Normal vectors relative to the current relations.

        Returns
        -------

        projections : torch tensor, dtype = float, shape = (batch_size, ent_emb_dim)
            Projection of the embedded entities on the hyperplanes defined by the provided normal\
            vectors.
        """
        # recover and normalize embeddings
        ent_emb = normalize(self.entity_embeddings(entities), p=self.norm_type, dim=1)

        # project entities into relation space
        normal_component = (ent_emb * normal_vectors).sum(dim=1).view((-1, 1))

        return ent_emb - normal_component * normal_vectors

    def normalize_parameters(self):
        """Normalize the embeddings of the entities using the model-specified norm and the normal \
        vectors using the L2 norm.
        """
        self.entity_embeddings.weight.data = normalize(self.entity_embeddings.weight.data,
                                                       p=self.norm_type, dim=1)
        self.normal_vectors = normalize(self.normal_vectors, p=2, dim=1)


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
        self.projection_matrices = Parameter(xavier_uniform_(Tensor(self.number_relations,
                                                                    self.rel_emb_dim,
                                                                    self.ent_emb_dim)))

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
        projected_heads = self.recover_and_project(heads, projection_matrices)
        projected_tails = self.recover_and_project(tails, projection_matrices)
        projected_neg_heads = self.recover_and_project(negative_heads, projection_matrices)
        projected_neg_tails = self.recover_and_project(negative_tails, projection_matrices)

        # compute dissimilarities
        golden_triplets = self.dissimilarity(projected_heads + relations_embeddings,
                                             projected_tails)
        negative_triplets = self.dissimilarity(projected_neg_heads + relations_embeddings,
                                               projected_neg_tails)
        return golden_triplets, negative_triplets

    def recover_and_project(self, entities, projection_matrices):
        """Recover entity (either head or tail) embeddings and project on hyperplane defined by\
        provided projection matrices.

        Parameters
        ----------

        entities : torch tensor, dtype = long, shape = (batch_size)
            Integer keys of entities
        projection_matrices : torch tensor, dtype = float, shape = (batch_size, rel_emb_dim, ent_emb_dim)
            Projection matrices for the current relations.

        Returns
        -------

        projections : torch tensor, dtype = float, shape = (batch_size, rel_emb_dim)
            Projection of the entities into relation-specific subspaces.
        """
        b_size = len(entities)
        # recover and normalize embeddings
        ent_emb = normalize(self.entity_embeddings(entities), p=2, dim=1)

        # project entities into relation space
        new_shape = (b_size, 1, self.ent_emb_dim)
        projection = matmul(ent_emb.view(new_shape), projection_matrices.transpose(1, 2))

        return projection.view(len(entities), self.rel_emb_dim)

    def normalize_parameters(self):
        """Normalize the parameters of the model using only L2 norm.
        """
        self.entity_embeddings.weight.data = normalize(self.entity_embeddings.weight.data,
                                                       p=2, dim=1)
        self.relation_embeddings = normalize(self.relation_embeddings, p=2, dim=1)
        self.projection_matrices = normalize(self.projection_matrices, p=2, dim=2)


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
        self.ent_proj_vects = Parameter(xavier_uniform_(Tensor(self.number_entities,
                                                               self.ent_emb_dim)))
        self.rel_proj_vects = Parameter(xavier_uniform_(Tensor(self.number_relations,
                                                               self.rel_emb_dim)))

        self.ent_proj_vects.data = normalize(self.ent_proj_vects.data, p=2, dim=1)
        self.rel_proj_vects.data = normalize(self.rel_proj_vects.data, p=2, dim=1)

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
        # recover relations projection vectors and relations embeddings
        rel_proj = normalize(self.rel_proj_vects[relations], p=2, dim=1)
        relations_embeddings = normalize(self.relation_embeddings(relations), p=2, dim=1)

        # project
        projected_heads = self.recover_and_project(heads, rel_proj)
        projected_tails = self.recover_and_project(tails, rel_proj)
        projected_neg_heads = self.recover_and_project(negative_heads, rel_proj)
        projected_neg_tails = self.recover_and_project(negative_tails, rel_proj)

        # compute dissimilarities
        golden_triplets = self.dissimilarity(projected_heads + relations_embeddings,
                                             projected_tails)
        negative_triplets = self.dissimilarity(projected_neg_heads + relations_embeddings,
                                               projected_neg_tails)

        return golden_triplets, negative_triplets

    def recover_and_project(self, entities, rel_proj):
        """Recover entity (either head or tail) embeddings and project on hyperplane defined by\
        provided normal vectors.

        Parameters
        ----------

        entities : torch tensor, dtype = long, shape = (batch_size)
            Integer keys of entities
        rel_proj : torch tensor, dtype = float, shape = (batch_size, rel_emb_dim)
            Projection vectors for the current relations.

        Returns
        -------

        projections : torch tensor, dtype = float, shape = (batch_size, rel_emb_dim)
            Projection of the entities into relation-specific subspaces.
        """
        b_size = len(entities)

        # recover and normalize embeddings
        ent_emb = normalize(self.entity_embeddings(entities), p=2, dim=1)

        # recover projection vectors
        ent_proj = normalize(self.ent_proj_vects[entities], p=2, dim=1)

        # project entities into relation space
        proj_mat = matmul(rel_proj.view((b_size, self.rel_emb_dim, 1)),
                          ent_proj.view((b_size, 1, self.ent_emb_dim)))
        if proj_mat.is_cuda:
            proj_mat += eye(n=self.rel_emb_dim, m=self.ent_emb_dim, device='cuda')
        else:
            proj_mat += eye(n=self.rel_emb_dim, m=self.ent_emb_dim)

        projection = matmul(ent_emb.view((b_size, 1, self.ent_emb_dim)), proj_mat.transpose(1, 2))

        return projection.view((b_size, self.rel_emb_dim))

    def normalize_parameters(self):
        """Normalize the parameters of the model using only L2 norm.
        """
        self.entity_embeddings = normalize(self.entity_embeddings, p=2, dim=1)
        self.relation_embeddings = normalize(self.relation_embeddings, p=2, dim=1)
        self.ent_proj_vects = normalize(self.ent_proj_vects, p=2, dim=1)
        self.rel_proj_vects = normalize(self.rel_proj_vects, p=2, dim=1)
