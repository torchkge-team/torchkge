from torch import nn, cat

from .interfaces import Model
from torchkge.utils import init_embedding, get_true_targets, get_rank


class ConvKBModel(Model):
    """Implementation of ConvKB model detailed in 2018 paper by Nguyen et al.. This class inherits from the
    :class:`torchkge.models.interfaces.Model` interface. It then has its attributes as well.


    References
    ----------
    * Nguyen, D. Q., Nguyen, T. D., Nguyen, D. Q., and Phung, D.
      `A Novel Embed- ding Model for Knowledge Base Completion Based on Convolutional Neural Network.
      <https://arxiv.org/abs/1712.02121>`_
      In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (2018), vol. 2, pp. 327–333.

    Parameters
    ----------
    emb_dim: int
        Dimension of the embedding of entities and relations.
    n_filters: int
        Number of filters used for convolution.
    n_entities: int
        Number of entities in the current data set.
    n_relations: int
        Number of relations in the current data set.

    Attributes
    ----------
    relation_embeddings: torch Embedding, shape: (number_relations, ent_emb_dim)
        Contains the embeddings of the relations. It is initialized with Xavier uniform and then\
        normalized.
    entity_embeddings: torch Embedding, shape: (number_relations, ent_emb_dim)
        Contains the embeddings of the entities. It is initialized with Xavier uniform and then\
        normalized.

    """

    def __init__(self, emb_dim, n_filters, n_entities, n_relations):
        super().__init__(emb_dim, n_entities, n_relations)

        self.entity_embeddings = init_embedding(n_entities, emb_dim)
        self.relation_embeddings = init_embedding(n_relations, emb_dim)

        self.convlayer = nn.Sequential(nn.Conv1d(3, n_filters, 1, stride=1), nn.ReLU())
        self.output = nn.Sequential(nn.Linear(emb_dim * n_filters, 2), nn.Softmax(dim=1))

    def scoring_function(self, heads_idx, tails_idx, rels_idx):

        """Compute the scoring function for the triplets given as argument.

        Parameters
        ----------
        heads_idx: `torch.Tensor`, dtype: `torch.long`, shape: (batch_size)
            Integer keys of the current batch's heads
        tails_idx: `torch.Tensor`, dtype: `torch.long`, shape: (batch_size)
            Integer keys of the current batch's tails.
        rels_idx: `torch.Tensor`, dtype: `torch.long`, shape: (batch_size)
            Integer keys of the current batch's relations.

        Returns
        -------
        score: `torch.Tensor`, dtype: `torch.float`, shape: (batch_size)
            Score function computed after convolutions.

        """
        b_size = len(heads_idx)
        h = self.entity_embeddings(heads_idx).view(b_size, 1, -1)
        t = self.entity_embeddings(tails_idx).view(b_size, 1, -1)
        r = self.relation_embeddings(rels_idx).view(b_size, 1, -1)
        concat = cat((h, r, t), dim=1)
        return self.output(self.convlayer(concat).reshape(b_size, -1))

    def normalize_parameters(self):
        raise NotImplementedError

    def evaluation_helper(self, h_idx, t_idx, r_idx):
        """Prepares current entities, relations and candidates into relation-specific sub-spaces.

        Parameters
        ----------
        h_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Tensor containing indices of current head entities.
        t_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Tensor containing indices of current tail entities.
        r_idx: `torch.Tensor`, shape: (b_size), dtype: `torch.long`
            Tensor containing indices of current relations.

        Returns
        -------
        h: `torch.Tensor`, shape: (b_size, rel_emb_dim), dtype: `torch.float`
            Tensor containing embeddings of current head entities.
        t: `torch.Tensor`, shape: (b_size, rel_emb_dim), dtype: `torch.float`
            Tensor containing embeddings of current tail entities.
        candidates: `torch.Tensor`, shape: (b_size, n_entities, 1, ent_emb_dim), dtype: `torch.float`
            Tensor containing all entities as candidates.
        r: `torch.Tensor`, shape: (b_size, rel_emb_dim), dtype: `torch.float`
            Tensor containing current relations embeddings.

        """
        b_size = len(h_idx)
        h = self.entity_embeddings(h_idx)
        t = self.entity_embeddings(t_idx)
        r = self.relation_embeddings(r_idx)
        candidates = self.entity_embeddings.weight.clone().view(1,
                                                                self.number_entities,
                                                                self.ent_emb_dim).expand(b_size,
                                                                                self.number_entities,
                                                                                self.ent_emb_dim)
        return h, t, candidates.view(b_size, self.number_entities, 1, self.ent_emb_dim), r

    def compute_ranks(self, e_emb, candidates, r_emb, e_idx, r_idx, true_idx, dictionary, heads=1):
        """Compute the ranks and the filtered ranks of true entities when doing link prediction. Note that the \
        best rank possible is 1.

        Parameters
        ----------
        e_emb: `torch.Tensor`, shape: (batch_size, rel_emb_dim), dtype: `torch.float`
            Tensor containing current embeddings of entities.
        candidates: `torch.Tensor`, shape: (b_size, rel_emb_dim, n_entities), dtype: `torch.float`
            Tensor containing embeddings of all entities.
        r_emb: `torch.Tensor`, shape: (batch_size, ent_emb_dim), dtype: `torch.float`
            Tensor containing current embeddings of relations.
        e_idx: `torch.Tensor`, shape: (batch_size), dtype: `torch.long`
            Tensor containing the indices of entities.
        r_idx: `torch.Tensor`, shape: (batch_size), dtype: `torch.long`
            Tensor containing the indices of relations.
        true_idx: `torch.Tensor`, shape: (batch_size), dtype: `torch.long`
            Tensor containing the true entity for each sample.
        dictionary: default dict
            Dictionary of keys (int, int) and values list of ints giving all possible entities for
            the (entity, relation) pair.
        heads: integer
            1 ou -1 (must be 1 if entities are heads and -1 if entities are tails). \
            We test dissimilarity_type between heads * entities + relations and heads * targets.


        Returns
        -------
        rank_true_entities: `torch.Tensor`, shape: (b_size), dtype: `torch.int`
            Tensor containing the rank of the true entities when ranking any entity based on \
            computation of d(hear+relation, tail).
        filtered_rank_true_entities: `torch.Tensor`, shape: (b_size), dtype: `torch.int`
            Tensor containing the rank of the true entities when ranking only true false entities \
            based on computation of d(hear+relation, tail).

        """
        current_batch_size, embedding_dimension = e_emb.shape

        if heads == 1:
            concat = cat((e_emb.view(current_batch_size, 1, self.ent_emb_dim),
                          r_emb.view(current_batch_size, 1, self.ent_emb_dim)),
                         dim=1)
            concat = concat.view(current_batch_size, 1, 2, self.ent_emb_dim)
            concat = concat.expand(current_batch_size, self.number_entities, 2, self.ent_emb_dim)
            concat = cat((concat, candidates), dim=2)  # shape = (b_size, n_entities, 3, emb_dim)
            concat = concat.reshape(-1, 3, self.ent_emb_dim)

        else:
            concat = cat((r_emb.view(current_batch_size, 1, self.ent_emb_dim),
                          e_emb.view(current_batch_size, 1, self.ent_emb_dim)),
                         dim=1)
            concat = concat.view(current_batch_size, 1, 2, self.ent_emb_dim)
            concat = concat.expand(current_batch_size, self.number_entities, 2, self.ent_emb_dim)
            concat = cat((candidates, concat), dim=2)  # shape = (b_size, n_entities, 3, emb_dim)
            concat = concat.reshape(-1, 3, self.ent_emb_dim)

        scores = self.output(self.convlayer(concat).reshape(len(concat), -1)).reshape(current_batch_size, -1, 2)
        scores = scores[:, :, 1]

        filt_scores = scores.clone()

        for i in range(current_batch_size):
            true_targets = get_true_targets(dictionary, e_idx, r_idx, true_idx, i)
            if true_targets is None:
                continue
            filt_scores[i][true_targets] = - float('Inf')

        rank_true_entities = get_rank(scores, true_idx)
        filtered_rank_true_entities = get_rank(filt_scores, true_idx)

        return rank_true_entities, filtered_rank_true_entities
