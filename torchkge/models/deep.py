# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""

from torch import nn, cat

from ..models.interfaces import Model
from ..utils import init_embedding


class ConvKBModel(Model):
    """Implementation of ConvKB model detailed in 2018 paper by Nguyen et al..
    This class inherits from the :class:`torchkge.models.interfaces.Model`
    interface. It then has its attributes as well.


    References
    ----------
    * Nguyen, D. Q., Nguyen, T. D., Nguyen, D. Q., and Phung, D.
      `A Novel Embed- ding Model for Knowledge Base Completion Based on
      Convolutional Neural Network.
      <https://arxiv.org/abs/1712.02121>`_
      In Proceedings of the 2018 Conference of the North American Chapter of
      the Association for Computational Linguistics: Human Language
      Technologies (2018), vol. 2, pp. 327–333.

    Parameters
    ----------
    emb_dim: int
        Dimension of embedding space.
    n_filters: int
        Number of filters used for convolution.
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

    def __init__(self, emb_dim, n_filters, n_entities, n_relations):
        super().__init__(n_entities, n_relations)
        self.emb_dim = emb_dim

        self.ent_emb = init_embedding(self.n_ent, self.emb_dim)
        self.rel_emb = init_embedding(self.n_rel, self.emb_dim)

        self.convlayer = nn.Sequential(nn.Conv1d(3, n_filters, 1, stride=1),
                                       nn.ReLU())
        self.output = nn.Sequential(nn.Linear(emb_dim * n_filters, 2),
                                    nn.Softmax(dim=1))

    def scoring_function(self, h_idx, t_idx, r_idx):
        """Compute the scoring function for the triplets given as argument:
        by applying convolutions to the concatenation of the embeddings. See
        referenced paper for more details on the score. See
        torchkge.models.interfaces.Models for more details on the API.

        """
        b_size = h_idx.shape[0]

        h = self.ent_emb(h_idx).view(b_size, 1, -1)
        t = self.ent_emb(t_idx).view(b_size, 1, -1)
        r = self.rel_emb(r_idx).view(b_size, 1, -1)
        concat = cat((h, r, t), dim=1)

        return self.output(self.convlayer(concat).reshape(b_size, -1))

    def normalize_parameters(self):
        """Normalize the entity embeddings, as explained in original paper.
        This methods should be called at the end of each training epoch and at
        the end of training as well.

        """
        raise NotImplementedError

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

    def lp_scoring_function(self, h, t, r):
        """Link prediction evaluation helper function. See
        torchkge.models.interfaces.Models for more details on the API.

        """
        b_size = h.shape[0]

        if len(h.shape) == 2:
            concat = cat((h.view(b_size, 1, self.emb_dim),
                          r.view(b_size, 1, self.emb_dim)),
                         dim=1)
            concat = concat.view(b_size, 1, 2, self.emb_dim)
            concat = concat.expand(b_size, self.n_ent, 2, self.emb_dim)
            concat = cat((concat, t), dim=2)
            # shape = (b_size, n_ent, 3, emb_dim)
            concat = concat.reshape(-1, 3, self.emb_dim)

        else:
            concat = cat((r.view(b_size, 1, self.emb_dim),
                          t.view(b_size, 1, self.emb_dim)),
                         dim=1)
            concat = concat.view(b_size, 1, 2, self.emb_dim)
            concat = concat.expand(b_size, self.n_ent, 2, self.emb_dim)
            concat = cat((h, concat), dim=2)
            # shape = (b_size, n_entities, 3, emb_dim)
            concat = concat.reshape(-1, 3, self.emb_dim)

        scores = self.output(self.convlayer(concat).reshape(concat.shape[0],
                                                            -1))
        scores = scores.reshape(b_size, -1, 2)

        return scores[:, :, 1]

    def lp_prep_cands(self, h_idx, t_idx, r_idx):
        """Link prediction evaluation helper function. Get entities embeddings
        and relations embeddings. The output will be fed to the
        `lp_scoring_function` method. See torchkge.models.interfaces.Models for
        more details on the API.

        """
        b_size = h_idx.shape[0]

        h = self.ent_emb(h_idx)
        t = self.ent_emb(t_idx)
        r = self.rel_emb(r_idx)

        candidates = self.ent_emb.weight.data.view(1, self.n_ent, self.emb_dim)
        candidates = candidates.expand(b_size, self.n_ent, self.emb_dim)

        return h, t, candidates.view(b_size, self.n_ent, 1, self.emb_dim), r


class KBGANModel(Model):

    def __init__(self, generator: torchkge.models.Model, discriminator: torchkge.models.Model, sampling_size,
                 kg_train: torchkge.data_structures.KnowledgeGraph):
        assert (kg_train.n_ent == generator.n_ent) & (kg_train.n_rel == generator.n_rel)
        assert (generator.n_ent == discriminator.n_ent) & (generator.n_rel == discriminator.n_rel)

        super().__init__(generator.n_ent, generator.n_rel)
        self.generator = generator
        self.discriminator = discriminator
        self.sampling_size = sampling_size
        self.sampler = BernoulliNegativeSampler(kg_train, kg_val=None, kg_test=None, n_neg=sampling_size)

    def scoring_function(self, h_idx, t_idx, r_idx):
        self.discriminator.eval()
        return self.discriminator.scoring_function(h_idx, t_idx, r_idx)

    def forward(self, heads, tails, relations):
        assert self.training

        n_heads, n_tails = self.sampler.corrupt_batch(heads, tails, relations)
        n_heads, n_tails = n_heads.view(-1), n_tails.view(-1)
        dist = Categorical(logits=self.generator.scoring_function(n_heads,
                                                                  n_tails,
                                                                  relations).view(-1, self.sampling_size))
        choice = dist.sample()

        discriminator_loss = self.discriminator.forward(heads, tails,
                                                        n_heads.gather(1, choice.view(-1, 1)),
                                                        n_tails.gather(1, choice.view(-1, 1)), relations)


        self.generator.scoring_function()

    def generator_step(self, h_idx, t_idx, r_idx, n_sample=1, temperature=1.0, train=True):
        if not hasattr(self, 'opt'):
            self.opt = Adam(self.mdl.parameters(), weight_decay=self.weight_decay)
        n, m = t_idx.size()
        rel_var = Variable(r_idx.cuda())
        src_var = Variable(h_idx.cuda())
        dst_var = Variable(t_idx.cuda())

        logits = self.mdl.prob_logit(src_var, rel_var, dst_var) / temperature
        probs = nnf.softmax(logits)
        row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, n_sample)
        sample_idx = torch.multinomial(probs, n_sample, replacement=True)
        sample_srcs = h_idx[row_idx, sample_idx.data.cpu()]
        sample_dsts = t_idx[row_idx, sample_idx.data.cpu()]
        rewards = yield sample_srcs, sample_dsts
        if train:
            self.mdl.zero_grad()
            log_probs = nnf.log_softmax(logits)
            reinforce_loss = -torch.sum(Variable(rewards) * log_probs[row_idx.cuda(), sample_idx.data])
            reinforce_loss.backward()
            self.opt.step()
            self.mdl.constraint()
        yield None

    def discriminator_step(self, src, rel, dst, src_fake, dst_fake, train=True):
        if not hasattr(self, 'opt'):
            self.opt = Adam(self.mdl.parameters(), weight_decay=self.weight_decay)
        src_var = Variable(src.cuda())
        rel_var = Variable(rel.cuda())
        dst_var = Variable(dst.cuda())
        src_fake_var = Variable(src_fake.cuda())
        dst_fake_var = Variable(dst_fake.cuda())
        losses = self.mdl.pair_loss(src_var, rel_var, dst_var, src_fake_var, dst_fake_var)
        fake_scores = self.mdl.score(src_fake_var, rel_var, dst_fake_var)
        if train:
            self.mdl.zero_grad()
            torch.sum(losses).backward()
            self.opt.step()
            self.mdl.constraint()
        return losses.data, -fake_scores.data