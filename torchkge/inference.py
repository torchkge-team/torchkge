# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""
from torch import empty, tensor
from tqdm.autonotebook import tqdm

from .exceptions import WrongArgumentsError
from .utils import filter_scores
from .utils.data import get_n_batches


class DataLoader_:
    """This class is inspired from :class:`torch.utils.dataloader.DataLoader`.
    It is however way simpler.

    """
    def __init__(self, a, b, batch_size, use_cuda=None):
        """

        Parameters
        ----------
        batch_size: int
            Size of the required batches.
        use_cuda: str (opt, default = None)
            Can be either None (no use of cuda at all), 'all' to move all the
            dataset to cuda and then split in batches or 'batch' to simply move
            the batches to cuda before they are returned.
        """
        self.a = a
        self.b = b

        self.use_cuda = use_cuda
        self.batch_size = batch_size

        if use_cuda is not None and use_cuda == 'all':
            self.a = self.a.cuda()
            self.b = self.b.cuda()

    def __len__(self):
        return get_n_batches(len(self.a), self.batch_size)

    def __iter__(self):
        return _DataLoaderIter(self)


class _DataLoaderIter:
    def __init__(self, loader):
        self.a = loader.a
        self.b = loader.b

        self.use_cuda = loader.use_cuda
        self.batch_size = loader.batch_size

        self.n_batches = get_n_batches(len(self.a), self.batch_size)
        self.current_batch = 0

    def __next__(self):
        if self.current_batch == self.n_batches:
            raise StopIteration
        else:
            i = self.current_batch
            self.current_batch += 1

            tmp_a = self.a[i * self.batch_size: (i + 1) * self.batch_size]
            tmp_b = self.b[i * self.batch_size: (i + 1) * self.batch_size]

            if self.use_cuda is not None and self.use_cuda == 'batch':
                return tmp_a.cuda(), tmp_b.cuda()
            else:
                return tmp_a, tmp_b

    def __iter__(self):
        return self


class RelationInference(object):
    """Use trained embedding model to infer missing relations in triples.

    Parameters
    ----------
    model: torchkge.models.interfaces.Model
        Embedding model inheriting from the right interface.
    entities1: `torch.Tensor`, shape: (n_facts), dtype: `torch.long`
        List of the indices of known entities 1.
    entities2: `torch.Tensor`, shape: (n_facts), dtype: `torch.long`
        List of the indices of known entities 2.
    top_k: int
        Indicates the number of top predictions to return.
    dictionary: dict, optional (default=None)
        Dictionary of possible relations. It is used to filter predictions
        that are known to be True in the training set in order to return
        only new facts.

    Attributes
    ----------
    model: torchkge.models.interfaces.Model
        Embedding model inheriting from the right interface.
    entities1: `torch.Tensor`, shape: (n_facts), dtype: `torch.long`
        List of the indices of known entities 1.
    entities2: `torch.Tensor`, shape: (n_facts), dtype: `torch.long`
        List of the indices of known entities 2.
    top_k: int
        Indicates the number of top predictions to return.
    dictionary: dict, optional (default=None)
        Dictionary of possible relations. It is used to filter predictions
        that are known to be True in the training set in order to return
        only new facts.
    predictions: `torch.Tensor`, shape: (n_facts, self.top_k), dtype: `torch.long`
        List of the indices of predicted relations for each test fact.
    scores: `torch.Tensor`, shape: (n_facts, self.top_k), dtype: `torch.float`
        List of the scores of resulting triples for each test fact.
    """
    # TODO: add the possibility to infer link orientation as well.

    def __init__(self, model, entities1, entities2, top_k=1, dictionary=None):

        self.model = model
        self.entities1 = entities1
        self.entities2 = entities2
        self.topk = top_k
        self.dictionary = dictionary

        self.predictions = empty(size=(len(entities1), top_k)).long()
        self.scores = empty(size=(len(entities2), top_k))

    def evaluate(self, b_size, verbose=True):
        use_cuda = next(self.model.parameters()).is_cuda

        if use_cuda:
            dataloader = DataLoader_(self.entities1, self.entities2, batch_size=b_size, use_cuda='batch')
            self.predictions = self.predictions.cuda()
        else:
            dataloader = DataLoader_(self.entities1, self.entities2, batch_size=b_size)

        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                             unit='batch', disable=(not verbose),
                             desc='Inference'):
            ents1, ents2 = batch[0], batch[1]
            h_emb, t_emb, _, candidates = self.model.inference_prepare_candidates(ents1, ents2, tensor([]).long(),
                                                                                  entities=False)
            scores = self.model.inference_scoring_function(h_emb, t_emb, candidates)

            if self.dictionary is not None:
                scores = filter_scores(scores, self.dictionary, ents1, ents2, None)

            scores, indices = scores.sort(descending=True)

            self.predictions[i * b_size: (i + 1) * b_size] = indices[:, :self.topk]
            self.scores[i * b_size, (i + 1) * b_size] = scores[:, :self.topk]

        if use_cuda:
            self.predictions = self.predictions.cpu()
            self.scores = self.scores.cpu()


class EntityInference(object):
    """Use trained embedding model to infer missing entities in triples.

        Parameters
        ----------
        model: torchkge.models.interfaces.Model
            Embedding model inheriting from the right interface.
        known_entities: `torch.Tensor`, shape: (n_facts), dtype: `torch.long`
            List of the indices of known entities.
        known_relations: `torch.Tensor`, shape: (n_facts), dtype: `torch.long`
            List of the indices of known relations.
        top_k: int
            Indicates the number of top predictions to return.
        missing: str
            String indicating if the missing entities are the heads or the tails.
        dictionary: dict, optional (default=None)
            Dictionary of possible heads or tails (depending on the value of `missing`).
            It is used to filter predictions that are known to be True in the training set
            in order to return only new facts.

        Attributes
        ----------
        model: torchkge.models.interfaces.Model
            Embedding model inheriting from the right interface.
        known_entities: `torch.Tensor`, shape: (n_facts), dtype: `torch.long`
            List of the indices of known entities.
        known_relations: `torch.Tensor`, shape: (n_facts), dtype: `torch.long`
            List of the indices of known relations.
        top_k: int
            Indicates the number of top predictions to return.
        missing: str
            String indicating if the missing entities are the heads or the tails.
        dictionary: dict, optional (default=None)
            Dictionary of possible heads or tails (depending on the value of `missing`).
            It is used to filter predictions that are known to be True in the training set
            in order to return only new facts.
        predictions: `torch.Tensor`, shape: (n_facts, self.top_k), dtype: `torch.long`
            List of the indices of predicted entities for each test fact.
        scores: `torch.Tensor`, shape: (n_facts, self.top_k), dtype: `torch.float`
            List of the scores of resulting triples for each test fact.

    """
    def __init__(self, model, known_entities, known_relations, top_k=1, missing='tails', dictionary=None):
        try:
            assert missing in ['heads', 'tails']
            self.missing = missing
        except AssertionError:
            raise WrongArgumentsError("missing entity should either be 'heads' or 'tails'")
        self.model = model
        self.known_entities = known_entities
        self.known_relations = known_relations
        self.missing = missing
        self.top_k = top_k
        self.dictionary = dictionary

        self.predictions = empty(size=(len(known_entities), top_k)).long()
        self.scores = empty(size=(len(known_entities), top_k))

    def evaluate(self, b_size, verbose=True):
        use_cuda = next(self.model.parameters()).is_cuda

        if use_cuda:
            dataloader = DataLoader_(self.known_entities, self.known_relations, batch_size=b_size, use_cuda='batch')
            self.predictions = self.predictions.cuda()
        else:
            dataloader = DataLoader_(self.known_entities, self.known_relations, batch_size=b_size)

        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                             unit='batch', disable=(not verbose),
                             desc='Inference'):
            known_ents, known_rels = batch[0], batch[1]
            if self.missing == 'heads':
                _, t_emb, rel_emb, candidates = self.model.inference_prepare_candidates(tensor([]).long(), known_ents,
                                                                                        known_rels,
                                                                                        entities=True)
                scores = self.model.inference_scoring_function(candidates, t_emb, rel_emb)
            else:
                h_emb, _, rel_emb, candidates = self.model.inference_prepare_candidates(known_ents, tensor([]).long(),
                                                                                        known_rels,
                                                                                        entities=True)
                scores = self.model.inference_scoring_function(h_emb, candidates, rel_emb)

            if self.dictionary is not None:
                scores = filter_scores(scores, self.dictionary, known_ents, known_rels, None)

            scores, indices = scores.sort(descending=True)

            self.predictions[i * b_size: (i+1)*b_size] = indices[:, :self.top_k]
            self.scores[i*b_size, (i+1)*b_size] = scores[:, :self.top_k]

        if use_cuda:
            self.predictions = self.predictions.cpu()
            self.scores = self.scores.cpu()
