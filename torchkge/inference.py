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
    def __init__(self, model, entities1, entities2, topk=1, dictionary=None):

        self.model = model
        self.entities1 = entities1
        self.entities2 = entities2
        self.topk = topk
        self.dictionary = dictionary

        self.predictions = empty(size=(len(entities1), topk)).long()
        self.scores = empty(size=(len(entities2), topk))

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
            h_emb, t_emb, _, candidates = self.model.inference_prepare_candidates(ents1, ents2, tensor([]).long(), entities=False)
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
    def __init__(self, model, known_entities, known_relations, topk=1, missing='tails', dictionary=None):
        try:
            assert missing in ['heads', 'tails']
            self.missing = missing
        except AssertionError:
            raise WrongArgumentsError("missing entity should either be 'heads' or 'tails'")
        self.model = model
        self.known_entities = known_entities
        self.known_relations = known_relations
        self.missing = missing
        self.topk = topk
        self.dictionary = dictionary

        self.predictions = empty(size=(len(known_entities), topk)).long()
        self.scores = empty(size=(len(known_entities), topk))

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
                _, t_emb, rel_emb, candidates = self.model.inference_prepare_candidates(tensor([]).long(), known_ents, known_rels,
                                                                                        entities=True)
                scores = self.model.inference_scoring_function(candidates, t_emb, rel_emb)
            else:
                h_emb, _, rel_emb, candidates = self.model.inference_prepare_candidates(known_ents, tensor([]).long(), known_rels,
                                                                                        entities=True)
                scores = self.model.inference_scoring_function(h_emb, candidates, rel_emb)

            if self.dictionary is not None:
                scores = filter_scores(scores, self.dictionary, known_ents, known_rels, None)

            scores, indices = scores.sort(descending=True)

            self.predictions[i * b_size: (i+1)*b_size] = indices[:, :self.topk]
            self.scores[i*b_size, (i+1)*b_size] = scores[:, :self.topk]

        if use_cuda:
            self.predictions = self.predictions.cpu()
            self.scores = self.scores.cpu()
