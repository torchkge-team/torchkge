# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""
from typing import Optional

from ..data_structures import SmallKG
from ..sampling import BernoulliNegativeSampler, UniformNegativeSampler
from ..utils.data import get_n_batches

from tqdm.autonotebook import tqdm


class TrainDataLoader:
    """Dataloader providing the training process with batches of true and
    negatively sampled facts.

    Parameters
    ----------
    kg: torchkge.data_structures.KnowledgeGraph
        Dataset to be divided in batches.
    batch_size: int
        Size of the batches.
    sampling_type: str
        Either 'unif' (uniform negative sampling) or 'bern' (Bernoulli negative
        sampling).
    use_cuda: str (opt, default = None)
        Can be either None (no use of cuda at all), 'all' to move all the
        dataset to cuda and then split in batches or 'batch' to simply move
        the batches to cuda before they are returned.

    """

    def __init__(self, kg, batch_size, sampling_type, use_cuda=None):
        self.h = kg.head_idx
        self.t = kg.tail_idx
        self.r = kg.relations

        self.use_cuda = use_cuda
        self.b_size = batch_size
        self.iterator = None

        if sampling_type == 'unif':
            self.sampler = UniformNegativeSampler(kg)
        elif sampling_type == 'bern':
            self.sampler = BernoulliNegativeSampler(kg)

        self.tmp_cuda = use_cuda in ['batch', 'all']

        if use_cuda is not None and use_cuda == 'all':
            self.h = self.h.cuda()
            self.t = self.t.cuda()
            self.r = self.r.cuda()

    def __len__(self):
        return get_n_batches(len(self.h), self.b_size)

    def __iter__(self):
        self.iterator = TrainDataLoaderIter(self)
        return self.iterator

    def get_counter_examples(self) -> SmallKG:
        return SmallKG(self.iterator.nh, self.iterator.nt, self.iterator.r)


class TrainDataLoaderIter:
    def __init__(self, loader):
        self.h = loader.h
        self.t = loader.t
        self.r = loader.r

        self.nh, self.nt = loader.sampler.corrupt_kg(loader.b_size,
                                                     loader.tmp_cuda)
        if loader.use_cuda:
            self.nh = self.nh.cuda()
            self.nt = self.nt.cuda()

        self.use_cuda = loader.use_cuda
        self.b_size = loader.b_size

        self.n_batches = get_n_batches(len(self.h), self.b_size)
        self.current_batch = 0

    def __next__(self):
        if self.current_batch == self.n_batches:
            raise StopIteration
        else:
            i = self.current_batch
            self.current_batch += 1

            batch = dict()
            batch['h'] = self.h[i * self.b_size: (i + 1) * self.b_size]
            batch['t'] = self.t[i * self.b_size: (i + 1) * self.b_size]
            batch['r'] = self.r[i * self.b_size: (i + 1) * self.b_size]
            batch['nh'] = self.nh[i * self.b_size: (i + 1) * self.b_size]
            batch['nt'] = self.nt[i * self.b_size: (i + 1) * self.b_size]

            if self.use_cuda == 'batch':
                batch['h'] = batch['h'].cuda()
                batch['t'] = batch['t'].cuda()
                batch['r'] = batch['r'].cuda()
                batch['nh'] = batch['nh'].cuda()
                batch['nt'] = batch['nt'].cuda()

            return batch

    def __iter__(self):
        return self


class Trainer:
    """This class simply wraps a simple training procedure.

    Parameters
    ----------
    model: torchkge.models.interfaces.Model
        Model to be trained.
    criterion:
        Criteria which should differentiate positive and negative scores. Can
        be an elements of torchkge.utils.losses
    kg_train: torchkge.data_structures.KnowledgeGraph
        KG used for training.
    n_epochs: int
        Number of epochs in the training procedure.
    batch_size: int
        Number of batches to use.
    sampling_type: str
        Either 'unif' (uniform negative sampling) or 'bern' (Bernoulli negative
        sampling).
    use_cuda: str (opt, default = None)
        Can be either None (no use of cuda at all), 'all' to move all the
        dataset to cuda and then split in batches or 'batch' to simply move
        the batches to cuda before they are returned.


    Attributes
    ----------

    """
    def __init__(self, model, criterion, kg_train, n_epochs, batch_size,
                 optimizer, sampling_type='bern', use_cuda=None):

        self.model = model
        self.criterion = criterion
        self.kg_train = kg_train
        self.use_cuda = use_cuda
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.sampling_type = sampling_type

        self.batch_size = batch_size
        self.n_triples = len(kg_train)
        self.counter_examples: Optional[SmallKG] = None

    def process_batch(self, current_batch):
        self.optimizer.zero_grad()

        h, t, r = current_batch['h'], current_batch['t'], current_batch['r']
        nh, nt = current_batch['nh'], current_batch['nt']

        p, n = self.model(h, t, r, nh, nt)
        loss = self.criterion(p, n)
        loss.backward()
        self.optimizer.step()

        return loss.detach().item()

    def run(self):
        if self.use_cuda in ['all', 'batch']:
            self.model.cuda()
            self.criterion.cuda()

        iterator = tqdm(range(self.n_epochs), unit='epoch')
        data_loader = TrainDataLoader(self.kg_train,
                                      batch_size=self.batch_size,
                                      sampling_type=self.sampling_type,
                                      use_cuda=self.use_cuda)
        self.counter_examples = data_loader.get_counter_examples()
        for epoch in iterator:
            sum_ = 0
            for i, batch in enumerate(data_loader):
                loss = self.process_batch(batch)
                sum_ += loss

            iterator.set_description(
                'Epoch {} | mean loss: {:.5f}'.format(epoch + 1, sum_ / len(data_loader)))
            self.model.normalize_parameters()

    def get_counter_examples(self) -> Optional[SmallKG]:
        """
        Retrieve the counter-examples generated while training the model.

        If the model has not been trained yet, return None

        Returns
        -------
        A simple knowledge graph containing the triplets that were used as counter-examples during the training phase.
        """
        return self.counter_examples
