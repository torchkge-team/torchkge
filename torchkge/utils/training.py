# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""
from ..utils.data_utils import get_n_batches
from ..sampling import UniformNegativeSampler, BernoulliNegativeSampler

from tqdm.autonotebook import tqdm


class TrainDataLoader:
    def __init__(self, kg, batch_size, sampling_type, use_cuda=None):
        self.h = kg.head_idx
        self.t = kg.tail_idx
        self.r = kg.relations

        self.use_cuda = use_cuda
        self.b_size = batch_size

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
        return TrainDataLoaderIter(self)


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
    def __init__(self, model, criterion, kg_train, use_gpu, lr,
                 n_triples, n_epochs, n_batches, optimizer,
                 sampling_type='bern'):
        self.model = model
        self.criterion = criterion
        self.kg_train = kg_train
        self.use_gpu = use_gpu
        self.lr = lr
        self.n_triples = n_triples
        self.n_epochs = n_epochs
        self.batch_size = int(len(kg_train) / n_batches) + 1
        self.optimizer = optimizer
        self.sampling_type = sampling_type

    def process_batch(self, current_batch):
        self.optimizer.zero_grad()

        h, t, r = current_batch['h'], current_batch['t'], current_batch['r']
        nh, nt = current_batch['nh'], current_batch['nt']

        p, n = self.model(h, t, nh, nt, r)
        loss = self.criterion(p, n)
        loss.backward()
        self.optimizer.step()

        return loss.detach().item()

    def run(self):
        if self.use_gpu:
            self.model.cuda()
            self.criterion.cuda()

        iterator = tqdm(range(self.n_epochs), unit='epoch')
        data_loader = TrainDataLoader(self.kg_train,
                                      batch_size=self.batch_size,
                                      sampling_type=self.sampling_type,
                                      use_cuda='all')
        for epoch in iterator:
            sum_ = 0
            for i, batch in enumerate(data_loader):
                loss = self.process_batch(batch)
                sum_ += loss

            iterator.set_description(
                'Epoch {} | mean loss: {:.5f}'.format(epoch + 1, sum_ / len(data_loader)))
            self.model.normalize_parameters()
