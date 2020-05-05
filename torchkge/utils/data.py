# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
"""

import shutil

from os import environ, makedirs
from os.path import exists, expanduser, join


def get_data_home(data_home=None):
    if data_home is None:
        data_home = environ.get('TORCHKGE_DATA',
                                join('~', 'torchkge_data'))
    data_home = expanduser(data_home)
    if not exists(data_home):
        makedirs(data_home)
    return data_home


def clear_data_home(data_home=None):
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)


def get_n_batches(n, b_size):
    n_batch = n // b_size
    if n % b_size > 0:
        n_batch += 1
    return n_batch


class DataLoader:
    """This class is inspired from :class:`torch.utils.dataloader.DataLoader`.
    It is however way simpler.

    """
    def __init__(self, kg, batch_size, use_cuda=None):
        """

        Parameters
        ----------
        kg: torchkge.data_structures.KnowledgeGraph
            Knowledge graph in the form of an object implemented in
            torchkge.data_structures.
        batch_size: int
            Size of the required batches.
        use_cuda: str (opt, default = None)
            Can be either None (no use of cuda at all), 'all' to move all the
            dataset to cuda and then split in batches or 'batch' to simply move
            the batches to cuda before they are returned.
        """
        self.h = kg.head_idx
        self.t = kg.tail_idx
        self.r = kg.relations

        self.use_cuda = use_cuda
        self.batch_size = batch_size

        if use_cuda is not None and use_cuda == 'all':
            self.h = self.h.cuda()
            self.t = self.t.cuda()
            self.r = self.r.cuda()

    def __len__(self):
        return get_n_batches(len(self.h), self.batch_size)

    def __iter__(self):
        return _DataLoaderIter(self)


class _DataLoaderIter:
    def __init__(self, loader):
        self.h = loader.h
        self.t = loader.t
        self.r = loader.r

        self.use_cuda = loader.use_cuda
        self.batch_size = loader.batch_size

        self.n_batches = get_n_batches(len(self.h), self.batch_size)
        self.current_batch = 0

    def __next__(self):
        if self.current_batch == self.n_batches:
            raise StopIteration
        else:
            i = self.current_batch
            self.current_batch += 1

            tmp_h = self.h[i * self.batch_size: (i + 1) * self.batch_size]
            tmp_t = self.t[i * self.batch_size: (i + 1) * self.batch_size]
            tmp_r = self.r[i * self.batch_size: (i + 1) * self.batch_size]

            if self.use_cuda is not None and self.use_cuda == 'batch':
                return tmp_h.cuda(), tmp_t.cuda(), tmp_r.cuda()
            else:
                return tmp_h, tmp_t, tmp_r

    def __iter__(self):
        return self
