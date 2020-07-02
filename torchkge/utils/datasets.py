# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>

This module's code is freely adapted from Scikit-Learn's
sklearn.datasets.base.py code.
"""

import shutil
import tarfile
import zipfile

from os import makedirs, remove
from os.path import exists
from pandas import concat, DataFrame, merge, read_csv
from urllib.request import urlretrieve

from torchkge.data_structures import KnowledgeGraph

from torchkge.utils import get_data_home


def load_fb13(data_home=None):
    """Load FB13 dataset.

    Parameters
    ----------
    data_home: str, optional
        Path to the `torchkge_data` directory (containing data folders). If
        files are not present on disk in this directory, they are downloaded
        and then placed in the right place.

    Returns
    -------
    kg_train: torchkge.data_structures.KnowledgeGraph
    kg_val: torchkge.data_structures.KnowledgeGraph
    kg_test: torchkge.data_structures.KnowledgeGraph

    """
    if data_home is None:
        data_home = get_data_home()
    data_path = data_home + '/FB13'
    if not exists(data_path):
        makedirs(data_path, exist_ok=True)
        urlretrieve("https://graphs.telecom-paristech.fr/data/torchkge/kgs/FB13.zip",
                    data_home + '/FB13.zip')
        with zipfile.ZipFile(data_home + '/FB13.zip', 'r') as zip_ref:
            zip_ref.extractall(data_home)
        remove(data_home + '/FB13.zip')
        shutil.rmtree(data_home + '/__MACOSX')

    df1 = read_csv(data_path + '/train2id.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df2 = read_csv(data_path + '/valid2id.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df3 = read_csv(data_path + '/test2id.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df = concat([df1, df2, df3])
    kg = KnowledgeGraph(df)

    return kg.split_kg(sizes=(len(df1), len(df2), len(df3)))


def load_fb15k(data_home=None):
    """Load FB15k dataset. See `here
    <https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data>`__
    for paper by Bordes et al. originally presenting the dataset.

    Parameters
    ----------
    data_home: str, optional
        Path to the `torchkge_data` directory (containing data folders). If
        files are not present on disk in this directory, they are downloaded
        and then placed in the right place.

    Returns
    -------
    kg_train: torchkge.data_structures.KnowledgeGraph
    kg_val: torchkge.data_structures.KnowledgeGraph
    kg_test: torchkge.data_structures.KnowledgeGraph

    """
    if data_home is None:
        data_home = get_data_home()
    data_path = data_home + '/FB15k'
    if not exists(data_path):
        makedirs(data_path, exist_ok=True)
        urlretrieve("https://graphs.telecom-paristech.fr/data/torchkge/kgs/FB15k.zip",
                    data_home + '/FB15k.zip')
        with zipfile.ZipFile(data_home + '/FB15k.zip', 'r') as zip_ref:
            zip_ref.extractall(data_home)
        remove(data_home + '/FB15k.zip')
        shutil.rmtree(data_home + '/__MACOSX')

    df1 = read_csv(data_path + '/freebase_mtr100_mte100-train.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df2 = read_csv(data_path + '/freebase_mtr100_mte100-valid.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df3 = read_csv(data_path + '/freebase_mtr100_mte100-test.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df = concat([df1, df2, df3])
    kg = KnowledgeGraph(df)

    return kg.split_kg(sizes=(len(df1), len(df2), len(df3)))


def load_fb15k237(data_home=None):
    """Load FB15k237 dataset. See `here
    <https://www.aclweb.org/anthology/D15-1174/>`__ for paper by Toutanova et
    al. originally presenting the dataset.

    Parameters
    ----------
    data_home: str, optional
        Path to the `torchkge_data` directory (containing data folders). If
        files are not present on disk in this directory, they are downloaded
        and then placed in the right place.

    Returns
    -------
    kg_train: torchkge.data_structures.KnowledgeGraph
    kg_val: torchkge.data_structures.KnowledgeGraph
    kg_test: torchkge.data_structures.KnowledgeGraph

    """
    if data_home is None:
        data_home = get_data_home()
    data_path = data_home + '/FB15k237'
    if not exists(data_path):
        makedirs(data_path, exist_ok=True)
        urlretrieve("https://graphs.telecom-paristech.fr/data/torchkge/kgs/FB15k237.zip",
                    data_home + '/FB15k237.zip')
        with zipfile.ZipFile(data_home + '/FB15k237.zip', 'r') as zip_ref:
            zip_ref.extractall(data_home)
        remove(data_home + '/FB15k237.zip')
        shutil.rmtree(data_home + '/__MACOSX')

    df1 = read_csv(data_path + '/train.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df2 = read_csv(data_path + '/valid.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df3 = read_csv(data_path + '/test.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df = concat([df1, df2, df3])
    kg = KnowledgeGraph(df)

    return kg.split_kg(sizes=(len(df1), len(df2), len(df3)))


def load_wn18(data_home=None):
    """Load WN18 dataset.

    Parameters
    ----------
    data_home: str, optional
        Path to the `torchkge_data` directory (containing data folders). If
        files are not present on disk in this directory, they are downloaded
        and then placed in the right place.

    Returns
    -------
    kg_train: torchkge.data_structures.KnowledgeGraph
    kg_val: torchkge.data_structures.KnowledgeGraph
    kg_test: torchkge.data_structures.KnowledgeGraph

    """
    if data_home is None:
        data_home = get_data_home()
    data_path = data_home + '/WN18'
    if not exists(data_path):
        makedirs(data_path, exist_ok=True)
        urlretrieve("https://graphs.telecom-paristech.fr/data/torchkge/kgs/WN18.zip",
                    data_home + '/WN18.zip')
        with zipfile.ZipFile(data_home + '/WN18.zip', 'r') as zip_ref:
            zip_ref.extractall(data_home)
        remove(data_home + '/WN18.zip')
        shutil.rmtree(data_home + '/__MACOSX')

    df1 = read_csv(data_path + '/wordnet-mlj12-train.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df2 = read_csv(data_path + '/wordnet-mlj12-valid.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df3 = read_csv(data_path + '/wordnet-mlj12-test.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df = concat([df1, df2, df3])
    kg = KnowledgeGraph(df)

    return kg.split_kg(sizes=(len(df1), len(df2), len(df3)))


def load_wn18rr(data_home=None):
    """Load WN18RR dataset. See `here
    <https://arxiv.org/abs/1707.01476>`__ for paper by Dettmers et
    al. originally presenting the dataset.

    Parameters
    ----------
    data_home: str, optional
        Path to the `torchkge_data` directory (containing data folders). If
        files are not present on disk in this directory, they are downloaded
        and then placed in the right place.

    Returns
    -------
    kg_train: torchkge.data_structures.KnowledgeGraph
    kg_val: torchkge.data_structures.KnowledgeGraph
    kg_test: torchkge.data_structures.KnowledgeGraph

    """
    if data_home is None:
        data_home = get_data_home()
    data_path = data_home + '/WN18RR'
    if not exists(data_path):
        makedirs(data_path, exist_ok=True)
        urlretrieve("https://graphs.telecom-paristech.fr/data/torchkge/kgs/WN18RR.zip",
                    data_home + '/WN18RR.zip')
        with zipfile.ZipFile(data_home + '/WN18RR.zip', 'r') as zip_ref:
            zip_ref.extractall(data_home)
        remove(data_home + '/WN18RR.zip')
        shutil.rmtree(data_home + '/__MACOSX')

    df1 = read_csv(data_path + '/train.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df2 = read_csv(data_path + '/valid.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df3 = read_csv(data_path + '/test.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df = concat([df1, df2, df3])
    kg = KnowledgeGraph(df)

    return kg.split_kg(sizes=(len(df1), len(df2), len(df3)))


def load_yago3_10(data_home=None):
    """Load YAGO3-10 dataset. See `here
    <https://arxiv.org/abs/1707.01476>`__ for paper by Dettmers et
    al. originally presenting the dataset.

    Parameters
    ----------
    data_home: str, optional
        Path to the `torchkge_data` directory (containing data folders). If
        files are not present on disk in this directory, they are downloaded
        and then placed in the right place.

    Returns
    -------
    kg_train: torchkge.data_structures.KnowledgeGraph
    kg_val: torchkge.data_structures.KnowledgeGraph
    kg_test: torchkge.data_structures.KnowledgeGraph

    """
    if data_home is None:
        data_home = get_data_home()
    data_path = data_home + '/YAGO3-10'
    if not exists(data_path):
        makedirs(data_path, exist_ok=True)
        urlretrieve("https://graphs.telecom-paristech.fr/data/torchkge/kgs/YAGO3-10.zip",
                    data_home + '/YAGO3-10.zip')
        with zipfile.ZipFile(data_home + '/YAGO3-10.zip', 'r') as zip_ref:
            zip_ref.extractall(data_home)
        remove(data_home + '/YAGO3-10.zip')
        shutil.rmtree(data_home + '/__MACOSX')

    df1 = read_csv(data_path + '/train.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df2 = read_csv(data_path + '/valid.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df3 = read_csv(data_path + '/test.txt',
                   sep='\t', header=None, names=['from', 'rel', 'to'])
    df = concat([df1, df2, df3])
    kg = KnowledgeGraph(df)

    return kg.split_kg(sizes=(len(df1), len(df2), len(df3)))


def load_wikidatasets(which, limit_=None, data_home=None):
    """Load WikiDataSets dataset. See `here
    <https://arxiv.org/abs/1906.04536>`__ for paper by Boschin et al.
    originally presenting the dataset.

    Parameters
    ----------
    which: str
        String indicating which subset of Wikidata should be loaded.
        Available ones are `humans`, `companies`, `animals`, `countries` and
        `films`.
    limit_: int, optional (default=0)
        This indicates a lower limit on the number of neighbors an entity
        should have in the graph to be kept.
    data_home: str, optional
        Path to the `torchkge_data` directory (containing data folders). If
        files are not present on disk in this directory, they are downloaded
        and then placed in the right place.

    Returns
    -------
    kg_train: torchkge.data_structures.KnowledgeGraph
    kg_val: torchkge.data_structures.KnowledgeGraph
    kg_test: torchkge.data_structures.KnowledgeGraph

    """
    assert which in ['humans', 'companies', 'animals', 'countries', 'films']

    if data_home is None:
        data_home = get_data_home()

    data_home = data_home + '/WikiDataSets'
    data_path = data_home + '/' + which
    if not exists(data_path):
        makedirs(data_path, exist_ok=True)
        urlretrieve("https://graphs.telecom-paristech.fr/WikiDataSets/{}.tar.gz".format(which),
                    data_home + '/{}.tar.gz'.format(which))

        with tarfile.open(data_home + '/{}.tar.gz'.format(which), 'r') as tf:
            tf.extractall(data_home)
        remove(data_home + '/{}.tar.gz'.format(which))

    df = read_csv(data_path + '/edges.txt'.format(which), sep='\t', header=1,
                  names=['from', 'to', 'rel'])

    a = df.groupby('from').count()['rel']
    b = df.groupby('to').count()['rel']

    # Filter out nodes with too few facts
    tmp = merge(right=DataFrame(a).reset_index(),
                left=DataFrame(b).reset_index(),
                how='outer', right_on='from', left_on='to', ).fillna(0)

    tmp['rel'] = tmp['rel_x'] + tmp['rel_y']
    tmp = tmp.drop(['from', 'rel_x', 'rel_y'], axis=1)

    tmp = tmp.loc[tmp['rel'] >= limit_]
    df_bis = df.loc[df['from'].isin(tmp['to']) | df['to'].isin(tmp['to'])]

    kg = KnowledgeGraph(df_bis)
    kg_train, kg_val, kg_test = kg.split_kg(share=0.8, validation=True)

    return kg_train, kg_val, kg_test
