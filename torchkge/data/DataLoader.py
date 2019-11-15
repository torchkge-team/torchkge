# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
aboschin@enst.fr
This module's code is freely adapted from Scikit-Learn's sklearn.datasets.base.py code.
"""

from .KnowledgeGraph import KnowledgeGraph

from os import environ, makedirs, remove
from os.path import exists, expanduser, join
from pandas import read_csv, concat, merge, DataFrame
from urllib.request import urlretrieve

import shutil
import tarfile
import zipfile


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


def load_fb13(data_home=None):
    if data_home is None:
        data_home = get_data_home()
    data_path = data_home + '/FB13'
    if not exists(data_path):
        makedirs(data_path, exist_ok=True)
        urlretrieve("https://graphs.telecom-paristech.fr/datasets/FB13.zip",
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
    if data_home is None:
        data_home = get_data_home()
    data_path = data_home + '/FB15k'
    if not exists(data_path):
        makedirs(data_path, exist_ok=True)
        urlretrieve("https://graphs.telecom-paristech.fr/datasets/FB15k.zip",
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
    if data_home is None:
        data_home = get_data_home()
    data_path = data_home + '/FB15k237'
    if not exists(data_path):
        makedirs(data_path, exist_ok=True)
        urlretrieve("https://graphs.telecom-paristech.fr/datasets/FB15k237.zip",
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
    if data_home is None:
        data_home = get_data_home()
    data_path = data_home + '/WN18'
    if not exists(data_path):
        makedirs(data_path, exist_ok=True)
        urlretrieve("https://graphs.telecom-paristech.fr/datasets/WN18.zip",
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


def load_wikidatasets(which, limit_=None, data_home=None):
    assert which in ['humans', 'companies', 'animals', 'countries', 'films']

    if limit_ is None:
        limit_ = 0

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
