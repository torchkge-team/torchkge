import os
import pickle
import pandas as pd

from getopt import getopt
from torch.optim import SGD, Adam

from torchkge.utils import Config
from torchkge.models import MarginLoss, MSE
from torchkge.models import TransEModel, TransHModel, TransRModel, TransDModel
from torchkge.models import RESCALModel, DistMultModel

from torchkge.data import KnowledgeGraph
from torchkge.evaluation import LinkPredictionEvaluator, l2_dissimilarity


def load_parameters(args):
    optlist, args = getopt(args, 'x', ['model=', 'optimizer=', 'criterion=', 'dataset=',
                                       'nb_epochs=', 'b_size=', 'b_size_eval=', 'cuda=', 'device=',
                                       'ent_emb_dim=', 'rel_emb_dim=', 'norm_type=',
                                       'lr=',
                                       'margin=',
                                       'limit_neighbors=',
                                       'test_epochs='
                                       ])

    d = {}
    for i, j in optlist:
        d[i[2:]] = j

    types = {'model': str, 'optimizer': str, 'criterion': str, 'dataset': str,
             'nb_epochs': int, 'b_size': int, 'b_size_eval': int, 'cuda': bool, 'device': int,
             'ent_emb_dim': int, 'rel_emb_dim': int, 'norm_type': int,
             'lr': float, 'margin': float, 'limit_neighbors': int, 'test_epochs': int}

    try:
        for k in args.keys():
            d[k] = types[k](d[k])
    except KeyError as e:
        print("key problem with one of the parameters")
        raise e

    for k in set(types.keys()) - set(d.keys()):
        d[k] = None

    if d['test_epochs'] is None:
        d['test_epochs'] = d['nb_epochs']

    assert d['model'] in ["transe", "transr", "transh", "transd", 'rescal', 'distmult']

    return d


def get_model(params, n_ent, n_rel):
    if params['model'] == 'transe':
        print('Using Model TransE')
        config = Config(ent_emb_dim=params['ent_emb_dim'], rel_emb_dim=params['ent_emb_dim'],
                        n_ent=n_ent, n_rel=n_rel, norm_type=params['norm_type'])
        return TransEModel(config, dissimilarity=l2_dissimilarity)
    elif params['model'] == 'transh':
        print('Using Model TransH')
        config = Config(ent_emb_dim=params['ent_emb_dim'], rel_emb_dim=params['rel_emb_dim'],
                        n_ent=n_ent, n_rel=n_rel, norm_type=params['norm_type'])
        return TransHModel(config, dissimilarity=l2_dissimilarity)
    elif params['model'] == 'transr':
        print('Using Model TransR')
        config = Config(ent_emb_dim=params['ent_emb_dim'], rel_emb_dim=params['rel_emb_dim'],
                        n_ent=n_ent, n_rel=n_rel, norm_type=params['norm_type'])
        return TransRModel(config, dissimilarity=l2_dissimilarity)
    elif params['model'] == 'transd':
        print('Using Model TransD')
        config = Config(ent_emb_dim=params['ent_emb_dim'], rel_emb_dim=params['rel_emb_dim'],
                        n_ent=n_ent, n_rel=n_rel, norm_type=params['norm_type'])
        return TransDModel(config, dissimilarity=l2_dissimilarity)
    elif params['model'] == 'distmult':
        print('Using model DistMult')
        config = Config(ent_emb_dim=params['ent_emb_dim'], n_ent=n_ent, n_rel=n_rel)
        return DistMultModel(config)
    elif params['model'] == 'rescal':
        print('Using model RESCAL')
        config = Config(ent_emb_dim=params['ent_emb_dim'], n_ent=n_ent, n_rel=n_rel)
        return RESCALModel(config)
    else:
        return None


def get_optimizer(model, params):
    if params['optimizer'] == 'sgd':
        return SGD(model.parameters(), lr=params['lr'])
    if params['optimize'] == 'adam':
        return Adam(model.parameters(), lr=params['lr'])


def get_criterion(params):
    if params['criterion'] == 'margin':
        return MarginLoss(params['margin'])

    if params['criterion'] == 'mse':
        return MSE()


def get_dataset(params):
    if params['dataset'] == 'fb15k':
        kg_train, kg_test = load_fb15k()

    else:
        kg_train, kg_test = load_wikidataset(params['dataset'], params['limit_neighbors'])
    return kg_train, kg_test


def save_current(model, kg_train, kg_test, b_size_eval, dicts, epoch, path):
    model.normalize_parameters()
    train_evaluator = LinkPredictionEvaluator(model, kg_train)
    test_evaluator = LinkPredictionEvaluator(model, kg_test)

    train_evaluator.evaluate(batch_size=b_size_eval, k_max=30)
    test_evaluator.evaluate(batch_size=b_size_eval, k_max=30)

    tmp = train_evaluator.hit_at_k(10)
    dicts[0][epoch + 1] = tmp[0]  # training hit@k
    dicts[1][epoch + 1] = tmp[1]  # filtered training hit@k

    tmp = test_evaluator.hit_at_k(10)
    dicts[2][epoch + 1] = tmp[0]  # testing hit@k
    dicts[3][epoch + 1] = tmp[1]  # filtered testing hit@k

    tmp = train_evaluator.mean_rank()
    dicts[4][epoch + 1] = tmp[0]  # training mean rank
    dicts[5][epoch + 1] = tmp[1]  # filtered training mean rank

    tmp = test_evaluator.mean_rank()
    dicts[6][epoch + 1] = tmp[0]  # testing mean rank
    dicts[7][epoch + 1] = tmp[1]  # filtered testing mean rank

    print(
        '[%d], Training Hit@10 : %.3f, Training MeanRank : %.3f, '
        'Testing Hit@10 : %.3f, Testing MeanRank : %.3f' % (
            epoch + 1, dicts[0][epoch + 1], dicts[4][epoch + 1],
            dicts[2][epoch + 1], dicts[6][epoch + 1]))

    print(
        '[%d], Filt. Training Hit@10 : %.3f, Filt. Training MeanRank : %.3f, '
        'Filt. Testing Hit@10 : %.3f, Filt. Testing MeanRank : %.3f' % (
            epoch + 1, dicts[1][epoch + 1], dicts[5][epoch + 1],
            dicts[3][epoch + 1], dicts[7][epoch + 1]))

    with open(path + 'model_epoch{}.pkl'.format(epoch + 1), 'wb') as f:
        try:
            pickle.dump((model, None, test_evaluator), f)
        except OverflowError:
            pickle.dump((model, None, test_evaluator), f, protocol=4)

    return dicts


def load_fb15k():
    df1 = pd.read_csv('/home/aboschin/datasets/FB15K/freebase_mtr100_mte100-train.txt',
                      sep='\t', header=None, names=['from', 'rel', 'to'])
    df2 = pd.read_csv('/home/aboschin/datasets/FB15K/freebase_mtr100_mte100-test.txt',
                      sep='\t', header=None, names=['from', 'rel', 'to'])
    df = pd.concat([df1, df2])

    kg = KnowledgeGraph(df)
    print('{} entities, {} relations, {} facts'.format(kg.n_ent, kg.n_rel, kg.n_sample))
    kg_train, kg_test = kg.split_kg(train_size=len(df1))
    print('Knowledge graph split')
    return kg_train, kg_test


def load_wikidataset(set_, limit_, split=True):
    if limit_ is None:
        limit_ = 0
    # Load data set_
    df = pd.read_csv('/home/public/WikiDataSets/{}/edges.txt'.format(set_), sep='\t',
                     names=['from', 'to', 'rel'], header=1)

    a = df.groupby('from').count()['rel']
    b = df.groupby('to').count()['rel']

    # Filter out nodes with too few facts
    tmp = pd.merge(right=pd.DataFrame(a).reset_index(),
                   left=pd.DataFrame(b).reset_index(),
                   how='outer', right_on='from', left_on='to', ).fillna(0)

    tmp['rel'] = tmp['rel_x'] + tmp['rel_y']
    tmp = tmp.drop(['from', 'rel_x', 'rel_y'], axis=1)

    tmp = tmp.loc[tmp['rel'] >= limit_]
    df_bis = df.loc[df['from'].isin(tmp['to']) | df['to'].isin(tmp['to'])]

    kg = KnowledgeGraph(df_bis)
    print('{} entities, {} relations, {} facts'.format(kg.n_ent, kg.n_rel, kg.n_sample))
    if split:
        kg_train, kg_test = kg.split_kg()
        print('Knowledge graph split')

        return kg_train, kg_test
    else:
        return kg


def get_path(model_letter, dataset_name):
    path = '/home/aboschin/learning_curves/data/{}/'.format(dataset_name)
    if not os.path.exists(path):
        os.makedirs(path)
    if len(model_letter) == 1:
        path = path + 'trans{}/'.format(model_letter)
    else:
        path = path + '{}/'.format(model_letter)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
