from itertools import count
from collections import namedtuple

KBIndex = namedtuple('KBIndex', ['ent_list', 'rel_list', 'ent_id', 'rel_id'])

def index_ent_rel(*filenames):
    ent_set = set()
    rel_set = set()
    for filename in filenames:
        with open(filename) as f:
            for ln in f:
                s, r, t = ln.strip().split('\t')[:3]
                ent_set.add(s)
                ent_set.add(t)
                rel_set.add(r)
    ent_list = sorted(list(ent_set))
    rel_list = sorted(list(rel_set))
    ent_id = dict(zip(ent_list, count()))
    rel_id = dict(zip(rel_list, count()))
    return KBIndex(ent_list, rel_list, ent_id, rel_id)


def graph_size(kb_index):
    return len(kb_index.ent_id), len(kb_index.rel_id)


def read_data(filename, kb_index):
    src = []
    rel = []
    dst = []
    with open(filename) as f:
        for ln in f:
            s, r, t = ln.strip().split('\t')
            src.append(kb_index.ent_id[s])
            rel.append(kb_index.rel_id[r])
            dst.append(kb_index.ent_id[t])
    return src, rel, dst
