import os
import logging
import torch as t
import torch.nn as nn
import torch.nn.functional as f
from torchkge.torchkge.config import config
from torch.optim import Adam, SGD, Adagrad
from torch.autograd import Variable
from torchkge.torchkge.data_utils import batch_by_num
from torchkge.torchkge.base_model import BaseModel, BaseModule

################
import os
import logging
import torch
# from .corrupter import BernCorrupter, BernCorrupterMulti
from torchkge.torchkge.sampling import KbganNegativeSampler
from torchkge.torchkge.read_data import index_ent_rel, graph_size, read_data
from torchkge.torchkge.config import config, overwrite_config_with_args
from torchkge.torchkge.logger_init import logger_init
from torchkge.torchkge.data_utils import inplace_shuffle, heads_tails
from torchkge.torchkge.select_gpu import select_gpu
# from .trans_e import TransE
# from .trans_d import TransD
# from .distmult import DistMult
# from .compl_ex import ComplEx

logger_init()
if torch.cuda.is_available():
    torch.cuda.set_device(select_gpu())
overwrite_config_with_args()

task_dir = config().task.dir
kb_index = index_ent_rel(os.path.join(task_dir, 'train.txt'),
                         os.path.join(task_dir, 'valid.txt'),
                         os.path.join(task_dir, 'test.txt'))
n_ent, n_rel = graph_size(kb_index)

train_data = read_data(os.path.join(task_dir, 'train.txt'), kb_index)
inplace_shuffle(*train_data)
valid_data = read_data(os.path.join(task_dir, 'valid.txt'), kb_index)
test_data = read_data(os.path.join(task_dir, 'test.txt'), kb_index)
heads, tails = heads_tails(n_ent, train_data, valid_data, test_data)
valid_data = [torch.LongTensor(vec) for vec in valid_data]
test_data = [torch.LongTensor(vec) for vec in test_data]
tester = lambda: gen.test_link(valid_data, n_ent, heads, tails)
train_data = [torch.LongTensor(vec) for vec in train_data]

mdl_type = config().pretrain_config
gen_config = config()[mdl_type]

sampler = KbganNegativeSampler(train_data, n_ent, n_rel)


#################


class TransEModule(BaseModule):
    def __init__(self, n_ent, n_rel, config):
        super(TransEModule, self).__init__()
        self.p = config.p
        self.margin = config.margin
        self.temp = config.get('temp', 1)
        self.rel_embed = nn.Embedding(n_rel, config.dim)
        self.ent_embed = nn.Embedding(n_ent, config.dim)
        self.init_weight()

    def init_weight(self):
        for param in self.parameters():
            param.data.normal_(1 / param.size(1) ** 0.5)
            param.data.renorm_(2, 0, 1)

    def forward(self, src, rel, dst):
        return t.norm(self.ent_embed(dst) - self.ent_embed(src) - self.rel_embed(rel) + 1e-30, p=self.p, dim=-1)

    def dist(self, src, rel, dst):
        return self.forward(src, rel, dst)

    def score(self, src, rel, dst):
        return self.forward(src, rel, dst)

    def prob_logit(self, src, rel, dst):
        return -self.forward(src, rel, dst) / self.temp

    def constraint(self):
        self.ent_embed.weight.data.renorm_(2, 0, 1)
        self.rel_embed.weight.data.renorm_(2, 0, 1)


class TransE(BaseModel):
    def __init__(self, n_ent, n_rel, config):
        super(TransE, self).__init__()
        self.mdl = TransEModule(n_ent, n_rel, config)
        if torch.cuda.is_available():
            self.mdl.cuda()
        self.config = config

    def pretrain(self, train_data, sampling, tester):
        src, rel, dst = train_data
        n_train = len(src)
        optimizer = Adam(self.mdl.parameters())
        # optimizer = SGD(self.mdl.parameters(), lr=1e-4)
        n_epoch = self.config.n_epoch
        n_batch = self.config.n_batch
        best_perf = 0
        for epoch in range(n_epoch):
            epoch_loss = 0
            rand_idx = t.randperm(n_train)
            src = src[rand_idx]
            rel = rel[rand_idx]
            dst = dst[rand_idx]
            # src_corrupted, dst_corrupted = corrupter.corrupt(src, rel, dst)
            src_corrupted, dst_corrupted = sampler.corrupt(src, rel, dst)
            if torch.cuda.is_available():
                src = src.cuda()
                rel = rel.cuda()
                dst = dst.cuda()
                src_corrupted = src_corrupted.cuda()
                dst_corrupted = dst_corrupted.cuda()
            for s0, r, t0, s1, t1 in batch_by_num(n_batch, src, rel, dst, src_corrupted, dst_corrupted,
                                                  n_sample=n_train):
                self.mdl.zero_grad()
                loss = t.sum(self.mdl.pair_loss(Variable(s0), Variable(r), Variable(t0), Variable(s1), Variable(t1)))
                loss.backward()
                optimizer.step()
                self.mdl.constraint()
                epoch_loss += loss.data
            logging.info('Epoch %d/%d, Loss=%f', epoch + 1, n_epoch, epoch_loss / n_train)
            if (epoch + 1) % self.config.epoch_per_test == 0:
                test_perf = tester()
                if test_perf > best_perf:
                    self.save(os.path.join(config().task.dir, self.config.model_file))
                    best_perf = test_perf
        return best_perf


gen = TransE(n_ent, n_rel, gen_config)

gen.pretrain(train_data, sampler, tester)
gen.load(os.path.join(task_dir, gen_config.model_file))
gen.test_link(test_data, n_ent, heads, tails)
