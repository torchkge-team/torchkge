from .data import DataLoader, get_data_home, clear_data_home

from .dissimilarities import l1_dissimilarity, l2_dissimilarity
from .dissimilarities import l1_torus_dissimilarity, l2_torus_dissimilarity, \
    el2_torus_dissimilarity

from .losses import MarginLoss, LogisticLoss

from .modelling import init_embedding, get_true_targets, load_embeddings

from .operations import get_rank, get_mask, get_dictionaries, \
    get_bernoulli_probs

from .training import Trainer, TrainDataLoader
