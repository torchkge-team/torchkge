from .dissimilarities import l1_dissimilarity, l2_dissimilarity
from .dissimilarities import l1_torus_dissimilarity, l2_torus_dissimilarity, el2_torus_dissimilarity
from .losses import MarginLoss, LogisticLoss

from .models_utils import init_embedding, get_true_targets
from .negative_sampling import get_possible_heads_tails
from .operations import get_rank, get_mask, get_rolling_matrix, get_n_batch, get_batches
from .preprocessing import get_dictionaries, get_bernoulli_probs
