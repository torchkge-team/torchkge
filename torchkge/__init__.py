# -*- coding: utf-8 -*-

"""Top-level package for TorchKGE."""

__author__ = """Armand Boschin"""
__email__ = 'armand.boschin@telecom-paristech.fr'
__version__ = '0.1.2'


from .evaluation.dissimilarities import l1_dissimilarity
from .evaluation.dissimilarities import l2_dissimilarity
from .evaluation.linkprediction import LinkPredictionEvaluator

from .models.losses import MarginLoss
from .models.translation_models import TransEModel
from .models.translation_models import TransHModel
from .models.translation_models import TransRModel
from .models.translation_models import TransDModel

from .data.KnowledgeGraph import KnowledgeGraph

from .utils import Config
from .exceptions import NotYetEvaluated
