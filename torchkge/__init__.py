# -*- coding: utf-8 -*-

"""Top-level package for TorchKGE."""

__author__ = """Armand Boschin"""
__email__ = 'aboschin@enst.fr'
__version__ = '0.15.4'

from .data import KnowledgeGraph

from .models import TransEModel, TransHModel, TransRModel, TransDModel
from .models import RESCALModel, DistMultModel
from .models import ConvKBModel

from torchkge.utils import l1_dissimilarity, l2_dissimilarity
from torchkge.utils import MarginLoss, LogisticLoss

from .evaluation.LinkPrediction import LinkPredictionEvaluator

from torchkge.exceptions import NotYetEvaluatedError
