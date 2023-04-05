# -*- coding: utf-8 -*-

"""Top-level package for TorchKGE."""

__author__ = """Armand Boschin"""
__email__ = 'aboschin@enst.fr'
__version__ = '0.17.7'

from torchkge.exceptions import NotYetEvaluatedError
from torchkge.utils import MarginLoss, LogisticLoss
from torchkge.utils import l1_dissimilarity, l2_dissimilarity
from .data_structures import KnowledgeGraph
from .evaluation import LinkPredictionEvaluator
from .evaluation import TripletClassificationEvaluator
from .models import ConvKBModel
from .models import RESCALModel, DistMultModel, HolEModel, ComplExModel, AnalogyModel
from .models import TransEModel, TransHModel, TransRModel, TransDModel, TorusEModel
