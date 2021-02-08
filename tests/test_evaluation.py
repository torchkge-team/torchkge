import pandas as pd
import unittest

from torch import long

from torchkge.data_structures import KnowledgeGraph
from torchkge.evaluation import LinkPredictionEvaluator, TripletClassificationEvaluator
from torchkge.models import TransEModel


class TestUtils(unittest.TestCase):

    def setUp(self):
        df = pd.DataFrame([[0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 4, 0], [1, 2, 1], [1, 3, 2], [2, 4, 0], [3, 4, 4],
                           [5, 4, 0]], columns=['from', 'to', 'rel'])
        self.kg = KnowledgeGraph(df)

    def checkSanityLinkPrediction(self, evaluator):
        assert evaluator.rank_true_heads.dtype == long
        assert evaluator.rank_true_tails.dtype == long
        assert evaluator.filt_rank_true_heads.dtype == long
        assert evaluator.filt_rank_true_tails.dtype == long

        assert evaluator.rank_true_heads.shape[0] == len(self.kg)
        assert evaluator.rank_true_tails.shape[0] == len(self.kg)
        assert evaluator.filt_rank_true_heads.shape[0] == len(self.kg)
        assert evaluator.filt_rank_true_tails.shape[0] == len(self.kg)

    def test_LinkPredictionEvaluator(self):
        model = TransEModel(100, self.kg.n_ent, self.kg.n_rel, 'L1')

        evaluator = LinkPredictionEvaluator(model, self.kg)
        self.checkSanityLinkPrediction(evaluator)

        evaluator.evaluate(b_size=len(self.kg))
        self.checkSanityLinkPrediction(evaluator)

    def test_TripletClassificationEvaluator(self):
        model = TransEModel(100, self.kg.n_ent, self.kg.n_rel, 'L1')
        kg1, kg2 = self.kg.split_kg(sizes=(4, 5))
        # kg2 contains all relations so it will be used as validation
        evaluator = TripletClassificationEvaluator(model, kg2, kg1)
        assert evaluator.thresholds is None
        assert not evaluator.evaluated

        evaluator.evaluate(b_size=len(self.kg))
        assert evaluator.evaluated
        assert evaluator.thresholds is not None
        assert (len(evaluator.thresholds.shape) == 1) & (evaluator.thresholds.shape[0] == self.kg.n_rel)
