# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
aboschin@enst.fr
"""

from torch.nn import Module, MarginRankingLoss, SoftMarginLoss
from torch import ones_like


class MarginLoss(Module):
    def __init__(self, margin):
        super().__init__()
        self.loss = MarginRankingLoss(margin=margin, reduction='sum')

    def forward(self, output):
        golden_triplets, negative_triplets = output[0], output[1]
        return self.loss(golden_triplets, negative_triplets, target=ones_like(golden_triplets))


class LogisticLoss(Module):
    def __init__(self):
        super().__init__()
        self.loss = SoftMarginLoss(reduction='sum')

    def forward(self, output):
        golden_triplets, negative_triplets = output[0], output[1]
        targets = ones_like(golden_triplets)
        return self.loss(golden_triplets, targets) + self.loss(negative_triplets, -targets)


class MSE(Module):
    def __init__(self):
        super().__init__()

    def forward(self, output):
        golden_triplets, negative_triplets = output[0], output[1]
        return 1/2 * (((1 - golden_triplets)**2).sum() + (negative_triplets**2).sum())
