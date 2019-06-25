# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
aboschin@enst.fr
"""

from torch.nn import Module
from torch.nn import SoftMarginLoss
from torch import ones_like
import torch.nn.functional as F


class MarginLoss(Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, output):
        golden_triplets, negative_triplets = output[0], output[1]
        return F.relu(self.margin + golden_triplets - negative_triplets).sum()


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
