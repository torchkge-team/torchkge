# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
armand.boschin@telecom-paristech.fr
"""

from torch.nn import Module
import torch.nn.functional as F


class MarginLoss(Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, output):
        golden_triplets, negative_triplets = output[0], output[1]
        return F.relu(self.margin + golden_triplets - negative_triplets).sum()
