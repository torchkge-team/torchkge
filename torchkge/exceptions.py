# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
armand.boschin@telecom-paristech.fr
"""


class NotYetEvaluated(Exception):
    def __init__(self, message):
        super().__init__(message)
