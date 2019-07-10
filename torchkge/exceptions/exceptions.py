# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
aboschin@enst.fr
"""


class NotYetEvaluated(Exception):
    def __init__(self, message):
        super().__init__(message)
