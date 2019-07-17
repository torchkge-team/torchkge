# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
aboschin@enst.fr
"""


class NotYetEvaluatedError(Exception):
    def __init__(self, message):
        super().__init__(message)


class SizeMismatchError(Exception):
    def __init__(self, message):
        super().__init__(message)


class WrongDimensionError(Exception):
    def __init__(self, message):
        super().__init__(message)
