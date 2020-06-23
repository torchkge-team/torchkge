# -*- coding: utf-8 -*-
"""
Copyright TorchKGE developers
@author: Armand Boschin <aboschin@enst.fr>
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


class NotYetImplementedError(Exception):
    def __init__(self, message):
        super().__init__(message)


class WrongArgumentsError(Exception):
    def __init__(self, message):
        super().__init__(message)


class SanityError(Exception):
    def __init__(self, message):
        super().__init__(message)


class SplitabilityError(Exception):
    def __init__(self, message):
        super().__init__(message)


class NoPreTrainedVersionError(Exception):
    def __init__(self, message):
        super().__init__(message)
