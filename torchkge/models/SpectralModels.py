from torchkge.models import Model

from sknetwork.embedding import GSVD


class GSVDModel(object):

    def __init__(self, biad, k, eps):
        self.biad = biad
        self.k = k
        self.eps = eps
        self.emb

    def embed(self):
        emb = GSVD(embedding_dimension=k, regularization=eps)
        emb.fit(biad)
        return emb.embedding_, emb.features_

