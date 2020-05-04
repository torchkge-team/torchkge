.. _utils:


Utils
*****

.. currentmodule:: torchkge.utils

Datasets loaders
----------------

.. autofunction:: torchkge.data.Datasets.load_fb13
.. autofunction:: torchkge.data.Datasets.load_fb15k
.. autofunction:: torchkge.data.Datasets.load_fb15k237
.. autofunction:: torchkge.data.Datasets.load_wn18
.. autofunction:: torchkge.data.Datasets.load_wikidatasets


Dissimilarities
---------------
.. autofunction:: torchkge.utils.dissimilarities.l1_dissimilarity
.. autofunction:: torchkge.utils.dissimilarities.l2_dissimilarity
.. autofunction:: torchkge.utils.dissimilarities.l1_torus_dissimilarity
.. autofunction:: torchkge.utils.dissimilarities.l2_torus_dissimilarity
.. autofunction:: torchkge.utils.dissimilarities.el2_torus_dissimilarity

Losses
------
.. autoclass:: torchkge.utils.losses.MarginLoss
    :members:
.. autoclass:: torchkge.utils.losses.LogisticLoss
    :members:


Operations
----------

.. autofunction:: torchkge.utils.operations.get_mask
.. autofunction:: torchkge.utils.operations.get_rank
