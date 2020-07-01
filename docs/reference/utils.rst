.. _utils:


Utils
*****

.. currentmodule:: torchkge.utils

Datasets loaders
----------------

.. autofunction:: torchkge.utils.datasets.load_fb13
.. autofunction:: torchkge.utils.datasets.load_fb15k
.. autofunction:: torchkge.utils.datasets.load_fb15k237
.. autofunction:: torchkge.utils.datasets.load_wn18
.. autofunction:: torchkge.utils.datasets.load_wn18rr
.. autofunction:: torchkge.utils.datasets.load_yago3_10
.. autofunction:: torchkge.utils.datasets.load_wikidatasets

Pre-trained models
------------------
.. autofunction:: torchkge.utils.pretrained_models.load_pretrained_transe

Data redundancy
---------------
.. autofunction:: torchkge.utils.data_redundancy.duplicates
.. autofunction:: torchkge.utils.data_redundancy.count_triplets
.. autofunction:: torchkge.utils.data_redundancy.cartesian_product_relations

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
.. autoclass:: torchkge.utils.losses.BinaryCrossEntropyLoss
    :members:

Training wrappers
-----------------
.. autoclass:: torchkge.utils.training.TrainDataLoader
    :members:
.. autoclass:: torchkge.utils.training.Trainer
    :members:
