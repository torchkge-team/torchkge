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
.. autofunction:: torchkge.utils.datasets.load_wikidata_vitals


Pre-trained models
------------------

TransE model
============
.. tabularcolumns:: p{3cm}p{3cm}p{3cm}p{3cm}p{3cm}

+-----------+-----------+-----------+----------+--------------------+
| Model     | Dataset   | Dimension | Test MRR | Filtered Test MRR  |
+===========+===========+===========+==========+====================+
| TransE    | FB15k     | 100       | 0.250    | 0.420              |
+-----------+-----------+-----------+----------+--------------------+
| TransE    | FB15k237  | 150       | 0.187    | 0.287              |
+-----------+-----------+-----------+----------+--------------------+
| TransE    | WDV5      | 150       | 0.258    | 0.305              |
+-----------+-----------+-----------+----------+--------------------+
| TransE    | WN18RR    | 100       | 0.201    | 0.236              |
+-----------+-----------+-----------+----------+--------------------+
| TransE    | Yago3-10  | 200       | 0.143    | 0.261              |
+-----------+-----------+-----------+----------+--------------------+

.. autofunction:: torchkge.utils.pretrained_models.load_pretrained_transe

ComplEx Model
=============
.. tabularcolumns:: p{3cm}p{3cm}p{3cm}p{3cm}

+-----------+-----------+-----------+----------+--------------------+
| Model     | Dataset   | Dimension | Test MRR | Filtered Test MRR  |
+===========+===========+===========+==========+====================+
| ComplEx   | FB15k237  | 200       | 0.180    | 0.308              |
+-----------+-----------+-----------+----------+--------------------+
| ComplEx   | WN18RR    | 200       | 0.290    | 0.455              |
+-----------+-----------+-----------+----------+--------------------+
| ComplEx   | WDV5      | 200       | 0.283    | 0.371              |
+-----------+-----------+-----------+----------+--------------------+

.. autofunction:: torchkge.utils.pretrained_models.load_pretrained_complex

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
