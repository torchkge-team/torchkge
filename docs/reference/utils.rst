.. _utils:


Utils
*****

.. currentmodule:: torchkge.utils

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
.. autoclass:: torchkge.utils.losses.MSE
    :members:


Operations
----------

.. autofunction:: torchkge.utils.operations.get_mask
.. autofunction:: torchkge.utils.operations.get_rolling_matrix
.. autofunction:: torchkge.utils.operations.get_rank
.. autofunction:: torchkge.utils.operations.pad_with_last_value
.. autofunction:: torchkge.utils.operations.concatenate_diff_sizes
.. autofunction:: torchkge.utils.operations.process_dissimilarities
