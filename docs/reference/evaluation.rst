.. _evaluation:


Evaluation
**********

Link Prediction
---------------
To assess the performance of the link prediction evaluation module of TorchKGE, it was compared with the ones of
`AmpliGraph <https://docs.ampligraph.org/>`_ (v1.3.1) and `OpenKE <https://github.com/thunlp/OpenKE>`_(version of
April, 9). The computation times (in seconds) reported in the following table are averaged over 5 independent evaluation
processes. Experiments were done using PyTorch 1.5, TensorFlow 1.15 and a Tesla K80 GPU. Missing values for AmpliGraph
are due to missing models in the library.

.. tabularcolumns:: p{2cm}p{3cm}p{3cm}p{3cm}p{3cm}p{3cm}p{3cm}p{3cm}p{3cm}

+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| Model     | TransE                | TransD                | RESCAL                | ComplEx               |
+===========+===========+===========+===========+===========+===========+===========+===========+===========+
| Dataset   |FB15k      | WN18      | FB15k     | WN18      | FB15k     | WN18      | FB15k     | WN18      |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
|AmpliGraph | 354.8     | 39.8      |           |           |           |           | 537.2     | 94.9      |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
|OpenKE     | 235.6     | 42.2      | 258.5     | 43.7      | 789.1     |  178.4    | 354.7     | 63.9      |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
|TorchKGE   | 76.1      | 13.8      | 60.8      | 11.1      | 46.9      |  7.1      | 96.4      | 18.6      |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+

.. autoclass:: torchkge.evaluation.link_prediction.LinkPredictionEvaluator
    :members:

Triplet Classification
----------------------
.. autoclass:: torchkge.evaluation.triplet_classification.TripletClassificationEvaluator
    :members:
