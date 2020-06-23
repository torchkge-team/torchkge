.. _models:

Models
******

Interfaces
==========

Model
-----
.. autoclass:: torchkge.models.interfaces.Model
   :members:

TranslationalModels
-------------------
.. autoclass:: torchkge.models.interfaces.TranslationModel
   :members:

Translational Models
====================

Parameters used to train models available in pre-trained version :

.. tabularcolumns:: p{2cm}p{3cm}p{3cm}p{3cm}p{3cm}p{3cm}p{3cm}p{3cm}p{3cm}

+-------+-----------+-----------+-----------+---------------+------------+--------+--------+-----------------+
|       | Dataset   | Dimension | Optimizer | Learning Rate | Batch Size | Loss   | Margin | L2 penalization |
+-------+-----------+-----------+-----------+---------------+------------+--------+--------+-----------------+
|TransE | FB15k     | 100       | Adam      | 2.1e-5        | 32768      | Margin | .651   | 1e-5            |
+-------+-----------+-----------+-----------+---------------+------------+--------+--------+-----------------+
|TransE | FB15k237  | 100       | Adam      | 2.1e-5        | 32768      | Margin | .651   | 1e-5            |
+-------+-----------+-----------+-----------+---------------+------------+--------+--------+-----------------+
|TransE | FB15k237  | 150       | Adam      | 2.7e-5        | 32768      | Margin | .648   | 1e-5            |
+-------+-----------+-----------+-----------+---------------+------------+--------+--------+-----------------+

TransE
------
.. autoclass:: torchkge.models.translation.TransEModel
   :members:

TransH
------
.. autoclass:: torchkge.models.translation.TransHModel
   :members:

TransR
------
.. autoclass:: torchkge.models.translation.TransRModel
   :members:

TransD
------
.. autoclass:: torchkge.models.translation.TransDModel
   :members:

TorusE
------
.. autoclass:: torchkge.models.translation.TorusEModel
   :members:

Bilinear Models
===============

RESCAL
------
.. autoclass:: torchkge.models.bilinear.RESCALModel
   :members:

DistMult
--------
.. autoclass:: torchkge.models.bilinear.DistMultModel
   :members:

HolE
----
.. autoclass:: torchkge.models.bilinear.HolEModel
   :members:

ComplEx
-------
.. autoclass:: torchkge.models.bilinear.ComplExModel
   :members:

ANALOGY
-------
.. autoclass:: torchkge.models.bilinear.AnalogyModel
   :members:

Deep Models
===========

ConvKB
------
.. autoclass:: torchkge.models.deep.ConvKBModel
   :members:
