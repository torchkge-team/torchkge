=======
History
=======

0.10.0 (2019-07-19)
-------------------

* Implemented Triplet Classification evaluation method
* Added Negative Sampler objects to standardize negative sampling methods.


0.9.0 (2019-07-17)
------------------

* Implemented HolE model (Nickel et al.)
* Implemented ComplEx model (Trouillon et al.)
* Implemented ANALOGY model (Liu et al.)
* Added knowledge graph splitting into train, validation and test instead of just train and test.

0.8.0 (2019-07-09)
------------------

* Implemented Bernoulli negative sampling as in Wang et al. paper on TransH (2014).

0.7.0 (2019-07-01)
------------------

* Implemented Mean Reciprocal Rank measure of performance.
* Implemented Logistic Loss.
* Changed implementation of margin loss to use torch methods.

0.6.0 (2019-06-25)
------------------

* Implemented DistMult

0.5.0 (2019-06-24)
------------------

* Changed implementation of LinkPrediction ranks by moving functions to model methods.
* Implemented RESCAL.


0.4.0 (2019-05-15)
------------------

* Fixed a major bug/problem in the Evaluation protocol of LinkPrediction.

0.3.1 (2019-05-10)
------------------

* Minor bug fixes in the various normalization functions.

0.3.0 (2019-05-09)
------------------

* Fixed CUDA support.

0.2.0 (2019-05-07)
------------------

* Added support for filtered performance measures.

0.1.7 (2019-04-03)
------------------

* First real release on PyPI.

0.1.0 (2019-04-01)
------------------

* First release on PyPI.
