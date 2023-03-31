=======
History
=======

0.17.6 (2023-03-31)
-------------------
* Fix embedding dimension mixup in translation models
* Fix implementation error in some bilinear models
* Fix various docstring typos

0.17.5 (2022-09-18)
-------------------
* Fix bug in TransH implementation

0.17.4 (2022-07-04)
-------------------
* Upgrade dependencies
* Fix normalization in translation models
* Improve loading function of wikidatavitals datasets.

0.17.3 (2022-04-21)
-------------------
* Fix ConvKB scoring function and normalization step

0.17.2 (2022-03-02)
-------------------
* Fix the documentation in evaluation and inference modules
* Fix a typo in the sampling module's documentation

0.17.1 (2022-02-25)
-------------------
* Add support of Python 3.7 back

0.17.0 (2022-02-25)
-------------------
* Add relation prediction evaluation
* Add relation negative sampling module
* Add inference module
* Update models' API accordingly to the previous new features
* Switch from TravisCI to GitHub Actions

0.16.25 (2021-03-01)
--------------------
* Update in available pretrained models

0.16.24 (2021-02-16)
--------------------
* Fix deployment

0.16.23 (2021-02-16)
--------------------
* Removed useless k_max parameter in link-prediction evaluation method

0.16.22 (2021-02-05)
--------------------
* Add pretrained version of TransE for yago310 and ComplEx for fb15k237 and wdv5.

0.16.21 (2021-02-02)
--------------------
* Add pretrained version of TransE for Wikidata-Vitals level 5

0.16.20 (2021-01-22)
--------------------
* Add support for Python 3.8
* Clean up loading process for kgs
* Fix deprecation warning

0.16.19 (2021-01-20)
--------------------
* Fix release

0.16.18 (2021-01-20)
--------------------
* Add data loader for wikidata vitals knowledge graphs

0.16.17 (2020-11-03)
--------------------
* Bug fix get_ranks method

0.16.16 (2020-10-07)
--------------------
* Bug fix in KG split method

0.16.15 (2020-10-07)
--------------------
* Fix WikiDataSets loader (again)

0.16.14 (2020-09-21)
--------------------
* Fix WikiDataSets loader

0.16.13 (2020-08-06)
--------------------
* Fix reduction in BCE loss
* Add pretrained models

0.16.12 (2020-07-07)
--------------------
* Release patch

0.16.11 (2020-07-07)
--------------------
* Fix bug in pre-trained models loading that made all models being redownloaded every time

0.16.10 (2020-07-02)
--------------------
* Minor bug patch

0.16.9 (2020-07-02)
-------------------
* Update urls to retrieve datasets and pre-trained models.

0.16.8 (2020-07-01)
-------------------
* Add binary cross-entropy loss

0.16.7 (2020-06-23)
-------------------
* Change API for pre-trained models

0.16.6 (2020-06-09)
-------------------
* Patch in pre-trained model loading
* Added pre-trained loading for TransE on FB15k237 in dimension 100.

0.16.5 (2020-06-02)
-------------------
* Release patch

0.16.4 (2020-06-02)
-------------------
* Add parameter in data redundancy to exclude know reverse triplets from
  duplicate search.

0.16.3 (2020-05-29)
-------------------
* Release patch

0.16.2 (2020-05-29)
-------------------
* Add methods to compute data redundancy in knowledge graphs as in 2020
  `paper <https://arxiv.org/pdf/2003.08001.pdf>`__ by Akrami et al
  (see references in concerned methods).

0.16.1 (2020-05-28)
-------------------
* Patch an awkward import
* Add dataset loaders for WN18RR and YAGO3-10

0.16.0 (2020-04-27)
-------------------
* Redefinition of the models' API (simplified interfaces, renamed LP
  methods and added get_embeddings method)
* Implementation of the new API for all models
* TorusE implementation fixed
* TransD reimplementation to avoid matmul usage (costly in
  back-propagation)
* Added feature to negative samplers to generate several negative
  samples from each fact. Those can be fed directly to the models.
* Added some wrappers for training to utils module.
* Progress bars now make the most of tqdm's possibilities
* Code reformatting
* Docstrings update

0.15.5 (2020-04-23)
-------------------
* Defined a new homemade and simpler DataLoader class.

0.15.4 (2020-04-22)
-------------------
* Removed the use of torch DataLoader object.

0.15.3 (2020-04-02)
-------------------
* Added a method to print results in link prediction evaluator

0.15.2 (2020-04-01)
-------------------
* Fixed a misfit test

0.15.1 (2020-04-01)
-------------------
* Cleared the definition of rank in link prediction

0.15.0 (2020-04-01)
-------------------
* Improved use of tqdm progress bars

0.14.0 (2020-04-01)
-------------------
* Change in the API of loss functions (margin and logistic loss)
* Documentation update

0.13.0 (2020-02-10)
-------------------
* Added ConvKB model

0.12.1 (2020-01-10)
-------------------
* Minor patch in interfaces
* Comment additions

0.12.0 (2019-12-05)
-------------------
* Various bug fixes
* New KG splitting method enforcing all entities and relations to appear at least once in the training set.

0.11.3 (2019-11-15)
-------------------
* Minor bug fixes

0.11.2 (2019-11-11)
-------------------
* Minor bug fixes

0.11.1 (2019-10-21)
-------------------
* Fixed requirements conflicts

0.11.0 (2019-10-21)
-------------------
* Added TorusE model
* Added dataloaders
* Fixed some bugs

0.10.4 (2019-10-07)
-------------------
* Fixed error in bilinear models.

0.10.3 (2019-07-23)
-------------------
* Added intermediate function for hit@k metric in link prediction.

0.10.2 (2019-07-22)
-------------------
* Fixed assertion error in Analogy model

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
