``smooth_rf`` package
=====================

.. image:: https://travis-ci.com/benjaminleroy/smooth_rf.svg?token=2pSzdX1d2UgeEbzsmGvQ&branch=master
    :target: https://travis-ci.com/benjaminleroy/smooth_rf

.. image:: https://coveralls.io/repos/github/benjaminleroy/smooth_rf/badge.svg?branch=master
    :target: https://coveralls.io/github/benjaminleroy/smooth_rf?branch=master


This repository provides an implimentation of smoothed random forests (cite) in ``python``. Specifically, it provides a ``python`` package that wraps around a |sklearn|_ random forest object, smooths it, and returns an updated random forest object.

.. |sklearn| replace:: ``sklearn``
.. _sklearn: https://scikit-learn.org

Basic Example
=============

.. code-block:: python

  import numpy as np
  import pandas as pd
  import sklearn.ensemble
  import sklearn.datasets
  import sklearn.metrics
  import smooth_rf

  np.random.seed(2019)

  data_train, y_train = sklearn.datasets.make_moons(n_samples=300,
                                                    noise=.3)
  data_test, y_test = sklearn.datasets.make_moons(n_samples=10000,
                                                  noise=.3)

  rf = sklearn.ensemble.RandomForestClassifier(n_estimators = 10)
  rf_model = rf.fit(data_train, y_train)

  # make more straight-forward
  smooth_rf_model, _ ,_,_ = smooth_rf.smooth(rf_model, X_trained = data_train,
                                             y_trained = y_train.ravel(),
                                             sgd_max_num = 10000,
                                             parents_all = True,
                                             resample_tune = False,
                                             no_constraint = False,
                                             adam = {"alpha": .001,
                                                     "beta_1": .9,
                                                     "beta_2": .999,
                                                     "eps": 1e-8})

  yhat_test_base = rf_model.predict(data_test)
  yhat_test_smooth = smooth_rf.predict(data_test)

  sklearn.metrics.accuracy_score(yhat_test_base,y_test)
  sklearn.metrics.accuracy_score(yhat_test_smooth,y_test)


Citation & Installation
=======================
... to be filled in ...


