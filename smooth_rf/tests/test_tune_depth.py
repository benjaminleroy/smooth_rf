import numpy as np
import pandas as pd
import scipy.sparse
import sparse
import sklearn
from sklearn.ensemble import RandomForestRegressor
from collections import Counter
import sys, os

import smooth_rf

def test_depth_tune_regression():
    """
    test depth_tune, regression rf (structure check)
    """
    n = 200

    X = np.random.uniform(size = (n, 510), low = -1,high = 1)
    y = 10 * np.sin(np.pi * X[:,0]*X[:,1]) + 20 * ( X[:,2] - .5)**2 +\
        10 * X[:,3] + 5 * X[:,4] + np.random.normal(size = n)

    rf_class = sklearn.ensemble.RandomForestRegressor(n_estimators = 2)
    random_forest = rf_class.fit(X = X,
                                 y = y.ravel())
    X_trained = X
    y_trained = y

    try:
        new_rf = smooth_rf.depth_tune(random_forest, X_trained, y_trained, verbose=False)
    except:
        assert False, \
            "Error in tuning regression rf with oob depth"

    assert type(new_rf) == type(random_forest), \
        "updated random forest should be same class as random forest put in"

    loss_vec = new_rf.loss_vec_depth

    _, max_depth = smooth_rf.calc_depth_for_forest(random_forest,verbose=False)

    assert loss_vec.shape[0] == np.int(max_depth), \
        "loss vector is incorrect dimension relative to maximum depth of rf"

def test_depth_tune_classification():
    """
    test depth_tune, classification rf (structure check)
    """
    n = 200
    min_size_leaf = 1

    X = np.random.uniform(size = (n, 510), low = -1,high = 1)
    y = 10 * np.sin(np.pi * X[:,0]*X[:,1]) + 20 * ( X[:,2] - .5)**2 +\
        10 * X[:,3] + 5 * X[:,4] + np.random.normal(size = n)

    y_cat = np.array(
                     pd.cut(y, bins = 5, labels = np.arange(5, dtype = np.int)),
                     dtype = np.int)

    y = y_cat

    num_classes = len(Counter(y_cat).keys())

    rf_class = sklearn.ensemble.RandomForestClassifier(n_estimators = 2,
                                            min_samples_leaf = min_size_leaf)
    random_forest = rf_class.fit(X = X,
                                 y = y.ravel())

    try:
        new_rf = smooth_rf.depth_tune(random_forest, X, y, verbose=False)
    except:
        assert False, \
            "Error in tuning regression rf with oob depth"

    assert type(new_rf) == type(random_forest), \
        "updated random forest should be same class as random forest put in"

    loss_vec = new_rf.loss_vec_depth

    _, max_depth = smooth_rf.calc_depth_for_forest(random_forest,verbose=False)

    assert loss_vec.shape[0] == np.int(max_depth), \
        "loss vector is incorrect dimension relative to maximum depth of rf"


def test_leaf_predicted_values():
    """
    test leaf_predicted_values (just on static example)
    """
    class inner_fake_tree():
        def __init__(self, nn, cl, cr, v):
            self.weighted_n_node_samples = nn
            self.children_left = cl
            self.children_right = cr
            self.value = v

    class fake_tree():
        def __init__(self, nn, cl, cr, v):
            self.tree_ = inner_fake_tree(nn, cl, cr, v)
            self.__class__ = sklearn.tree.tree.DecisionTreeRegressor

    weighted_n_node_samples = np.array([34,10,24,9,15,8,7], dtype = np.int)
    children_left = np.array([2,-1,4,-1,6,-1,-1], dtype = np.int)
    children_right = np.array([1,-1,3,-1,5,-1,-1], dtype = np.int)
    value = np.array([0, 1, 2, 3, 4, 5, 6]).reshape((-1,1,1))

    tree = fake_tree(weighted_n_node_samples,
                     children_left,
                     children_right,
                     value)

        #                    3 2 1 0
        #    |--1            1 1 1 0
        # -0-|
        #    |   |--3        3 3 2 0
        #    |-2-|
        #        |   |--5    5 4 2 0
        #        |-4-|
        #            |--6    6 4 2 0


    vals_expected = np.array([[0,0,0,0],
                              [1,2,2,2],
                              [1,3,4,4],
                              [1,3,5,6]])

    vals_predicted = smooth_rf.leaf_predicted_values(tree)

    for md, pred in vals_predicted.items():
        assert np.all(pred.ravel() ==\
                            vals_expected[np.int(md),:].ravel()),\
            "predicted values don't match desired values when pruning level %i" %max_depth_selected

