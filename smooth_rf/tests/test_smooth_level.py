import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import sklearn
import sys, os
import matplotlib.pyplot as plt
import scipy
import progressbar
import sparse
import scipy.sparse
from collections import Counter
import sklearn
import sklearn.ensemble
import copy
import quadprog
import scipy.sparse
import pdb

import smooth_rf

# private functions
from smooth_rf.smooth_level import _decision_list_nodes, _make_Vt_mat_tree

def test_make_Vt_mat():
    """
    tests for make_Vt_mat
    """
    X_train = np.concatenate(
        (np.random.normal(loc = (1,2), scale = .6, size = (100,2)),
        np.random.normal(loc = (-1.2, -.5), scale = .6, size = (100,2))),
    axis = 0)
    y_train = np.concatenate((np.zeros(100, dtype = np.int),
                             np.ones(100, dtype = np.int)))
    amount = np.int(200)
    s = 20
    c = y_train[:amount]
    # creating a random forest
    rf_class_known = sklearn.ensemble.RandomForestClassifier(
                                                        n_estimators = 100,
                                                        min_samples_leaf = 1)
    fit_rf_known = rf_class_known.fit(X = np.array(X_train)[:amount,:],
                                      y = y_train[:amount].ravel())
    random_forest = fit_rf_known

    data = np.array(X_train[:amount,:])

    Vt_dict = smooth_rf.make_Vt_mat(random_forest, data, verbose = False)

    assert len(Vt_dict) == 100, \
     "incorrect number of trees suggested in the full Vt output"

    for _ in range(10):
        r_idx = np.random.randint(100)
        random_Vt_dict = Vt_dict[r_idx]

        assert type(random_Vt_dict) == dict, \
         "output of _make_Vt_mat_tree is not a dictionary"

        assert np.all([x.shape[0] == data.shape[0] \
         for x in random_Vt_dict.values()]), \
         "output of _make_Vt_mat_tree elements have incorrect number of rows"

        assert np.sum([x.shape[1]  for x in random_Vt_dict.values()]) == \
         len(random_forest.estimators_[r_idx].tree_.children_left), \
         "output of split of Vt matrices have more columns than a" + \
         " full Vt mat would"


def test_make_Vt_mat_tree():
    """
    test for _make_Vt_mat_tree
    """
    X_train = np.concatenate(
        (np.random.normal(loc = (1,2), scale = .6, size = (100,2)),
        np.random.normal(loc = (-1.2, -.5), scale = .6, size = (100,2))),
    axis = 0)
    y_train = np.concatenate((np.zeros(100, dtype = np.int),
                             np.ones(100, dtype = np.int)))
    amount = np.int(200)
    s = 20
    c = y_train[:amount]
    # creating a random forest
    rf_class_known = sklearn.ensemble.RandomForestClassifier(
                                                        n_estimators = 1,
                                                        min_samples_leaf = 1)
    fit_rf_known = rf_class_known.fit(X = np.array(X_train)[:amount,:],
                                      y = y_train[:amount].ravel())
    forest = fit_rf_known.estimators_
    tree = forest[0]

    data = np.array(X_train[:amount,:])

    Vt_dict = _make_Vt_mat_tree(tree,data)

    assert type(Vt_dict) == dict, \
     "output of _make_Vt_mat_tree is not a dictionary"

    assert np.all([x.shape[0] == data.shape[0] for x in Vt_dict.values()]), \
     "output of _make_Vt_mat_tree elements have incorrect number of rows"

    assert np.sum([x.shape[1]  for x in Vt_dict.values()]) == \
     len(tree.tree_.children_left), \
     "output of split of Vt matrices have more columns than a full Vt mat would"



def test_remove_0_from_Ut_prime():
    """
    test of remove_0_from_Ut_prime
    """
    test_dict = {0: 0, 1: 1, 2: 2, 3: 4}
    test_dict_updated = smooth_rf.remove_0_from_Ut_prime(test_dict)

    assert np.all([test_dict[1] == test_dict_updated[0],
                   test_dict[2] == test_dict_updated[1],
                   test_dict[3] == test_dict_updated[2]]), \
     "objects in the dictionary don't get preserved correctly."

    assert len(test_dict_updated) == len(test_dict) - 1, \
     "incorrect length of returned dictionary"



def test_make_kernel():
    """
    test for make_kernel
    """
    X_train = np.concatenate(
        (np.random.normal(loc = (1,2), scale = .6, size = (100,2)),
        np.random.normal(loc = (-1.2, -.5), scale = .6, size = (100,2))),
    axis = 0)
    y_train = np.concatenate((np.zeros(100, dtype = np.int),
                             np.ones(100, dtype = np.int)))
    amount = np.int(200)
    # creating a random forest
    rf_class_known = sklearn.ensemble.RandomForestClassifier(
                                                        n_estimators = 1,
                                                        min_samples_leaf = 1)
    random_forest = rf_class_known.fit(X = np.array(X_train)[:amount,:],
                                      y = y_train[:amount].ravel())

    data = np.array(X_train[:amount,:])

    depth_dict, max_depth = smooth_rf.calc_depth_for_forest(random_forest,
                                                  verbose = False)

    Vt_dict = smooth_rf.make_Vt_mat(random_forest, data,
                                    depth_dict = depth_dict,
                                    verbose = False)

    Ut_prime_dict = smooth_rf.make_Ut_prime_mat_no_sym(Vt_dict, Vt_dict,
                                      max_depth = max_depth,
                                      verbose = False)

    Ut_prime_dict = smooth_rf.remove_0_from_Ut_prime(Ut_prime_dict)

    if len(Ut_prime_dict) > 0:
        K_mat = smooth_rf.make_kernel(Ut_prime_dict)

        assert K_mat.shape[0:2] == (K_mat.shape[0],K_mat.shape[0]), \
            "returned K matrix is not symmetric when the inputs were."

        assert len(K_mat.shape) == 2, \
            "returned K matrix is not 2d as expected."
    else:
        K_mat = smooth_rf.make_kernel(Ut_prime_dict)
        assert K_mat == 0, \
            "when you provide an empty Ut_prime_dict you should get a 0"



def test_categorical_depth_expand():
    """
    tests for categorical_depth_expand
    """
    D_mat = np.array([[1, 2, 1, 2],
                      [2, 1, 2, 1],
                      [1, 1, 3, 1],
                      [4, 1, 1, 1]])

    s_mat = smooth_rf.categorical_depth_expand(D_mat)

    assert s_mat.shape == (5,4,4), \
        "incorrect output dimensions for sparse matrix"

    for val in np.arange(5):
        s_mat_dense = s_mat.todense()
        assert np.all(1* (D_mat == val) == s_mat_dense[val,:,:]), \
            "incorrect compression of D_mat"



def test_depth_dist():
    """
    test for depth_dist function
    """
    X_train = np.concatenate(
        (np.random.normal(loc = (1,2), scale = .6, size = (100,2)),
        np.random.normal(loc = (-1.2, -.5), scale = .6, size = (100,2))),
    axis = 0)
    y_train = np.concatenate((np.zeros(100, dtype = np.int),
                             np.ones(100, dtype = np.int)))
    amount = np.int(200)
    # creating a random forest
    rf_class_known = sklearn.ensemble.RandomForestClassifier(
                                                        n_estimators = 1,
                                                        min_samples_leaf = 1)
    random_forest = rf_class_known.fit(X = np.array(X_train)[:amount,:],
                                      y = y_train[:amount].ravel())

    data = np.array(X_train[:amount,:])

    depth_dict, max_depth = smooth_rf.calc_depth_for_forest(random_forest,
                                                  verbose = False)

    Vt_dict = smooth_rf.make_Vt_mat(random_forest, data,
                                    depth_dict = depth_dict,
                                    verbose = False)

    Ut_prime_dict = smooth_rf.make_Ut_prime_mat_no_sym(Vt_dict, Vt_dict,
                                      max_depth = max_depth,
                                      verbose = False)

    Ut_prime_dict = smooth_rf.remove_0_from_Ut_prime(Ut_prime_dict)

    if len(Ut_prime_dict) > 0:
        K_mat = smooth_rf.make_kernel(Ut_prime_dict)

        DD_mat = smooth_rf.depth_dist(K_mat)

        assert K_mat.shape == DD_mat.shape, \
            "dimensions between K_mat and DD_mat should be the same"

        if type(DD_mat) is sparse.coo.core.COO:
            assert np.all(np.diag(DD_mat.todense()) == 0), \
                "diagonal should be naturally 0 (has error)"
        else:
            assert np.all(np.diag(DD_mat) == 0), \
                "diagonal should be naturally 0 (has error)"

        assert np.all(DD_mat >= 0), \
            "all entries should be positive in DD (has error)"
    else:
        K_mat = smooth_rf.make_kernel(Ut_prime_dict)
        assert K_mat == 0, \
            "when you provide an empty Ut_prime_dict you should get a 0"



def test_decision_list_nodes():
    """
    test for _decision_list_nodes

    just checks dimension
    """

    c_r = np.array([1,3,-1,-1,-1])
    c_l = np.array([2,4,-1,-1,-1])

    a, b = _decision_list_nodes(c_r, c_l)

    assert len(a) == len(b), \
        "length of lists should be the same length"


def test_decision_path_nodes():
    """
    test for decision_path_nodes
    """

    # static test
    c_r = np.array([1,3,-1,-1,-1])
    c_l = np.array([2,4,-1,-1,-1])

    x = smooth_rf.decision_path_nodes(c_r, c_l)

    if type(x) is sparse.coo.core.COO:
        x = x.todense()

    assert np.all(
        x == np.array([[1,0,1,0,0],  #node 2 is just child of 0
                       [1,1,0,1,0],  #node 3 child of 0,1
                       [1,1,0,0,1]]) #node 4 child of 0,1
                                ), \
        "produces incorrect decision path for small test"


    assert x.shape == (3, 5),\
        "decision_path() returns incorrect shape array, " + \
        "for lit static test"

    # values of array on only 0,1
    assert np.all(
        [y in np.array([0.0,1.0]) \
            for y in np.array(list(dict(Counter(x.ravel())).keys()))] ), \
        "values of decision_path() are not only 0 or 1, :("

