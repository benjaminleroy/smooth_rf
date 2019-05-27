import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import sklearn
import sys, os
import matplotlib.pyplot as plt
import scipy
import scipy.linalg
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



def test_max_depth_dist():
    """
    test of max_depth_dist
    """
    X_train = np.concatenate(
        (np.random.normal(loc = (1,2), scale = .6, size = (100,2)),
        np.random.normal(loc = (-.2, 1), scale = .6, size = (100,2))),
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

        DD_mat = smooth_rf.max_depth_dist(K_mat)

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

        # new checks:

        assert np.all(DD_mat == DD_mat), \
            "DD_mat should be symmetric"

        DD_base_mat = smooth_rf.depth_dist(K_mat)

        assert np.all(DD_mat >= DD_base_mat), \
            "DD_mat_max should always be >= DD_mat_basic"

        assert np.all(DD_mat[DD_mat != DD_base_mat] == \
                     DD_base_mat.T[DD_mat != DD_base_mat]), \
            "DD_mat_max should take the max of DD_bases value " +\
            "across ij and ji combos"

    else:
        K_mat = smooth_rf.make_kernel(Ut_prime_dict)
        assert K_mat == 0, \
            "when you provide an empty Ut_prime_dict you should get a 0"



def test_min_depth_dist():
    """
    test of min_depth_dist
    """
    X_train = np.concatenate(
        (np.random.normal(loc = (1,2), scale = .6, size = (100,2)),
        np.random.normal(loc = (-.2, 1), scale = .6, size = (100,2))),
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

        DD_mat = smooth_rf.min_depth_dist(K_mat)

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

        # new checks:

        assert np.all(DD_mat == DD_mat), \
            "DD_mat should be symmetric"

        DD_base_mat = smooth_rf.depth_dist(K_mat)

        assert np.all(DD_mat <= DD_base_mat), \
            "DD_mat_min should always be <= DD_mat_basic"

        assert np.all(DD_mat[DD_mat != DD_base_mat] == \
                     DD_base_mat.T[DD_mat != DD_base_mat]), \
            "DD_mat_max should take the min of DD_bases value " +\
            "across ij and ji combos"

    else:
        K_mat = smooth_rf.make_kernel(Ut_prime_dict)
        assert K_mat == 0, \
            "when you provide an empty Ut_prime_dict you should get a 0"

def test_is_pos_def():
    """
    test for is_pos_def function
    """

    A = np.arange(16).reshape(4,4)
    assert not smooth_rf.is_pos_def(A), \
        "Error seeing a non-symetric matrix is not PSD"

    A = np.array([[1,0],
                 [0,1]])
    assert smooth_rf.is_pos_def(A), \
        "identity matrix should bee seen as PSD"

    A = np.array([[2,-1,0],
                 [-1,2,-1],
                 [0,-1,2]])
    assert smooth_rf.is_pos_def(A), \
        "example from wikipedia should be PSD"

    A = np.array([[1,2],
                 [2,1]])
    assert not smooth_rf.is_pos_def(A), \
        "thinks a matrix with all postive values is PSD when it is not"

    A = np.array([[4,9],
                 [1,4]])
    assert not smooth_rf.is_pos_def(A), \
        "thinks a matrix with positive eigenvalues but not symmetric is PSD when it is not"

def test_process_tuning_leaf_attributes_tree():
    """
    test process_tuning_leaf_attributes_tree
    """
    for _ in range(10):
        X_trained = np.concatenate(
            (np.random.normal(loc = (1,2), scale = .6, size = (100,2)),
            np.random.normal(loc = (-1.2, -.5), scale = .6, size = (100,2))),
            axis = 0)
        y_trained = np.concatenate((np.zeros(100, dtype = np.int),
                             np.ones(100, dtype = np.int))) + 100

        X_tuned = np.concatenate(
            (np.random.normal(loc = (1,2), scale = .6, size = (50,2)),
            np.random.normal(loc = (-1.2, -.5), scale = .6, size = (50,2))),
            axis = 0)
        y_tuned = np.concatenate((np.zeros(50, dtype = np.int),
                             np.ones(50, dtype = np.int))) + 100


        amount = np.int(200)
        # creating a random forest
        rf_reg = sklearn.ensemble.RandomForestRegressor(n_estimators = 1,
                                                        min_samples_leaf = 1)
        fit_reg = rf_reg.fit(X = np.array(X_trained)[:amount,:],
                                      y = y_trained[:amount].ravel())
        forest = fit_reg.estimators_

        t = forest[0]

        a, b = smooth_rf.process_tuning_leaf_attributes_tree(t, eps = -1,
                                            X_tune = X_tuned,
                                            y_tune = y_tuned)

        assert np.all(b[a == -1] == 0), \
            "leaves with no elements from tuning data should also have "+\
            "y predicted value of 0"

        assert np.sum([a[a != -1]]) == 1, \
            "weights should be scaled to sum to 1 (excluding those values "+\
            "with eps value inserted)"

        assert np.sum(a[a != -1] * b[a!=-1]) == 100.5, \
            "should preserve correct data structure from tuning data"

        assert a.shape[0] == np.sum(t.tree_.children_left == -1) and \
            a.shape[0] == b.shape[0], \
            "a and b should have the same length as the number of tree leaves"

def test_update_til_psd():
    """
    test update_til_psd
    """
    A = np.arange(16).reshape(4,4)
    expection_returned = False
    try:
        smooth_rf.update_til_psd(A)
    except:
        expection_returned = True

    assert expection_returned, \
        "Should raise error if input is not symmetric"

    A = np.array([[1,0],
                 [0,1]])
    assert np.array_equal(smooth_rf.update_til_psd(A), A), \
        "identity matrix should already be PSD"

    A = np.array([[2,-1,0],
                 [-1,2,-1],
                 [0,-1,2]])
    assert np.array_equal(smooth_rf.update_til_psd(A), A), \
        "example from wikipedia should already be PSD"

    A = np.array([[1,2],
                 [2,1]])

    updated_A = smooth_rf.update_til_psd(A, verbose=False)

    assert not np.array_equal(A,updated_A), \
        "Matrix with all positive values but not PSD should need update"

    diff = np.diag(updated_A - A)
    assert np.array_equal(updated_A - np.diag(diff), A), \
        "updated matrix should only be updated from a diagonal matrix"

    assert np.all(diff == diff[0]), \
        "updated diagonal entries should have the same value"

    steps = np.round(np.log(diff[0] / np.finfo(float).eps)/np.log(2), 1 )

    assert np.int(steps) == steps, \
        "should have integer in added 2^K* eps value - weak check"



def test_smooth_all_regressor():
    """
    test for smooth_all - regressor, only runs on example dataset,
    checks for errs (takes a bit)
    """
    # easy to seperate example
    X_trained = np.concatenate(
        (np.random.normal(loc = (1,2), scale = .6, size = (100,2)),
        np.random.normal(loc = (-1.2, -.5), scale = .6, size = (100,2))),
        axis = 0)
    y_trained = np.concatenate((np.zeros(100, dtype = np.int),
                         np.ones(100, dtype = np.int))) + 100
    amount = np.int(200)
    # creating a random forest
    rf_reg = sklearn.ensemble.RandomForestRegressor(
                                                    n_estimators = 5,
                                                    min_samples_leaf = 1)
    fit_reg = rf_reg.fit(X = np.array(X_trained)[:amount,:],
                                  y = y_trained[:amount].ravel())
    forest = fit_reg.estimators_

    random_forest = fit_reg

    # general check for erroring
    try:
        updated_rf = smooth_rf.smooth_all(random_forest, X_trained, y_trained,
                                    parents_all=True, verbose=False,
                                    no_constraint=True,
                                    sanity_check={'sanity check':False,
                                                  'tol_pow':None})
    except:
        assert False, \
            "error running smoothing_function for a random forest regressor"


    # general check for erroring, no constraint
    try:
        updated_rf = smooth_rf.smooth_all(random_forest, X_trained, y_trained,
                                    parents_all=True, verbose=False,
                                    no_constraint=False,
                                    sanity_check={'sanity check':False,
                                                  'tol_pow':None})
    except:
        assert False, \
            "error running smoothing_function for a random forest regressor"
    assert np.isclose(np.sum(updated_rf.lamb), 1, rtol =1e-8),\
        "lambda should be constrained to sum to 1 if 'no_constraint=False'"


    # sanity check
    a = smooth_rf.smooth_all(random_forest, X_trained, y_trained,
                             parents_all=True, verbose=False,
                             sanity_check=True)

    no_update_pred = a.predict(X_trained)
    base_pred = random_forest.predict(X_trained)

    assert np.all(no_update_pred == base_pred), \
        "sanity check for rf regressor in smoother failed"


    # harder to seperate example

    X_trained = np.concatenate(
        (np.random.normal(loc = (1,2), scale = .6, size = (200,2)),
        np.random.normal(loc = (.5,2), scale = .6, size = (200,2))),
        axis = 0)
    y_trained = np.concatenate((np.zeros(200, dtype = np.int),
                         np.ones(200, dtype = np.int))) + 100
    amount = np.int(400)
    # creating a random forest
    rf_reg = sklearn.ensemble.RandomForestRegressor(
                                                    n_estimators = 10,
                                                    min_samples_leaf = 1)
    fit_reg = rf_reg.fit(X = np.array(X_trained)[:amount,:],
                                  y = y_trained[:amount].ravel())
    forest = fit_reg.estimators_

    random_forest = fit_reg

    # general check for erroring
    fail_count = 0
    for tol_pow in [None, 10, 20]:
        early_fail_count = fail_count
        try:
            updated_rf = smooth_rf.smooth_all(random_forest, X_trained, y_trained,
                                        parents_all=False, verbose=False,
                                        no_constraint=True,
                                        sanity_check={'sanity check':False,
                                                      'tol_pow':None})
        except:
            fail_count += 1
        if early_fail_count == fail_count:
            pass
    assert fail_count < 3, \
        "error running smoothing_function for a random forest regressor " +\
        "across all tol_pow options"


    # general check for erroring, no constraint
    fail_count = 0
    for tol_pow in [None, 10, 20]:
        early_fail_count = fail_count
        try:
            updated_rf = smooth_rf.smooth_all(random_forest, X_trained, y_trained,
                                        parents_all=False, verbose=False,
                                        no_constraint=False,
                                        sanity_check={'sanity check':False,
                                                      'tol_pow':10})
        except:
            fail_count += 1
        if early_fail_count == fail_count:
            pass

    assert fail_count < 3, \
        "error running smoothing_function for a random forest regressor "+\
        "(with constraint)" +\
        "across all tol_pow options"

    assert np.isclose(np.sum(updated_rf.lamb), 1, rtol =1e-8),\
        "lambda should be constrained to sum to 1 if 'no_constraint=False'"


    # sanity check
    a = smooth_rf.smooth_all(random_forest, X_trained, y_trained,
                             parents_all=False, verbose=False,
                             sanity_check={'sanity check':True,
                                           'tol_pow':20})

    no_update_pred = a.predict(X_trained)
    base_pred = random_forest.predict(X_trained)

    assert np.all(no_update_pred == base_pred), \
        "sanity check for rf regressor in smoother failed"


def test_check_in_null():
    """
    test check_in_null
    """

    # in the null (basic):
    G = np.array([[2,3,5],
                 [-4,2,3]])
    v = np.array([-1/16, -13/8,1])

    assert smooth_rf.check_in_null(G,v), \
        "wikipedia's first example should see v being in the null space of G"

    # in the null (not basic):
    G = np.array([[1, 0 ,-3, 0, 2, -8],
                  [0, 1, 5, 0, -1, 4],
                  [0, 0, 0, 1, 7, -9],
                  [0, 0, 0, 0, 0, 0]
                 ])

    v1 = np.array([3,-5,1,0,0,0])
    v2 = np.array([-2,1,0,-7,1,0])
    v3 = np.array([8,-4,0,9,0,1])

    for _ in range(10):
        v_new = np.random.uniform(size = 1) * v1 +\
                    np.random.uniform(size = 1) * v2 +\
                    np.random.uniform(size = 1) * v3
        assert smooth_rf.check_in_null(G, v_new),\
            "wikipedia's second example should see any combo of the 3 v "+\
            "vectors belong to the null space of G"

    # not in the null (sup basic):
    G = np.array([[1,0],
                 [0,1]])
    for _ in range(10):
        v = np.random.uniform(size = 1)
        assert not smooth_rf.check_in_null(G,v),\
            "There is no null space for the identity matrix - so nothing "+\
            "should be in the null"

    # not in the null (basic):
    G = np.array([[2,3,5],
                 [-4,2,3]])
    v = np.array([-2, 5,8])
    assert not smooth_rf.check_in_null(G,v),\
        "A basic linear combination of the rows should also be in the span of G"

    # not in null
    v = np.array([-2, 5,8]) + np.array([-1/16, -13/8,1])
    assert not smooth_rf.check_in_null(G,v),\
        "A combination of all vectors (including 1 from null should not be "+\
        " the null space of G"

