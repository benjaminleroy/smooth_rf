import numpy as np
import pandas as pd
import scipy.sparse
import sparse
import sklearn
from sklearn.ensemble import RandomForestRegressor
from collections import Counter
import sys, os
import nose
import smooth_rf

def test_depth_per_node():
    """
    test depth_per_node on random forest tree
    Tests for:
    1) depth_per_node function makes sure all children are 1 (and only 1) level
        deeper
    """

    # data creation
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

    predicted_depth = smooth_rf.depth_per_node(tree)

    c_left  = tree.tree_.children_left
    c_right = tree.tree_.children_right
    left_minus_1  = predicted_depth[c_left != -1] - \
                        predicted_depth[c_left][c_left != -1]
    right_minus_1 = predicted_depth[c_right != -1] - \
                        predicted_depth[c_right][c_right != -1]
    assert np.all([np.all(left_minus_1 == -1), np.all(right_minus_1 == -1)]), \
        "parent - children depth != -1 (which it should)"

    unique_values = np.array(list(dict(Counter(predicted_depth)).keys()))
    unique_values.sort()

    assert np.all(unique_values == np.arange(len(unique_values))), \
        "jump in depth between at least one parent and child is more than 1"



def test_calc_depth_for_forest():
    """
    test calc_depth_for_forest on random forest
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
    forest = fit_rf_known.estimators_

    dict_depth, max_depth = smooth_rf.calc_depth_for_forest(fit_rf_known,
                                                         verbose = False)
    assert type(dict_depth) == dict, \
        "first output is not a dictionary"

    for _ in np.arange(10):
        idx = np.random.randint(100)
        tree = forest[idx]
        depth_vec = smooth_rf.depth_per_node(tree)
        assert np.all(depth_vec == dict_depth[idx]), \
            "dictionary does not match depth_per_node function"

        assert np.all(depth_vec <= max_depth), \
            "maximum depth is calculated wrong"

def test_create_decision_per_leafs():
    """
    test for create_decision_per_leafs

    Both static and random tests (random tests are more relative to structure
    than exact answers)
    """
    # data creation
    n = 200
    min_size_leaf = 1

    X = np.random.uniform(size = (n, 510), low = -1,high = 1)
    y = 10 * np.sin(np.pi * X[:,0]*X[:,1]) + 20 * ( X[:,2] - .5)**2 +\
        10 * X[:,3] + 5 * X[:,4] + np.random.normal(size = n)

    rf_class = sklearn.ensemble.RandomForestRegressor(n_estimators = 2,
                                            min_samples_leaf = min_size_leaf)
    random_forest = rf_class.fit(X = X,
                                 y = y.ravel())

    tree = random_forest.estimators_[0]

    v_leaf, v_all = smooth_rf.create_decision_per_leafs(tree)

    assert v_all.shape[0] == v_all.shape[1] and \
           v_all.shape[1] == v_leaf.shape[1], \
        "number of nodes in tree not preserved in output matrices shapes"

    assert v_leaf.shape[0] == \
            np.sum(tree.tree_.children_right == -1), \
        "number of leaves doesn't matrix ouput matrix shape (v_leaf)"


    # static check

    # tree structure:
    # ~upper: left, lower: right~
    #    |--1
    # -0-|
    #    |   |--3
    #    |-2-|
    #        |   |--5
    #        |-4-|
    #            |--6

    # creating desired structure
    class inner_fake_tree():
        def __init__(self, cl, cr):
            self.children_left = cl
            self.children_right = cr

    class fake_tree():
        def __init__(self, cl, cr):
            self.tree_ = inner_fake_tree(cl, cr)
            self.__class__ = sklearn.tree.tree.DecisionTreeRegressor

    children_left = np.array([2,-1,4,-1,6,-1,-1], dtype = np.int)
    children_right = np.array([1,-1,3,-1,5,-1,-1], dtype = np.int)

    test = fake_tree(children_left,children_right)


    v_all_should = np.array([[1,0,0,0,0,0,0],
                             [1,1,0,0,0,0,0],
                             [1,0,1,0,0,0,0],
                             [1,0,1,1,0,0,0],
                             [1,0,1,0,1,0,0],
                             [1,0,1,0,1,1,0],
                             [1,0,1,0,1,0,1]],dtype = np.int)

    v_leaf_should = v_all_should[test.tree_.children_left == -1,:]

    v_leaf_static, v_all_static = smooth_rf.create_decision_per_leafs(test)

    if type(v_all_static) == scipy.sparse.coo.coo_matrix:
        v_all_static = v_all_static.todense()
    if type(v_leaf_static) == scipy.sparse.coo.coo_matrix:
        v_leaf_static = v_leaf_static.todense()

    assert np.all(v_all_should == v_all_static), \
        "static test failed to reproduce correct solutions, (v_all)"
    assert np.all(v_leaf_should == v_leaf_static), \
        "static test failed to reproduce correct solutions, (v_leaf)"


def test_create_distance_mat_leaves():
    """
    test for test_create_distance_mat_leaves (depth based)

    Both static and random tests (random tests are more relative to structure
    than exact answers)

    Note this test examines the "standard" distance, and "min" and "max"
    distances
    """
    # data creation
    n = 200
    min_size_leaf = 1

    X = np.random.uniform(size = (n, 510), low = -1, high = 1)
    y = 10 * np.sin(np.pi * X[:,0]*X[:,1]) + 20 * ( X[:,2] - .5)**2 +\
        10 * X[:,3] + 5 * X[:,4] + np.random.normal(size = n)

    rf_class = sklearn.ensemble.RandomForestRegressor(n_estimators = 2,
                                            min_samples_leaf = min_size_leaf)
    random_forest = rf_class.fit(X = X,
                                 y = y.ravel())

    tree = random_forest.estimators_[0]

    v_leaf, v_all = smooth_rf.create_decision_per_leafs(tree)

    for style in ["standard","min","max"]:
        d1, _ = smooth_rf.create_distance_mat_leaves(tree, style = style)
        d2, _ = smooth_rf.create_distance_mat_leaves(decision_mat_leaves = v_leaf,
                                                  style = style)

        assert d2.shape == d1.shape, \
            "distance matrix shape differences from creation with same structure"

        assert d1.shape[0] == np.sum(tree.tree_.children_left == -1) and \
            d1.shape[0] == d1.shape[1], \
            "distance matrix correct shape relave to number of leaves"

        assert np.all(d1 == d2), \
            "distance matrix differs from creation with same structure..."

    # static check

    # tree structure:
    # ~upper: left, lower: right~
    #    |--1
    # -0-|
    #    |   |--3
    #    |-2-|
    #        |   |--5
    #        |-4-|
    #            |--6

    d1_should = list()

    # distance (standard)
    # (1) 0 | 1 | 1 | 1
    # (3) 2 | 0 | 1 | 1
    # (5) 3 | 2 | 0 | 1
    # (6) 3 | 2 | 1 | 0

    d1_should.append(np.array([[0,1,1,1],
                               [2,0,1,1],
                               [3,2,0,1],
                               [3,2,1,0]], dtype = np.int))
    # distance (max)
    # (1) 0 | 2 | 3 | 3
    # (3) 2 | 0 | 2 | 2
    # (5) 3 | 2 | 0 | 1
    # (6) 3 | 2 | 1 | 0

    d1_should.append(np.array([[0,2,3,3],
                               [2,0,2,2],
                               [3,2,0,1],
                               [3,2,1,0]], dtype = np.int))
    # distance (min)
    # (1) 0 | 1 | 1 | 1
    # (3) 1 | 0 | 1 | 1
    # (5) 1 | 1 | 0 | 1
    # (6) 1 | 1 | 1 | 0

    d1_should.append(np.array([[0,1,1,1],
                               [1,0,1,1],
                               [1,1,0,1],
                               [1,1,1,0]], dtype = np.int))

    # creating desired structure
    class inner_fake_tree():
        def __init__(self, cl, cr):
            self.children_left = cl
            self.children_right = cr

    class fake_tree():
        def __init__(self, cl, cr):
            self.tree_ = inner_fake_tree(cl, cr)
            self.__class__ = sklearn.tree.tree.DecisionTreeRegressor

    children_left = np.array([2,-1,4,-1,6,-1,-1], dtype = np.int)
    children_right = np.array([1,-1,3,-1,5,-1,-1], dtype = np.int)

    test = fake_tree(children_left,children_right)

    for d_idx, style in enumerate(["standard","max","min"]):
        d1, _ = smooth_rf.create_distance_mat_leaves(test, style = style)

        assert np.all(d1 == d1_should[d_idx]), \
            "static test failed to reproduce correct solutions"


def test_create_distance_mat_leaves_impurity():
    """
    test for test_create_distance_mat_leaves for impurity

    Both static and random tests (random tests are more relative to structure
    than exact answers)

    Note this test examines the "standard" distance, and "min" and "max"
    distances
    """
    # data creation
    n = 200
    min_size_leaf = 1

    X = np.random.uniform(size = (n, 510), low = -1, high = 1)
    y = 10 * np.sin(np.pi * X[:,0]*X[:,1]) + 20 * ( X[:,2] - .5)**2 +\
        10 * X[:,3] + 5 * X[:,4] + np.random.normal(size = n)

    rf_class = sklearn.ensemble.RandomForestRegressor(n_estimators = 2,
                                            min_samples_leaf = min_size_leaf)
    random_forest = rf_class.fit(X = X,
                                 y = y.ravel())

    tree = random_forest.estimators_[0]

    v_leaf, v_all = smooth_rf.create_decision_per_leafs(tree)
    impurity_diff = smooth_rf.change_in_impurity(tree)

    for levels in np.random.choice(np.arange(3,20), size = 5, replace = False):
        for style in ["standard","min","max"]:
            d1, _ = smooth_rf.create_distance_mat_leaves(tree,
                    style=style,
                    distance_style="impurity",
                    levels = levels)
            d2, _ = smooth_rf.create_distance_mat_leaves(
                    decision_mat_leaves=v_leaf,
                    change_in_impurity_vec=impurity_diff,
                    style=style,
                    distance_style="impurity",
                    levels=levels)

            assert d2.shape == d1.shape, \
                "distance matrix shape differences from creation with same structure"

            assert d1.shape[0] == np.sum(tree.tree_.children_left == -1) and \
                d1.shape[0] == d1.shape[1], \
                "distance matrix correct shape relave to number of leaves"

            assert np.all(d1 == d2), \
                "distance matrix differs from creation with same structure..."

    # static check

    # tree structure:
    # ~upper: left, lower: right~ | impurity
    #    |--1                     | .2
    # -0-|                        | .7
    #    |   |--3                 | .2
    #    |-2-|                    | .3
    #        |   |--5             | 0
    #        |-4-|                | .1
    #            |--6             | 0

    d1_should = list()

    # distance (standard)
    # (1)  0 | .5 | .5 | .5
    # (3) .5 |  0 | .1 | .1
    # (5) .7 | .3 |  0 | .1
    # (6) .7 | .3 | .1 |  0

    d1_should.append(np.array([[0 ,.5,.5,.5],
                               [.5,0 ,.1,.1],
                               [.7,.3,0 ,.1],
                               [.7,.3,.1,0 ]]))
    # distance (max)
    # (1)  0 | .5 | .7 | .7
    # (3) .5 |  0 | .3 | .3
    # (5) .7 | .3 |  0 | .1
    # (6) .7 | .3 | .1 |  0

    d1_should.append(np.array([[0 ,.5,.7,.7],
                               [.5,0 ,.3,.3],
                               [.7,.3,0 ,.1],
                               [.7,.3,.1,0 ]]))
    # distance (min)
    # (1)  0 | .5 | .5 | .5
    # (3) .5 |  0 | .1 | .1
    # (5) .5 | .1 |  0 | .1
    # (6) .5 | .1 | .1 |  0

    d1_should.append(np.array([[0 ,.5,.5,.5],
                               [.5,0 ,.1,.1],
                               [.5,.1,0 ,.1],
                               [.5,.1,.1,0 ]]))

    # creating desired structure
    class inner_fake_tree():
        def __init__(self, cl, cr, impurity):
            self.children_left = cl
            self.children_right = cr
            self.impurity = impurity

    class fake_tree():
        def __init__(self, cl, cr, impurity):
            self.tree_ = inner_fake_tree(cl, cr, impurity)
            self.__class__ = sklearn.tree.tree.DecisionTreeRegressor

    children_left = np.array([2,-1,4,-1,6,-1,-1], dtype = np.int)
    children_right = np.array([1,-1,3,-1,5,-1,-1], dtype = np.int)

    impurity = np.array([.7, .2, .3, .2, .1, 0, 0])

    test = fake_tree(children_left,children_right,impurity)

    for d_idx, style in enumerate(["standard","max","min"]):
        d1, _ = smooth_rf.create_distance_mat_leaves(test,
                                        style = style,
                                        distance_style = "impurity",
                                        levels = None)
        if type(d1) is scipy.sparse.coo.coo_matrix or \
            type(d1) is scipy.sparse.csr.csr_matrix:
            d1 = d1.todense()
        assert np.allclose(d1,d1_should[d_idx]), \
            "static test failed to reproduce correct solutions "+\
            "(when levels = None)"

def test_create_Gamma_eta_tree_regression():
    """
    test for create_Gamma_eta_tree, regression tree - standard depth only

    Both static and random tests (random tests are more relative to structure
    than exact answers)
    """


    # random - structure output check
    # data creation
    n = 200
    min_size_leaf = 1

    X = np.random.uniform(size = (n, 510), low = -1,high = 1)
    y = 10 * np.sin(np.pi * X[:,0]*X[:,1]) + 20 * ( X[:,2] - .5)**2 +\
        10 * X[:,3] + 5 * X[:,4] + np.random.normal(size = n)

    rf_class = sklearn.ensemble.RandomForestRegressor(n_estimators = 2,
                                            min_samples_leaf = min_size_leaf)
    random_forest = rf_class.fit(X = X,
                                 y = y.ravel())

    tree = random_forest.estimators_[0]

    max_depth_range = np.max(smooth_rf.depth_per_node(tree)) + 1

    G, n = smooth_rf.create_Gamma_eta_tree(tree)

    assert G.shape == (np.sum(tree.tree_.children_left == -1),
                       max_depth_range), \
        "Gamma returned does not have the correct shape"

    assert n.shape ==  G.shape, \
        "eta returned does not have the correct shape"

    assert np.all(n >= 0), \
        "eta returned has negative values"

    assert np.all(n[:,0] ==
        tree.tree_.weighted_n_node_samples[tree.tree_.children_left == -1]),\
        "eta structure doesn't match up with number of observes per leaf"

    # static check

    # tree structure:
    # ~upper: left, lower: right~
    #                       num obs
    #    |--1                   10
    # -0-|                       34
    #    |   |--3              9
    #    |-2-|                  24
    #        |   |--5         8
    #        |-4-|             15
    #            |--6         7


    # eta
    # (1) 10 | 24 | 0  | 0
    # (3) 9  | 15 | 10 | 0
    # (5) 8  | 7  | 9  | 10
    # (6) 7  | 8  | 9  | 10

    # Gamma
    # (1) 10         | 18+24+28 = 70 | 0  | 0
    # (3) 9 * 2 = 18 | 24+28 = 52    | 10 | 0
    # (5) 8 * 3 = 24 | 28            | 18 | 10
    # (6) 7 * 4 = 28 | 24            | 18 | 10


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
    value = np.array([-99, 1, -99, 2, -99, 3, 4]).reshape((-1,1,1))

    test = fake_tree(weighted_n_node_samples,
                     children_left,
                     children_right,
                     value)

    n_leaf = 4

    g_static, n_static = smooth_rf.create_Gamma_eta_tree(test)

    n_expected = np.array([[10,24,0,0],
                           [9,15,10,0],
                           [8,7,9,10],
                           [7,8,9,10]])
    g_expected = np.array([[10,70,0,0],
                           [18,52,10,0],
                           [24,28,18,10],
                           [28,24,18,10]])
    assert np.all(g_static == g_expected), \
        "static test's Gamma failed to reproduce correct solutions"
    assert np.all(n_static == n_expected), \
        "static test's eta failed to reproduce correct solutions"

def test_create_Gamma_eta_tree_regression_impurity():
    """
    test for create_Gamma_eta_tree, regression tree - standard, impurity only

    Both static and random tests (random tests are more relative to structure
    than exact answers)
    """


    # random - structure output check
    # data creation
    n = 200
    min_size_leaf = 1

    X = np.random.uniform(size = (n, 510), low = -1,high = 1)
    y = 10 * np.sin(np.pi * X[:,0]*X[:,1]) + 20 * ( X[:,2] - .5)**2 +\
        10 * X[:,3] + 5 * X[:,4] + np.random.normal(size = n)

    rf_class = sklearn.ensemble.RandomForestRegressor(n_estimators = 2,
                                            min_samples_leaf = min_size_leaf)
    random_forest = rf_class.fit(X = X,
                                 y = y.ravel())

    tree = random_forest.estimators_[0]

    max_depth_range = np.max(smooth_rf.depth_per_node(tree)) + 1

    G, n = smooth_rf.create_Gamma_eta_tree(tree)

    assert G.shape == (np.sum(tree.tree_.children_left == -1),
                       max_depth_range), \
        "Gamma returned does not have the correct shape"

    assert n.shape ==  G.shape, \
        "eta returned does not have the correct shape"

    assert np.all(n >= 0), \
        "eta returned has negative values"

    assert np.all(n[:,0] ==
        tree.tree_.weighted_n_node_samples[tree.tree_.children_left == -1]),\
        "eta structure doesn't match up with number of observes per leaf"

    # static check

    # tree structure:
    # ~upper: left, lower: right~
    #                       num obs
    #    |--1                   10
    # -0-|                       34
    #    |   |--3              9
    #    |-2-|                  24
    #        |   |--5         8
    #        |-4-|             15
    #            |--6         7


    # eta
    # (1) 10 | 24 | 0  | 0
    # (3) 9  | 15 | 10 | 0
    # (5) 8  | 7  | 9  | 10
    # (6) 7  | 8  | 9  | 10

    # Gamma
    # (1) 10         | 18+24+28 = 70 | 0  | 0
    # (3) 9 * 2 = 18 | 24+28 = 52    | 10 | 0
    # (5) 8 * 3 = 24 | 28            | 18 | 10
    # (6) 7 * 4 = 28 | 24            | 18 | 10


    class inner_fake_tree():
        def __init__(self, nn, cl, cr, v, impurity):
            self.weighted_n_node_samples = nn
            self.children_left = cl
            self.children_right = cr
            self.value = v
            self.impurity = impurity

    class fake_tree():
        def __init__(self, nn, cl, cr, v, impurity):
            self.tree_ = inner_fake_tree(nn, cl, cr, v, impurity)
            self.__class__ = sklearn.tree.tree.DecisionTreeRegressor

    weighted_n_node_samples = np.array([34,10,24,9,15,8,7], dtype = np.int)
    children_left = np.array([2,-1,4,-1,6,-1,-1], dtype = np.int)
    children_right = np.array([1,-1,3,-1,5,-1,-1], dtype = np.int)
    value = np.array([-99, 1, -99, 2, -99, 3, 4]).reshape((-1,1,1))
    impurity = np.array([4, 3, 3, 2,2,1,1])


    test = fake_tree(weighted_n_node_samples,
                     children_left,
                     children_right,
                     value,
                     impurity)

    n_leaf = 4

    # inner calulation of levels

    g_static, n_static = smooth_rf.create_Gamma_eta_tree(test,
                                            distance_style = "impurity",
                                            levels = 5)
    # ^ levels =5 gives us 4 levels in the end (which is desirable)
    n_expected = np.array([[10,24,0,0],
                           [9,15,10,0],
                           [8,7,9,10],
                           [7,8,9,10]])
    g_expected = np.array([[10,70,0,0],
                           [18,52,10,0],
                           [24,28,18,10],
                           [28,24,18,10]])
    assert np.all(g_static == g_expected), \
        "static test's Gamma failed to reproduce correct solutions "+\
        "(levels internally created)"
    assert np.all(n_static == n_expected), \
        "static test's eta failed to reproduce correct solutions "+\
        "(levels internally created)"


    # external calulation of levels
    levels = np.array([0,1,2,3,4])

    g_static, n_static = smooth_rf.create_Gamma_eta_tree(test,
                                            distance_style = "impurity",
                                            levels = levels)
    # ^ levels =5 gives us 4 levels in the end (which is desirable)
    n_expected = np.array([[10,24,0,0],
                           [9,15,10,0],
                           [8,7,9,10],
                           [7,8,9,10]])
    g_expected = np.array([[10,70,0,0],
                           [18,52,10,0],
                           [24,28,18,10],
                           [28,24,18,10]])
    assert np.all(g_static == g_expected), \
        "static test's Gamma failed to reproduce correct solutions "+\
        "(levels externally provided)"
    assert np.all(n_static == n_expected), \
        "static test's eta failed to reproduce correct solutions "+\
        "(levels externally provided)"

def test_create_Gamma_eta_tree_classification():
    """
    test for create_Gamma_eta_tree, classification tree - standard depth only

    Both static and random tests (random tests are more relative to structure
    than exact answers)
    """


    # random - structure output check
    # data creation
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

    tree = random_forest.estimators_[0]

    max_depth_range = np.max(smooth_rf.depth_per_node(tree)) + 1

    G, n = smooth_rf.create_Gamma_eta_tree(tree)

    assert G.shape == (num_classes,
                       np.sum(tree.tree_.children_left == -1),
                       max_depth_range), \
        "Gamma returned does not have the correct shape"

    assert n.shape ==  G.shape[1:3], \
        "eta returned does not have the correct shape"

    assert np.all(n >= 0), \
        "eta returned has negative values"

    assert np.all(n[:,0] ==
        tree.tree_.weighted_n_node_samples[tree.tree_.children_left == -1]),\
        "eta structure doesn't match up with number of observes per leaf"

    # static check

    # tree structure:
    # ~upper: left, lower: right~
    #                       num obs     class 1   class 2
    #    |--1                   10          5       5
    # -0-|                       34         21      13
    #    |   |--3              9            9       0
    #    |-2-|                  24          16      8
    #        |   |--5         8             7       1
    #        |-4-|             15           7       8
    #            |--6         7             0       7


    # eta
    # (1) 10 | 24 | 0  | 0
    # (3) 9  | 15 | 10 | 0
    # (5) 8  | 7  | 9  | 10
    # (6) 7  | 8  | 9  | 10


    # Gamma (class 1)
    # (1) 5 | 9+7 = 16| 0 | 0
    # (3) 9 | 7       | 5 | 0
    # (5) 7 | 0       | 9 | 5
    # (6) 0 | 7       | 9 | 5

    # Gamma (class 2)
    # (1) 5 | 1+7 = 8| 0 | 0
    # (3) 0 | 8      | 5 | 0
    # (5) 1 | 7      | 0 | 5
    # (6) 7 | 1      | 0 | 5

    class inner_fake_tree():
        def __init__(self, nn, cl, cr, v):
            self.weighted_n_node_samples = nn
            self.children_left = cl
            self.children_right = cr
            self.value = v

    class fake_tree():
        def __init__(self, nn, cl, cr, v):
            self.tree_ = inner_fake_tree(nn, cl, cr, v)
            self.__class__ = sklearn.tree.tree.DecisionTreeClassifier

    weighted_n_node_samples = np.array([34,10,24,9,15,8,7], dtype = np.int)
    children_left = np.array([2,-1,4,-1,6,-1,-1], dtype = np.int)
    children_right = np.array([1,-1,3,-1,5,-1,-1], dtype = np.int)
    value = np.array([[21, 13],
                      [5, 5],
                      [16, 8],
                      [9, 0],
                      [7, 8],
                      [7, 1],
                      [0, 7]], dtype = np.float).reshape((-1,1,2))

    test = fake_tree(weighted_n_node_samples,
                     children_left,
                     children_right,
                     value)

    n_leaf = 4

    g_static, n_static = smooth_rf.create_Gamma_eta_tree(test)

    n_expected = np.array([[10,24,0,0],
                           [9,15,10,0],
                           [8,7,9,10],
                           [7,8,9,10]])
    g_expected = np.array([[[5,16,0,0],
                            [9,7,5,0],
                            [7,0,9,5],
                            [0,7,9,5]],
                           [[5,8,0,0],
                            [0,8,5,0],
                            [1,7,0,5],
                            [7,1,0,5]]])
    assert np.all(g_static == g_expected), \
        "static test's Gamma failed to reproduce correct solutions"
    assert np.all(n_static == n_expected), \
        "static test's eta failed to reproduce correct solutions"


def test_create_Gamma_eta_tree_classification_impurity():
    """
    test for create_Gamma_eta_tree, classification tree -standard,impurity only

    Both static and random tests (random tests are more relative to structure
    than exact answers)
    """


    # random - structure output check
    # data creation
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

    tree = random_forest.estimators_[0]

    max_depth_range = np.max(smooth_rf.depth_per_node(tree)) + 1

    G, n = smooth_rf.create_Gamma_eta_tree(tree,
                                           distance_style = "impurity",
                                           levels = 10)

    # given we don't actually know the number of levels in each tree
    assert G.shape[0] == num_classes and \
           G.shape[1] == np.sum(tree.tree_.children_left == -1) and \
           len(G.shape) == 3, \
        "Gamma returned does not have the correct shape"

    assert n.shape ==  G.shape[1:3], \
        "eta returned does not have the correct shape"

    assert np.all(n >= 0), \
        "eta returned has negative values"

    assert np.all(n[:,0] ==
        tree.tree_.weighted_n_node_samples[tree.tree_.children_left == -1]),\
        "eta structure doesn't match up with number of observes per leaf"

    # static check

    # tree structure:
    # ~upper: left, lower: right~
    #                       num obs     class 1   class 2
    #    |--1                   10          5       5
    # -0-|                       34         21      13
    #    |   |--3              9            9       0
    #    |-2-|                  24          16      8
    #        |   |--5         8             7       1
    #        |-4-|             15           7       8
    #            |--6         7             0       7


    # eta
    # (1) 10 | 24 | 0  | 0
    # (3) 9  | 15 | 10 | 0
    # (5) 8  | 7  | 9  | 10
    # (6) 7  | 8  | 9  | 10


    # Gamma (class 1)
    # (1) 5 | 9+7 = 16| 0 | 0
    # (3) 9 | 7       | 5 | 0
    # (5) 7 | 0       | 9 | 5
    # (6) 0 | 7       | 9 | 5

    # Gamma (class 2)
    # (1) 5 | 1+7 = 8| 0 | 0
    # (3) 0 | 8      | 5 | 0
    # (5) 1 | 7      | 0 | 5
    # (6) 7 | 1      | 0 | 5

    class inner_fake_tree():
        def __init__(self, nn, cl, cr, v, impurity):
            self.weighted_n_node_samples = nn
            self.children_left = cl
            self.children_right = cr
            self.value = v
            self.impurity = impurity

    class fake_tree():
        def __init__(self, nn, cl, cr, v, impurity):
            self.tree_ = inner_fake_tree(nn, cl, cr, v, impurity)
            self.__class__ = sklearn.tree.tree.DecisionTreeClassifier

    weighted_n_node_samples = np.array([34,10,24,9,15,8,7], dtype = np.int)
    children_left = np.array([2,-1,4,-1,6,-1,-1], dtype = np.int)
    children_right = np.array([1,-1,3,-1,5,-1,-1], dtype = np.int)
    value = np.array([[21, 13],
                      [5, 5],
                      [16, 8],
                      [9, 0],
                      [7, 8],
                      [7, 1],
                      [0, 7]], dtype = np.float).reshape((-1,1,2))
    impurity = np.array([4,3,3,2,2,1,1])



    test = fake_tree(weighted_n_node_samples,
                     children_left,
                     children_right,
                     value,
                     impurity)

    n_leaf = 4

    g_static, n_static = smooth_rf.create_Gamma_eta_tree(test,
                                                distance_style = "impurity",
                                                levels = np.array([0,1,2,3,4]))

    n_expected = np.array([[10,24,0,0],
                           [9,15,10,0],
                           [8,7,9,10],
                           [7,8,9,10]])
    g_expected = np.array([[[5,16,0,0],
                            [9,7,5,0],
                            [7,0,9,5],
                            [0,7,9,5]],
                           [[5,8,0,0],
                            [0,8,5,0],
                            [1,7,0,5],
                            [7,1,0,5]]])
    assert np.all(g_static == g_expected), \
        "static test's Gamma failed to reproduce correct solutions"
    assert np.all(n_static == n_expected), \
        "static test's eta failed to reproduce correct solutions"

    g_static, n_static = smooth_rf.create_Gamma_eta_tree(test,
                                                distance_style = "impurity",
                                                levels = np.array([0,1,2,2.1,2.5,3,4]))

    n_expected = np.array([[10,24, 0,  0,0,  0 ],
                           [9 ,15,10,  0,0,  0 ],
                           [8 ,7 , 9,  0,0,  10],
                           [7 ,8 , 9,  0,0,  10]])
    g_expected = np.array([[[5,16,0,  0,0,  0],
                            [9,7 ,5,  0,0,  0],
                            [7,0 ,9,  0,0,  5],
                            [0,7 ,9,  0,0,  5]],
                           [[5,8 ,0,  0,0,  0],
                            [0,8 ,5,  0,0,  0],
                            [1,7 ,0,  0,0,  5],
                            [7,1 ,0,  0,0,  5]]])
    assert np.all(g_static == g_expected), \
        "static test's Gamma failed to reproduce correct solutions "+\
        "(extra levels)"
    assert np.all(n_static == n_expected), \
        "static test's eta failed to reproduce correct solutions "+\
        "(extra levels)"


def test_create_Gamma_eta_forest_regression():
    """
    test create_Gamma_eta_forest, regression forests - standard depth only

    compares to what is expected to be returned from create_Gamma_eta_tree -
    mostly just structurally
    """
    # parent = F
    n = 200
    n_tree = 10
    min_size_leaf = 1

    X = np.random.uniform(size = (n, 510), low = -1,high = 1)
    y = 10 * np.sin(np.pi * X[:,0]*X[:,1]) + 20 * ( X[:,2] - .5)**2 +\
        10 * X[:,3] + 5 * X[:,4] + np.random.normal(size = n)

    rf_class = sklearn.ensemble.RandomForestRegressor(n_estimators = n_tree,
                                            min_samples_leaf = min_size_leaf)
    random_forest = rf_class.fit(X = X,
                                 y = y.ravel())

    g, n, t = smooth_rf.create_Gamma_eta_forest(random_forest)

    assert g.shape == n.shape, \
        "Gamma and eta matrices are not the correct shared size "+\
        "(parents_all = False)"
    assert g.shape[0] == t.shape[0], \
        "the tree index vector doesn't have the correct number of observations "+\
        "(parents_all = False)"

    assert np.all(
        np.array(list(dict(Counter(t)).keys())) == np.arange(n_tree)),\
        "tree index doesn't contain expected tree index values "+\
        "(parents_all = False)"

    for t_idx, tree in enumerate(random_forest.estimators_):
        max_depth_range = np.int(np.max(smooth_rf.depth_per_node(tree)) + 1)
        G_tree, n_tree = smooth_rf.create_Gamma_eta_tree(tree)

        assert G_tree.shape[0] == np.sum(t == t_idx), \
            "shape of single Gamma from create_Gamma_eta_tree" +\
            "does not match structure from t_idx output "+\
            "(parents_all = False)"

        assert np.all(G_tree == g[t==t_idx,:][:,:max_depth_range]), \
            "doesn't match create_Gamma_eta_tree function for Gamma "+\
            "(parents_all = False)"
        if max_depth_range != g.shape[1]:
            assert np.all(g[t==t_idx,][:,max_depth_range:] == 0), \
                "extra dimensions, based on the global forest having larger" +\
                "depth than the individual tree (num %d) in Gamma are "+\
                "non-zero (parents_all = False)" %t_idx

        assert np.all(n_tree == n[t==t_idx,:][:,:max_depth_range]), \
            "doesn't match create_Gamma_eta_tree function for eta " +\
            "(parents_all = False"
        if max_depth_range != g.shape[1]:
            assert np.all(n[t==t_idx,][:,max_depth_range:] == 0), \
                "extra dimensions, based on the global forest having larger" +\
                "depth than the individual tree (num %d) in eta are "+\
                "non-zero (parents_all = False)" %t_idx

    # parent = T
    n = 200
    n_tree = 10
    min_size_leaf = 1

    X = np.random.uniform(size = (n, 510), low = -1,high = 1)
    y = 10 * np.sin(np.pi * X[:,0]*X[:,1]) + 20 * ( X[:,2] - .5)**2 +\
        10 * X[:,3] + 5 * X[:,4] + np.random.normal(size = n)

    rf_class = sklearn.ensemble.RandomForestRegressor(n_estimators = n_tree,
                                            min_samples_leaf = min_size_leaf)
    random_forest = rf_class.fit(X = X,
                                 y = y.ravel())

    g, n, t = smooth_rf.create_Gamma_eta_forest(random_forest,
                                                parents_all=True)

    assert g.shape == n.shape, \
        "Gamma and eta matrices are not the correct shared size "+\
        "(parents_all = True)"
    assert g.shape[0] == t.shape[0], \
        "the tree index vector doesn't have the correct number of observations "+\
        "(parents_all = True)"

    assert np.all(
        np.array(list(dict(Counter(t)).keys())) == np.arange(n_tree)),\
        "tree index doesn't contain expected tree index values "+\
        "(parents_all = True)"

    for t_idx, tree in enumerate(random_forest.estimators_):
        max_depth_range = np.int(np.max(smooth_rf.depth_per_node(tree)) + 1)
        G_tree, n_tree = smooth_rf.create_Gamma_eta_tree(tree,
                                                         parents_all=True)

        assert G_tree.shape[0] == np.sum(t == t_idx), \
            "shape of single Gamma from create_Gamma_eta_tree" +\
            "does not match structure from t_idx output "+\
            "(parents_all = True)"

        assert np.all(G_tree == g[t==t_idx,:][:,:max_depth_range]), \
            "doesn't match create_Gamma_eta_tree function for Gamma "+\
            "(parents_all = True)"

        if max_depth_range != g.shape[1]:
            for _ in range(5):
                idx = np.random.choice(np.int(g.shape[1] - max_depth_range))+\
                         max_depth_range
                assert np.all(g[t==t_idx,][:,idx] == \
                              g[t==t_idx,][:, max_depth_range]), \
                    "extra dimensions, based on the global forest having larger" +\
                    "depth than the individual tree (num %d) in Gamma are "+\
                    "equal to the last column of tree Gamma "+\
                    "(parents_all = True)" %t_idx

        assert np.all(n_tree == n[t==t_idx,:][:,:max_depth_range]), \
            "doesn't match create_Gamma_eta_tree function for eta " +\
            "(parents_all = True)"
        if max_depth_range != g.shape[1]:
            for _ in range(5):
                idx = np.random.choice(np.int(g.shape[1] - max_depth_range))+\
                         max_depth_range
                assert np.all(n[t==t_idx,][:,idx] == \
                              n[t==t_idx,][:, max_depth_range]), \
                    "extra dimensions, based on the global forest having larger" +\
                    "depth than the individual tree (num %d) in eta are "+\
                    "equal to the last column of tree eta "+\
                    "(parents_all = True)" %t_idx


def test_create_Gamma_eta_forest_regression_impurity():
    """
    test create_Gamma_eta_forest, regression forests - standard depth, impurity
    only

    compares to what is expected to be returned from create_Gamma_eta_tree -
    mostly just structurally
    """
    # parent = F
    n = 200
    n_tree = 10
    min_size_leaf = 1

    X = np.random.uniform(size = (n, 510), low = -1,high = 1)
    y = 10 * np.sin(np.pi * X[:,0]*X[:,1]) + 20 * ( X[:,2] - .5)**2 +\
        10 * X[:,3] + 5 * X[:,4] + np.random.normal(size = n)

    rf_class = sklearn.ensemble.RandomForestRegressor(n_estimators = n_tree,
                                            min_samples_leaf = min_size_leaf)
    random_forest = rf_class.fit(X = X,
                                 y = y.ravel())

    nose.tools.assert_raises(ValueError,
                             smooth_rf.create_Gamma_eta_forest,
                                random_forest,
                                distance_style="impurity")

    get_quantiles = smooth_rf.quantiles_distance_trees(
                            forest = [random_forest.estimators_[t_idx]
                                        for t_idx in np.arange(5)],
                            distance_style="impurity",
                                            levels = 5)

    g, n, t = smooth_rf.create_Gamma_eta_forest(random_forest,
                                            distance_style="impurity",
                                            levels = get_quantiles)


    assert g.shape == n.shape, \
        "Gamma and eta matrices are not the correct shared size "+\
        "(parents_all = False)"
    assert g.shape[0] == t.shape[0], \
        "the tree index vector doesn't have the correct number of observations "+\
        "(parents_all = False)"

    assert np.all(
        np.array(list(dict(Counter(t)).keys())) == np.arange(n_tree)),\
        "tree index doesn't contain expected tree index values "+\
        "(parents_all = False)"

    for t_idx, tree in enumerate(random_forest.estimators_):
        distance_mat, expected_levels = \
            smooth_rf.create_distance_mat_leaves(tree,
                                     distance_style="impurity",
                                     levels=get_quantiles)
        actual_levels = np.int(np.max(distance_mat)) + 1
        G_tree, n_tree = smooth_rf.create_Gamma_eta_tree(tree,
                                            distance_style="impurity",
                                            levels = get_quantiles)

        assert G_tree.shape[0] == np.sum(t == t_idx), \
            "shape of single Gamma from create_Gamma_eta_tree" +\
            "does not match structure from t_idx output "+\
            "(parents_all = False)"

        assert np.all(G_tree == g[t==t_idx,:][:,:actual_levels]), \
            "doesn't match create_Gamma_eta_tree function for Gamma "+\
            "(parents_all = False)"

        assert np.all(n_tree == n[t==t_idx,:][:,:actual_levels]), \
            "doesn't match create_Gamma_eta_tree function for eta " +\
            "(parents_all = False"


    # parent = T
    n = 200
    n_tree = 10
    min_size_leaf = 1

    X = np.random.uniform(size = (n, 510), low = -1,high = 1)
    y = 10 * np.sin(np.pi * X[:,0]*X[:,1]) + 20 * ( X[:,2] - .5)**2 +\
        10 * X[:,3] + 5 * X[:,4] + np.random.normal(size = n)

    rf_class = sklearn.ensemble.RandomForestRegressor(n_estimators = n_tree,
                                            min_samples_leaf = min_size_leaf)
    random_forest = rf_class.fit(X = X,
                                 y = y.ravel())

    g, n, t = smooth_rf.create_Gamma_eta_forest(random_forest,
                                                parents_all=True,
                                                distance_style="impurity",
                                                levels = 5)
    get_quantiles = smooth_rf.quantiles_distance_trees(
                            forest = [random_forest.estimators_[t_idx]
                                        for t_idx in np.arange(5)],
                            distance_style="impurity",
                                            levels = 5)
    assert g.shape == n.shape, \
        "Gamma and eta matrices are not the correct shared size "+\
        "(parents_all = True)"
    assert g.shape[0] == t.shape[0], \
        "the tree index vector doesn't have the correct number of observations "+\
        "(parents_all = True)"

    assert np.all(
        np.array(list(dict(Counter(t)).keys())) == np.arange(n_tree)),\
        "tree index doesn't contain expected tree index values "+\
        "(parents_all = True)"

    for t_idx, tree in enumerate(random_forest.estimators_):
        distance_mat, expected_levels = \
            smooth_rf.create_distance_mat_leaves(tree,
                                     distance_style="impurity",
                                     levels=get_quantiles)
        actual_levels = np.int(np.max(distance_mat)) + 1
        G_tree, n_tree = smooth_rf.create_Gamma_eta_tree(tree,
                                            distance_style="impurity",
                                            parents_all=True,
                                            levels = get_quantiles)

        assert G_tree.shape[0] == np.sum(t == t_idx), \
            "shape of single Gamma from create_Gamma_eta_tree" +\
            "does not match structure from t_idx output "+\
            "(parents_all = True)"

        assert np.all(G_tree == g[t==t_idx,:][:,:actual_levels]), \
            "doesn't match create_Gamma_eta_tree function for Gamma "+\
            "(parents_all = True)"

        assert np.all(n_tree == n[t==t_idx,:][:,:actual_levels]), \
            "doesn't match create_Gamma_eta_tree function for eta " +\
            "(parents_all = True)"




def test_create_Gamma_eta_forest_classification():
    """
    test create_Gamma_eta_forest, classification forests - standard depth only

    compares to what is expected to be returned from create_Gamma_eta_tree -
    mostly just structurally


    """
    # parent = F
    n = 200
    n_tree = 10
    min_size_leaf = 1

    X = np.random.uniform(size = (n, 510), low = -1,high = 1)
    y = 10 * np.sin(np.pi * X[:,0]*X[:,1]) + 20 * ( X[:,2] - .5)**2 +\
        10 * X[:,3] + 5 * X[:,4] + np.random.normal(size = n)

    y_cat = np.array(
                     pd.cut(y, bins = 5, labels = np.arange(5, dtype = np.int)),
                     dtype = np.int)

    y = y_cat

    num_classes = len(Counter(y_cat).keys())

    rf_class = sklearn.ensemble.RandomForestClassifier(n_estimators = n_tree,
                                            min_samples_leaf = min_size_leaf)
    random_forest = rf_class.fit(X = X,
                                 y = y.ravel())

    g, n, t = smooth_rf.create_Gamma_eta_forest(random_forest)

    assert g.shape[1:] == n.shape, \
        "Gamma and eta matrices are not the correct shared size" +\
            "(parents_all = False)"
    assert g.shape[1] == t.shape[0], \
        "the tree index vector doesn't have the correct number of observations" +\
            "(parents_all = False)"
    assert g.shape[0] == num_classes, \
        "Gamma matrix dimensions don't match the number of classes correctly" +\
            "(parents_all = False)"

    assert np.all(
        np.array(list(dict(Counter(t)).keys())) == np.arange(n_tree)),\
        "tree index doesn't contain expected tree index values" +\
            "(parents_all = False)"

    for t_idx, tree in enumerate(random_forest.estimators_):
        max_depth_range = np.int(np.max(smooth_rf.depth_per_node(tree)) + 1)
        G_tree, n_tree = smooth_rf.create_Gamma_eta_tree(tree)

        assert G_tree.shape[1] == np.sum(t == t_idx), \
            "shape of single Gamma from create_Gamma_eta_tree" +\
            "does not match structure from t_idx output" +\
            "(parents_all = False)"

        assert np.all(G_tree == g[:,t==t_idx,:][:,:,:max_depth_range]), \
            "doesn't match create_Gamma_eta_tree function for Gamma" +\
            "(parents_all = False)"
        if max_depth_range != g.shape[2]:
            assert np.all(g[:,t==t_idx,][:,:,max_depth_range:] == 0), \
                "extra dimensions, based on the global forest having larger" +\
                "depth than the individual tree (num %d) in Gamma are "+\
                "non-zero (parents_all = False)"%t_idx

        assert np.all(n_tree == n[t==t_idx,:][:,:max_depth_range]), \
            "doesn't match create_Gamma_eta_tree function for eta" +\
            "(parents_all = False)"
        if max_depth_range != g.shape[2]:
            assert np.all(n[t==t_idx,][:,max_depth_range:] == 0), \
                "extra dimensions, based on the global forest having larger" +\
                "depth than the individual tree (num %d) in eta are "+\
                "non-zero (parents_all = False)" %t_idx
    # parent = T
    n = 200
    n_tree = 10
    min_size_leaf = 1

    X = np.random.uniform(size = (n, 510), low = -1,high = 1)
    y = 10 * np.sin(np.pi * X[:,0]*X[:,1]) + 20 * ( X[:,2] - .5)**2 +\
        10 * X[:,3] + 5 * X[:,4] + np.random.normal(size = n)

    y_cat = np.array(
                     pd.cut(y, bins = 5, labels = np.arange(5, dtype = np.int)),
                     dtype = np.int)

    y = y_cat

    num_classes = len(Counter(y_cat).keys())

    rf_class = sklearn.ensemble.RandomForestClassifier(n_estimators = n_tree,
                                            min_samples_leaf = min_size_leaf)
    random_forest = rf_class.fit(X = X,
                                 y = y.ravel())

    g, n, t = smooth_rf.create_Gamma_eta_forest(random_forest,
                                                parents_all=True)

    assert g.shape[1:] == n.shape, \
        "Gamma and eta matrices are not the correct shared size" +\
            "(parents_all = True)"
    assert g.shape[1] == t.shape[0], \
        "the tree index vector doesn't have the correct number of observations" +\
            "(parents_all = True)"
    assert g.shape[0] == num_classes, \
        "Gamma matrix dimensions don't match the number of classes correctly" +\
            "(parents_all = True)"

    assert np.all(
        np.array(list(dict(Counter(t)).keys())) == np.arange(n_tree)),\
        "tree index doesn't contain expected tree index values" +\
            "(parents_all = True)"

    for t_idx, tree in enumerate(random_forest.estimators_):
        max_depth_range = np.int(np.max(smooth_rf.depth_per_node(tree)) + 1)
        G_tree, n_tree = smooth_rf.create_Gamma_eta_tree(tree,
                                                         parents_all=True)

        assert G_tree.shape[1] == np.sum(t == t_idx), \
            "shape of single Gamma from create_Gamma_eta_tree" +\
            "does not match structure from t_idx output" +\
            "(parents_all = True)"

        assert np.all(G_tree == g[:,t==t_idx,:][:,:,:max_depth_range]), \
            "doesn't match create_Gamma_eta_tree function for Gamma" +\
            "(parents_all = True)"
        if max_depth_range != g.shape[2]:
            for _ in range(5):
                idx = np.random.choice(np.int(g.shape[2] - max_depth_range))+\
                         max_depth_range
                assert np.all(g[:,t==t_idx,:][:,:,idx] == \
                              g[:,t==t_idx,:][:,:,max_depth_range]), \
                    "extra dimensions, based on the global forest having larger" +\
                    "depth than the individual tree (num %d) in Gamma are "+\
                    "equal to the last column of tree Gamma "+\
                    "(parents_all = True)" %t_idx

        assert np.all(n_tree == n[t==t_idx,:][:,:max_depth_range]), \
            "doesn't match create_Gamma_eta_tree function for eta" +\
            "(parents_all = True)"
        if max_depth_range != g.shape[2]:
            for _ in range(5):
                idx = np.random.choice(np.int(g.shape[2] - max_depth_range))+\
                         max_depth_range
                assert np.all(n[t==t_idx,][:,idx] == \
                              n[t==t_idx,][:,max_depth_range]), \
                    "extra dimensions, based on the global forest having larger" +\
                    "depth than the individual tree (num %d) in eta are "+\
                    "equal to the last column of tree eta "+\
                    "(parents_all = True)" %t_idx





def test_create_Gamma_eta_forest_classification_impurity():
    """
    test create_Gamma_eta_forest, classification forests - standard, impurity
    only

    compares to what is expected to be returned from create_Gamma_eta_tree -
    mostly just structurally


    """
    # parent = F
    n = 200
    n_tree = 10
    min_size_leaf = 1

    X = np.random.uniform(size = (n, 510), low = -1,high = 1)
    y = 10 * np.sin(np.pi * X[:,0]*X[:,1]) + 20 * ( X[:,2] - .5)**2 +\
        10 * X[:,3] + 5 * X[:,4] + np.random.normal(size = n)

    y_cat = np.array(
                     pd.cut(y, bins = 5, labels = np.arange(5, dtype = np.int)),
                     dtype = np.int)

    y = y_cat

    num_classes = len(Counter(y_cat).keys())

    rf_class = sklearn.ensemble.RandomForestClassifier(n_estimators = n_tree,
                                            min_samples_leaf = min_size_leaf)
    random_forest = rf_class.fit(X = X,
                                 y = y.ravel())

    get_quantiles = smooth_rf.quantiles_distance_trees(
                            forest = [random_forest.estimators_[t_idx]
                                        for t_idx in np.arange(5)],
                            distance_style="impurity",
                                            levels = 5)

    g, n, t = smooth_rf.create_Gamma_eta_forest(random_forest,
                                            distance_style="impurity",
                                            levels = get_quantiles)

    assert g.shape[1:] == n.shape, \
        "Gamma and eta matrices are not the correct shared size" +\
            "(parents_all = False)"
    assert g.shape[1] == t.shape[0], \
        "the tree index vector doesn't have the correct number of observations" +\
            "(parents_all = False)"
    assert g.shape[0] == num_classes, \
        "Gamma matrix dimensions don't match the number of classes correctly" +\
            "(parents_all = False)"

    assert np.all(
        np.array(list(dict(Counter(t)).keys())) == np.arange(n_tree)),\
        "tree index doesn't contain expected tree index values" +\
            "(parents_all = False)"

    for t_idx, tree in enumerate(random_forest.estimators_):
        distance_mat, expected_levels = \
            smooth_rf.create_distance_mat_leaves(tree,
                                     distance_style="impurity",
                                     levels=get_quantiles)
        actual_levels = np.int(np.max(distance_mat)) + 1
        G_tree, n_tree = smooth_rf.create_Gamma_eta_tree(tree,
                                            distance_style="impurity",
                                            levels = get_quantiles)

        assert G_tree.shape[1] == np.sum(t == t_idx), \
            "shape of single Gamma from create_Gamma_eta_tree" +\
            "does not match structure from t_idx output" +\
            "(parents_all = False)"

        assert np.all(G_tree == g[:,t==t_idx,:][:,:,:actual_levels]), \
            "doesn't match create_Gamma_eta_tree function for Gamma" +\
            "(parents_all = False)"
        if actual_levels != g.shape[2]:
            assert np.all(g[:,t==t_idx,][:,:,actual_levels:] == 0), \
                "extra dimensions, based on the global forest having larger" +\
                "depth than the individual tree (num %d) in Gamma are "+\
                "non-zero (parents_all = False)"%t_idx

        assert np.all(n_tree == n[t==t_idx,:][:,:actual_levels]), \
            "doesn't match create_Gamma_eta_tree function for eta" +\
            "(parents_all = False)"
        if actual_levels != g.shape[2]:
            assert np.all(n[t==t_idx,][:,actual_levels:] == 0), \
                "extra dimensions, based on the global forest having larger" +\
                "depth than the individual tree (num %d) in eta are "+\
                "non-zero (parents_all = False)" %t_idx
    # parent = T
    n = 200
    n_tree = 10
    min_size_leaf = 1

    X = np.random.uniform(size = (n, 510), low = -1,high = 1)
    y = 10 * np.sin(np.pi * X[:,0]*X[:,1]) + 20 * ( X[:,2] - .5)**2 +\
        10 * X[:,3] + 5 * X[:,4] + np.random.normal(size = n)

    y_cat = np.array(
                     pd.cut(y, bins = 5, labels = np.arange(5, dtype = np.int)),
                     dtype = np.int)

    y = y_cat

    num_classes = len(Counter(y_cat).keys())

    rf_class = sklearn.ensemble.RandomForestClassifier(n_estimators = n_tree,
                                            min_samples_leaf = min_size_leaf)
    random_forest = rf_class.fit(X = X,
                                 y = y.ravel())

    g, n, t = smooth_rf.create_Gamma_eta_forest(random_forest,
                                                parents_all=True)

    assert g.shape[1:] == n.shape, \
        "Gamma and eta matrices are not the correct shared size" +\
            "(parents_all = True)"
    assert g.shape[1] == t.shape[0], \
        "the tree index vector doesn't have the correct number of observations" +\
            "(parents_all = True)"
    assert g.shape[0] == num_classes, \
        "Gamma matrix dimensions don't match the number of classes correctly" +\
            "(parents_all = True)"

    assert np.all(
        np.array(list(dict(Counter(t)).keys())) == np.arange(n_tree)),\
        "tree index doesn't contain expected tree index values" +\
            "(parents_all = True)"

    for t_idx, tree in enumerate(random_forest.estimators_):
        max_depth_range = np.int(np.max(smooth_rf.depth_per_node(tree)) + 1)
        G_tree, n_tree = smooth_rf.create_Gamma_eta_tree(tree,
                                                         parents_all=True)

        assert G_tree.shape[1] == np.sum(t == t_idx), \
            "shape of single Gamma from create_Gamma_eta_tree" +\
            "does not match structure from t_idx output" +\
            "(parents_all = True)"

        assert np.all(G_tree == g[:,t==t_idx,:][:,:,:max_depth_range]), \
            "doesn't match create_Gamma_eta_tree function for Gamma" +\
            "(parents_all = True)"
        if max_depth_range != g.shape[2]:
            for _ in range(5):
                idx = np.random.choice(np.int(g.shape[2] - max_depth_range))+\
                         max_depth_range
                assert np.all(g[:,t==t_idx,:][:,:,idx] == \
                              g[:,t==t_idx,:][:,:,max_depth_range]), \
                    "extra dimensions, based on the global forest having larger" +\
                    "depth than the individual tree (num %d) in Gamma are "+\
                    "equal to the last column of tree Gamma "+\
                    "(parents_all = True)" %t_idx

        assert np.all(n_tree == n[t==t_idx,:][:,:max_depth_range]), \
            "doesn't match create_Gamma_eta_tree function for eta" +\
            "(parents_all = True)"
        if max_depth_range != g.shape[2]:
            for _ in range(5):
                idx = np.random.choice(np.int(g.shape[2] - max_depth_range))+\
                         max_depth_range
                assert np.all(n[t==t_idx,][:,idx] == \
                              n[t==t_idx,][:,max_depth_range]), \
                    "extra dimensions, based on the global forest having larger" +\
                    "depth than the individual tree (num %d) in eta are "+\
                    "equal to the last column of tree eta "+\
                    "(parents_all = True)" %t_idx



def test_smooth_classifier():
    """
    test for smooth- classifier, only runs on example dataset, checks for errs
    """

    X_trained = np.concatenate(
    (np.random.normal(loc = (1,2), scale = .6, size = (100,2)),
    np.random.normal(loc = (-1.2, -.5), scale = .6, size = (100,2))),
    axis = 0)
    y_trained = np.concatenate((np.zeros(100, dtype = np.int),
                         np.ones(100, dtype = np.int)))
    amount = np.int(200)

    # creating a random forest
    rf_class_known = sklearn.ensemble.RandomForestClassifier(
                                                    n_estimators = 5,
                                                    min_samples_leaf = 1)
    fit_rf_known = rf_class_known.fit(X = np.array(X_trained)[:amount,:],
                                  y = y_trained[:amount].ravel())
    forest = fit_rf_known.estimators_

    random_forest = fit_rf_known

    # general check for erroring
    try:
        a,b,c,d = smooth_rf.smooth(random_forest, X_trained, y_trained,
                                    parents_all=True, verbose = False)
    except:
        assert False, \
            "error running smoothing_function for a random forest classifier"

    # sanity check
    a,b,c,d = smooth_rf.smooth(random_forest, X_trained, y_trained,
                                    parents_all=True, verbose=False,
                                    sanity_check=True)

    no_update_pred = a.predict(X_trained)
    base_pred = random_forest.predict(X_trained)

    assert np.all(no_update_pred == base_pred), \
        "sanity check for rf classifier in smoother failed"

    # general check for erroring adam
    try:
        a,b,c,d = smooth_rf.smooth(random_forest, X_trained, y_trained,
                                    parents_all = True, verbose = False,
                                    adam = {"alpha": .001, "beta_1": .9,
                                            "beta_2": .999,"eps": 1e-8})
    except:
        assert False, \
            "error running smoothing_function for a random forest "+\
            "classifier with adam"


def test_smooth_classifier_impurity():
    """
    test for smooth- classifier - distance=impurity, only runs on example
    dataset, checks for errs
    """

    X_trained = np.concatenate(
    (np.random.normal(loc = (1,2), scale = .6, size = (100,2)),
    np.random.normal(loc = (-1.2, -.5), scale = .6, size = (100,2))),
    axis = 0)
    y_trained = np.concatenate((np.zeros(100, dtype = np.int),
                         np.ones(100, dtype = np.int)))
    amount = np.int(200)

    # creating a random forest
    rf_class_known = sklearn.ensemble.RandomForestClassifier(
                                                    n_estimators = 5,
                                                    min_samples_leaf = 1)
    fit_rf_known = rf_class_known.fit(X = np.array(X_trained)[:amount,:],
                                  y = y_trained[:amount].ravel())
    forest = fit_rf_known.estimators_

    random_forest = fit_rf_known

    # general check for erroring
    try:
        a,b,c,d = smooth_rf.smooth(random_forest, X_trained, y_trained,
                                    parents_all=True, verbose = False,
                                    distance_style = "impurity",
                                    levels = 10)
    except:
        assert False, \
            "error running smoothing_function for a random forest "+\
            "classifier (distance = impurity)"

    # sanity check
    a,b,c,d = smooth_rf.smooth(random_forest, X_trained, y_trained,
                                    parents_all=True, verbose=False,
                                    sanity_check=True)

    no_update_pred = a.predict(X_trained)
    base_pred = random_forest.predict(X_trained)

    assert np.all(no_update_pred == base_pred), \
        "sanity check for rf classifier in smoother failed"

    # general check for erroring adam
    try:
        a,b,c,d = smooth_rf.smooth(random_forest, X_trained, y_trained,
                                    parents_all = True, verbose = False,
                                    distance_style = "impurity",
                                    levels = 10,
                                    adam = {"alpha": .001, "beta_1": .9,
                                            "beta_2": .999,"eps": 1e-8})
    except:
        assert False, \
            "error running smoothing_function for a random forest "+\
            "classifier with adam (distance = impurity)"





def test_smooth_regressor():
    """
    test for smooth- regressor, only runs on example dataset, checks for errs
    """

    X_trained = np.concatenate(
        (np.random.normal(loc = (1,2), scale = .6, size = (100,2)),
        np.random.normal(loc = (-1.2, -.5), scale = .6, size = (100,2))),
        axis = 0)
    y_trained = np.concatenate((np.zeros(100, dtype = np.int),
                         np.ones(100, dtype = np.int)))
    amount = np.int(200)
    # creating a random forest
    rf_reg = sklearn.ensemble.RandomForestRegressor(
                                                    n_estimators = 5,
                                                    min_samples_leaf = 1)
    fit_reg = rf_reg.fit(X = np.array(X_trained)[:amount,:],
                                  y = y_trained[:amount].ravel())
    forest = fit_reg.estimators_

    random_forest = fit_reg
    verbose = True
    parents_all = True
    distance_style = "standard"

    # general check for erroring
    try:
        a,b,c,d = smooth_rf.smooth(random_forest, X_trained, y_trained,
                                    parents_all=True, verbose = False)

    except:
        assert False, \
            "error running smoothing_function for a random forest regressor"

    # sanity check
    a,b,c,d = smooth_rf.smooth(random_forest, X_trained, y_trained,
                                    parents_all=True, verbose=False,
                                    sanity_check=True)

    no_update_pred = a.predict(X_trained)
    base_pred = random_forest.predict(X_trained)

    assert np.all(no_update_pred == base_pred), \
        "sanity check for rf regressor in smoother failed"

    try:
        a,b,c,d = smooth_rf.smooth(random_forest, X_trained, y_trained,
                                    parents_all=True, verbose=False,
                                    adam = {"alpha": .001, "beta_1": .9,
                                            "beta_2": .999,"eps": 1e-8})
    except:
        assert False, \
            "error running smoothing_function for a random forest "+\
            "classifier with adam"


    # harder example
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
    verbose = False
    parents_all = True
    distance_style = "standard"

    # general check for erroring
    try:
        a,b,c,d = smooth_rf.smooth(random_forest, X_trained, y_trained,
                                    parents_all=parents_all, verbose = verbose,
                                    adam = {"alpha": .001, "beta_1": .9,
                                            "beta_2": .999,"eps": 1e-8})

    except:
        assert False, \
            "error running smoothing_function for a random forest regressor"


def test_smooth_regressor_impurity():
    """
    test for smooth- regressor (dist = impurity), only runs on example dataset,
    checks for errs
    """

    X_trained = np.concatenate(
        (np.random.normal(loc = (1,2), scale = .6, size = (100,2)),
        np.random.normal(loc = (-1.2, -.5), scale = .6, size = (100,2))),
        axis = 0)
    y_trained = np.concatenate((np.zeros(100, dtype = np.int),
                         np.ones(100, dtype = np.int)))
    amount = np.int(200)
    # creating a random forest
    rf_reg = sklearn.ensemble.RandomForestRegressor(
                                                    n_estimators = 5,
                                                    min_samples_leaf = 1)
    fit_reg = rf_reg.fit(X = np.array(X_trained)[:amount,:],
                                  y = y_trained[:amount].ravel())
    forest = fit_reg.estimators_

    random_forest = fit_reg
    verbose = True
    parents_all = True
    distance_style = "standard"

    # general check for erroring
    try:
        a,b,c,d = smooth_rf.smooth(random_forest, X_trained, y_trained,
                                    parents_all=True, verbose = False,
                                    distance_style = "impurity",
                                    levels = 10)

    except:
        assert False, \
            "error running smoothing_function for a random forest regressor"

    # sanity check
    a,b,c,d = smooth_rf.smooth(random_forest, X_trained, y_trained,
                                    parents_all=True, verbose=False,
                                    sanity_check=True)

    no_update_pred = a.predict(X_trained)
    base_pred = random_forest.predict(X_trained)

    assert np.all(no_update_pred == base_pred), \
        "sanity check for rf regressor in smoother failed"

    try:
        a,b,c,d = smooth_rf.smooth(random_forest, X_trained, y_trained,
                                    parents_all=True, verbose=False,
                                    distance_style = "impurity",
                                    levels = 10,
                                    adam = {"alpha": .001, "beta_1": .9,
                                            "beta_2": .999,"eps": 1e-8})
    except:
        assert False, \
            "error running smoothing_function for a random forest "+\
            "classifier with adam"


    # harder example
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
    verbose = False
    parents_all = True
    distance_style = "standard"

    # general check for erroring
    try:
        a,b,c,d = smooth_rf.smooth(random_forest, X_trained, y_trained,
                                    parents_all=parents_all, verbose = verbose,
                                    distance_style = "impurity",
                                    levels = 10,
                                    adam = {"alpha": .001, "beta_1": .9,
                                            "beta_2": .999,"eps": 1e-8})

    except:
        assert False, \
            "error running smoothing_function for a random forest regressor"



def test_bound_box_tree():
    """
    test bound_box_tree function (static and random shape analysis)
    """
    # random analysis
    X_train = np.concatenate(
        (np.random.normal(loc = (1,2,0), scale = .6, size = (100,3)),
        np.random.normal(loc = (-1.2, -.5,0), scale = .6, size = (100,3))),
    axis = 0)
    y_train = np.concatenate((np.zeros(100, dtype = np.int),
                             np.ones(100, dtype = np.int)))
    amount = np.int(200)
    s = 20
    c = y_train[:amount]
    # creating a random forest
    rf_class_known = sklearn.ensemble.RandomForestClassifier(
                                                        n_estimators = 3,
                                                        min_samples_leaf = 1)
    fit_rf_known = rf_class_known.fit(X = np.array(X_train)[:amount,:],
                                      y = y_train[:amount].ravel())
    forest = fit_rf_known.estimators_
    t = forest[0]

    bb = smooth_rf.bound_box_tree(t, X_train)

    n_node = t.tree_.children_left.shape[0]
    assert bb.shape == (n_node, X_train.shape[1], 2), \
        "bounded box shape is not as expected"

    for idx in np.arange(n_node):
        if t.tree_.children_left[idx] != -1:
            own_bb = bb[idx,
                np.arange(bb.shape[1],dtype = np.int) != t.tree_.feature[idx],:]
            l_bb = bb[t.tree_.children_left[idx],
                np.arange(bb.shape[1],dtype = np.int) != t.tree_.feature[idx],:]
            r_bb = bb[t.tree_.children_right[idx],
                np.arange(bb.shape[1],dtype = np.int) != t.tree_.feature[idx],:]

            assert np.all(own_bb == l_bb) and np.all(own_bb == r_bb), \
                "children bounding boxes differ more than expected"


    # static check

    # tree structure:
    # ~upper: left, lower: right~.
    #                   lower   upper       split
    #    |--1           0,0    50, 100      -1
    # -0-|.             0,0     100, 100    0
    #    |   |--3       50,0    100, 50     -1
    #    |-2-|          50,0    100,100     1
    #        |   |--5   50,50   75,100      -1
    #        |-4-|      50,50   100,100     0
    #            |--6   75,50   100,100     -1

    # creating desired structure
    class inner_fake_tree():
        def __init__(self, cl, cr, f, t):
            self.children_left = cl
            self.children_right = cr
            self.feature = f
            self.threshold = t

    class fake_tree():
        def __init__(self, cl, cr, f, t):
            self.tree_ = inner_fake_tree(cl, cr, f, t)
            self.__class__ = sklearn.tree.tree.DecisionTreeRegressor

    children_right = np.array([2,-1,4,-1,6,-1,-1], dtype = np.int)
    children_left = np.array([1,-1,3,-1,5,-1,-1], dtype = np.int)
    feature = np.array([0,-1,1,-1,0,-1,-1],dtype = np.int)
    threshold = np.array([50,-1,50,-1,75,-1,-1])


    test = fake_tree(children_left,children_right,feature,threshold)

    X = np.array([[100,100],
                  [0, 0]])

    bb_truth = np.array([[[0,100],
                          [0,100]],
                         [[0,50],
                          [0,100]],
                         [[50,100],
                          [0,100]],
                         [[50,100],
                          [0, 50]],
                         [[50,100],
                          [50,100]],
                         [[50,75],
                          [50,100]],
                         [[75,100],
                          [50,100]]])

    bb_static = smooth_rf.bound_box_tree(test, X)

    assert np.all(bb_truth == bb_static), \
        "bounding box doesn't replicate expected boxes for static example"


def test_center_tree():
    """
    test center_tree function (static and random shape analysis)
    """

    # random analysis
    X_train = np.concatenate(
        (np.random.normal(loc = (1,2,0), scale = .6, size = (100,3)),
        np.random.normal(loc = (-1.2, -.5,0), scale = .6, size = (100,3))),
    axis = 0)
    y_train = np.concatenate((np.zeros(100, dtype = np.int),
                             np.ones(100, dtype = np.int)))
    amount = np.int(200)
    s = 20
    c = y_train[:amount]
    # creating a random forest
    rf_class_known = sklearn.ensemble.RandomForestClassifier(
                                                        n_estimators = 3,
                                                        min_samples_leaf = 1)
    fit_rf_known = rf_class_known.fit(X = np.array(X_train)[:amount,:],
                                      y = y_train[:amount].ravel())
    forest = fit_rf_known.estimators_
    t = forest[0]

    cc = smooth_rf.center_tree(t, X_train)

    n_node = t.tree_.children_left.shape[0]
    assert cc.shape == (n_node, X_train.shape[1]), \
        "center shape is not as expected"

    for idx in np.arange(n_node):
        if t.tree_.children_left[idx] != -1:
            own_cc = cc[idx,
                np.arange(cc.shape[1],dtype = np.int) != t.tree_.feature[idx]]
            l_cc = cc[t.tree_.children_left[idx],
                np.arange(cc.shape[1],dtype = np.int) != t.tree_.feature[idx]]
            r_cc = cc[t.tree_.children_right[idx],
                np.arange(cc.shape[1],dtype = np.int) != t.tree_.feature[idx]]

            assert np.all(own_cc == l_cc) and np.all(own_cc == r_cc), \
                "children centers differ more than expected"


    # static check

    # tree structure:
    # ~upper: left, lower: right~.
    #                   lower   upper       split
    #    |--1           0,0    50, 100      -1
    # -0-|.             0,0     100, 100    0
    #    |   |--3       50,0    100, 50     -1
    #    |-2-|          50,0    100,100     1
    #        |   |--5   50,50   75,100      -1
    #        |-4-|      50,50   100,100     0
    #            |--6   75,50   100,100     -1

    # creating desired structure
    class inner_fake_tree():
        def __init__(self, cl, cr, f, t):
            self.children_left = cl
            self.children_right = cr
            self.feature = f
            self.threshold = t

    class fake_tree():
        def __init__(self, cl, cr, f, t):
            self.tree_ = inner_fake_tree(cl, cr, f, t)
            self.__class__ = sklearn.tree.tree.DecisionTreeRegressor

    children_right = np.array([2,-1,4,-1,6,-1,-1], dtype = np.int)
    children_left = np.array([1,-1,3,-1,5,-1,-1], dtype = np.int)
    feature = np.array([0,-1,1,-1,0,-1,-1],dtype = np.int)
    threshold = np.array([50,-1,50,-1,75,-1,-1])


    test = fake_tree(children_left,children_right,feature,threshold)

    X = np.array([[100,100],
                  [0, 0]])

    bb_truth = np.array([[[0,0],[100,100]],
                       [[0,0],[50,100]],
                       [[50,0],[100,100]],
                       [[50,0],[100,50]],
                       [[50,50],[100,100]],
                       [[50,50],[75,100]],
                       [[75,50],[100,100]]])

    cc_truth = bb_truth.mean(axis = 1)

    cc_static = smooth_rf.center_tree(test, X)

    assert np.all(cc_truth == cc_static), \
        "centers doesn't replicate expected boxes for static example"

def test_take_gradient():
    """
    test take_gradient (just check dimensions currently)

    TODO: check some individual level math...
    """
    # sketch this out (needed for other tests)
    n = 200
    min_size_leaf = 1

    X = np.random.uniform(size = (n, 510), low = -1,high = 1)
    y = 10 * np.sin(np.pi * X[:,0]*X[:,1]) + 20 * ( X[:,2] - .5)**2 +\
        10 * X[:,3] + 5 * X[:,4] + np.random.normal(size = n)

    rf_class = sklearn.ensemble.RandomForestRegressor(n_estimators = 2,
                                            min_samples_leaf = min_size_leaf)
    random_forest = rf_class.fit(X = X,
                                 y = y.ravel())

    tree = random_forest.estimators_[0]
    max_depth_range = np.max(smooth_rf.depth_per_node(tree)) + 1

    # y_leaves & weights (training):
    V = tree.decision_path(X)
    V_leaf = V[:,tree.tree_.children_left == -1]
    weight = np.array(V_leaf.sum(axis = 0)).ravel() # by column (leaf)

    weight_div = weight.copy()
    weight_div[weight_div == 0] = 1

    y_leaf = (V_leaf.T @ y) / weight_div

    Gamma, eta = smooth_rf.create_Gamma_eta_tree(tree)

    for _ in range(10):
        lamb = np.random.uniform(size = Gamma.shape[1])
        lamb = lamb / lamb.sum()

        grad = smooth_rf.take_gradient(y_leaf.ravel(),
                                       Gamma, eta,
                                       weight, lamb)

        assert grad.shape == lamb.shape, \
            "gradient returned has incorrect shape"

        # individual (try to compute just 1 gradient value?):
        #y_leaf_ind= y_leaf[np.random.] # TODO

def test_take_gradient_ce():
    """
    test take_gradient_ce (just check dimensions currently)

    TODO: check some individual level math...
    """
    # sketch this out (needed for other tests)
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

    tree = random_forest.estimators_[0]

    max_depth_range = np.max(smooth_rf.depth_per_node(tree)) + 1

    Gamma, eta = smooth_rf.create_Gamma_eta_tree(tree)

    # y_leaves & weights (training):
    V = tree.decision_path(X)
    V_leaf = V[:,tree.tree_.children_left == -1]
    weight = V_leaf.sum(axis = 0) # by column (leaf)

    weight_div = weight.copy()
    weight_div[weight_div == 0] = 1

    y_tune = np.array(pd.get_dummies(y))
    y_leaf = (V_leaf.T @ y_tune) / weight_div.T

    Gamma, eta = smooth_rf.create_Gamma_eta_tree(tree)

    Gamma_shape = Gamma.shape
    num_classes = Gamma.shape[0]
    Gamma = Gamma.reshape((Gamma.shape[0] * Gamma.shape[1],
                                 Gamma.shape[2]))

    eta = np.tile(eta, (num_classes,1))
    y_leaf = y_leaf.T.reshape((-1,))
    weight = np.tile(weight, num_classes)


    for _ in range(10):
        lamb = np.random.uniform(size = Gamma.shape[1])
        lamb = lamb / lamb.sum()

        grad = smooth_rf.take_gradient_ce(np.array(y_leaf).ravel(),
                                          Gamma, eta,
                                          np.array(weight).ravel(),
                                          lamb)

        assert grad.shape == lamb.shape, \
            "gradient returned has incorrect shape"

        # individual (try to compute just 1 gradient value?):
        #y_leaf_ind= y_leaf[np.random.] # TODO


def test_l2_s_grad_for_adam_wrapper():
    """
    test l2_s_grad_for_adam_wrapper
    """
    n = 200
    min_size_leaf = 1

    X = np.random.uniform(size = (n, 510), low = -1,high = 1)
    y = 10 * np.sin(np.pi * X[:,0]*X[:,1]) + 20 * ( X[:,2] - .5)**2 +\
        10 * X[:,3] + 5 * X[:,4] + np.random.normal(size = n)

    rf_class = sklearn.ensemble.RandomForestRegressor(n_estimators = 2,
                                            min_samples_leaf = min_size_leaf)
    random_forest = rf_class.fit(X = X,
                                 y = y.ravel())

    tree = random_forest.estimators_[0]
    max_depth_range = np.max(smooth_rf.depth_per_node(tree)) + 1

    # y_leaves & weights (training):
    V = tree.decision_path(X)
    V_leaf = V[:,tree.tree_.children_left == -1]
    weight = np.array(V_leaf.sum(axis = 0)).ravel() # by column (leaf)

    weight_div = weight.copy()
    weight_div[weight_div == 0] = 1

    y_leaf = (V_leaf.T @ y) / weight_div

    Gamma, eta = smooth_rf.create_Gamma_eta_tree(tree)

    wrapper_funct = smooth_rf.l2_s_grad_for_adam_wrapper(y_leaf,
                                                         Gamma, eta,
                                                         weight)

    for _ in range(10):
        lamb = np.random.uniform(size = Gamma.shape[1])
        lamb = lamb / lamb.sum()

        g_straight = smooth_rf.take_gradient(y_leaf, Gamma, eta, weight,
                                             lamb)
        g_wrapper = wrapper_funct(lamb)

        assert np.all(g_straight == g_wrapper), \
            "l2 wrapper function should return same values at object it wraps"


def test_ce_s_grad_for_adam_wrapper():
    """
    test ce_s_grad_for_adam_wrapper
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

    tree = random_forest.estimators_[0]

    max_depth_range = np.max(smooth_rf.depth_per_node(tree)) + 1

    Gamma, eta = smooth_rf.create_Gamma_eta_tree(tree)

    # y_leaves & weights (training):
    V = tree.decision_path(X)
    V_leaf = V[:,tree.tree_.children_left == -1]
    weight = V_leaf.sum(axis = 0) # by column (leaf)

    weight_div = weight.copy()
    weight_div[weight_div == 0] = 1

    y_tune = np.array(pd.get_dummies(y))
    p_leaf = (V_leaf.T @ y_tune) / weight_div.T

    Gamma, eta = smooth_rf.create_Gamma_eta_tree(tree)

    Gamma_shape = Gamma.shape
    num_classes = Gamma.shape[0]
    Gamma = Gamma.reshape((Gamma.shape[0] * Gamma.shape[1],
                                 Gamma.shape[2]))

    eta = np.tile(eta, (num_classes,1))
    p_leaf = p_leaf.T.reshape((-1,))
    weight = np.tile(weight, num_classes)

    wrapper_funct = smooth_rf.ce_s_grad_for_adam_wrapper(
                                        np.array(p_leaf).ravel(),
                                        Gamma,
                                        eta,
                                        np.array(weight).ravel())

    for _ in range(10):
        lamb = np.random.uniform(size = Gamma.shape[1])
        lamb = lamb / lamb.sum()

        g_straight = smooth_rf.take_gradient_ce(np.array(p_leaf).ravel(),
                                                Gamma, eta,
                                                np.array(weight).ravel(),
                                                lamb)

        g_wrapper = wrapper_funct(lamb)

        assert np.all(g_straight == g_wrapper), \
            "ce wrapper function should return same values at object it wraps"


def test_change_in_impurity():
    """
    test for change_in_impurity function
    """
    for i in range(5):
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

        tree = random_forest.estimators_[0]

        c_left  = tree.tree_.children_left
        c_right = tree.tree_.children_right

        impurity_diff = smooth_rf.change_in_impurity(tree)

        n_nodes = c_right.shape[0]

        impurity = np.zeros(n_nodes)

        for split in np.arange(n_nodes, dtype = np.int):
            impurity[split] += impurity_diff[split]
            if c_left[split] != -1:
                impurity[c_left[split]] += impurity[split]
            if c_right[split] != -1:
                impurity[c_right[split]] += impurity[split]

        assert np.all(impurity == tree.tree_.impurity), \
            "impurity differences cannot re-create impurity vector"


def test_quantiles_distance_trees():
    """
    test quantiles_distance_trees function
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

    rf_class = sklearn.ensemble.RandomForestClassifier(n_estimators = 5,
                                            min_samples_leaf = min_size_leaf)
    random_forest = rf_class.fit(X = X,
                                 y = y.ravel())

    forest = random_forest.estimators_

    for style in ["standard", "max", "min"]:
        values = smooth_rf.quantiles_distance_trees(forest,
                                          style=style,
                                          distance_style="impurity",
                                 levels=None)

        all_vals = np.zeros(0)
        for tree in forest:
            single_tree_values, _ = smooth_rf.create_distance_mat_leaves(tree,
                                                    style=style,
                                                    distance_style="impurity",
                                                    levels=None)
            all_vals = np.append(all_vals, single_tree_values)

        assert np.all(values == all_vals), \
            "calculating each tree values is the same as doing it for the "+\
            "forest"

    for levels in np.random.choice(np.arange(5,15), replace=False, size = 5):
        for style in ["standard", "max", "min"]:
            values = smooth_rf.quantiles_distance_trees(forest,
                                              style=style,
                                              distance_style="impurity",
                                              levels=levels)
            assert values.shape[0] == levels + 1, \
                "quantiles returned should be a vector of 1 more length "+\
                "than levels integter"

            assert np.all(values == np.array(sorted(values))), \
                "quantiles returned should naturally be sorted"

            all_vals = np.zeros(0)
            for tree in forest:
                single_tree_values, _ = smooth_rf.create_distance_mat_leaves(tree,
                                                    style=style,
                                                    distance_style="impurity",
                                                    levels=None)
                all_vals = np.append(all_vals, single_tree_values)

            output = np.quantile(all_vals, q = np.arange(levels + 1)/(levels) )

            assert np.all(output == values), \
                "output quantiles match correct quantiles for distances "+\
                "from trees"
