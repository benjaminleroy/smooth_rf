import numpy as np
import pandas as pd
import scipy.sparse
import sparse
import sklearn
from sklearn.ensemble import RandomForestRegressor
from collections import Counter
import sys, os
#
#sys.path.append("../functions/")
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
    test for test_create_distance_mat_leaves

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
        d1 = smooth_rf.create_distance_mat_leaves(tree, style = style)
        d2 = smooth_rf.create_distance_mat_leaves(decision_mat_leaves = v_leaf,
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
        d1 = smooth_rf.create_distance_mat_leaves(test, style = style)

        assert np.all(d1 == d1_should[d_idx]), \
            "static test failed to reproduce correct solutions"

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

def test_create_Gamma_eta_forest_regression():
    """
    test create_Gamma_eta_forest, regression forests - standard depth only

    compares to what is expected to be returned from create_Gamma_eta_tree -
    mostly just structurally
    """
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
        "Gamma and eta matrices are not the correct shared size"
    assert g.shape[0] == t.shape[0], \
        "the tree index vector doesn't have the correct number of observations"

    assert np.all(
        np.array(list(dict(Counter(t)).keys())) == np.arange(n_tree)),\
        "tree index doesn't contain expected tree index values"

    for t_idx, tree in enumerate(random_forest.estimators_):
        max_depth_range = np.int(np.max(smooth_rf.depth_per_node(tree)) + 1)
        G_tree, n_tree = smooth_rf.create_Gamma_eta_tree(tree)

        assert G_tree.shape[0] == np.sum(t == t_idx), \
            "shape of single Gamma from create_Gamma_eta_tree" +\
            "does not match structure from t_idx output"

        assert np.all(G_tree == g[t==t_idx,:][:,:max_depth_range]), \
            "doesn't match create_Gamma_eta_tree function for Gamma"
        if max_depth_range != g.shape[1]:
            assert np.all(g[t==t_idx,][:,max_depth_range:] == 0), \
                "extra dimensions, based on the global forest having larger" +\
                "depth than the individual tree (num %d) in Gamma are "+\
                "non-zero" %t_idx

        assert np.all(n_tree == n[t==t_idx,:][:,:max_depth_range]), \
            "doesn't match create_Gamma_eta_tree function for eta"
        if max_depth_range != g.shape[1]:
            assert np.all(n[t==t_idx,][:,max_depth_range:] == 0), \
                "extra dimensions, based on the global forest having larger" +\
                "depth than the individual tree (num %d) in eta are "+\
                "non-zero" %t_idx

def test_create_Gamma_eta_forest_classification():
    """
    test create_Gamma_eta_forest, classification forests - standard depth only

    compares to what is expected to be returned from create_Gamma_eta_tree -
    mostly just structurally


    """
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
        "Gamma and eta matrices are not the correct shared size"
    assert g.shape[1] == t.shape[0], \
        "the tree index vector doesn't have the correct number of observations"
    assert g.shape[0] == num_classes, \
        "Gamma matrix dimensions don't match the number of classes correctly"

    assert np.all(
        np.array(list(dict(Counter(t)).keys())) == np.arange(n_tree)),\
        "tree index doesn't contain expected tree index values"

    for t_idx, tree in enumerate(random_forest.estimators_):
        max_depth_range = np.int(np.max(smooth_rf.depth_per_node(tree)) + 1)
        G_tree, n_tree = smooth_rf.create_Gamma_eta_tree(tree)

        assert G_tree.shape[1] == np.sum(t == t_idx), \
            "shape of single Gamma from create_Gamma_eta_tree" +\
            "does not match structure from t_idx output"

        assert np.all(G_tree == g[:,t==t_idx,:][:,:,:max_depth_range]), \
            "doesn't match create_Gamma_eta_tree function for Gamma"
        if max_depth_range != g.shape[1]:
            assert np.all(g[:,t==t_idx,][:,:,max_depth_range:] == 0), \
                "extra dimensions, based on the global forest having larger" +\
                "depth than the individual tree (num %d) in Gamma are "+\
                "non-zero" %t_idx

        assert np.all(n_tree == n[t==t_idx,:][:,:max_depth_range]), \
            "doesn't match create_Gamma_eta_tree function for eta"
        if max_depth_range != g.shape[1]:
            assert np.all(n[t==t_idx,][:,max_depth_range:] == 0), \
                "extra dimensions, based on the global forest having larger" +\
                "depth than the individual tree (num %d) in eta are "+\
                "non-zero" %t_idx


def test_take_gradient():
    """
    test for take_gradient TODO
    """
    pass


def test_smooth_classifier():
    """
    test for smooth- classifier, only runs on example dataset, checks for errs

    """

    try:
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
        verbose = True
        parents_all = True
        distance_style = "standard"

        # Gamma (2, 370, 8)
        # eta (370, 8)
        a,b,c,d = smooth_rf.smooth(random_forest, X_trained, y_trained,
                                    parents_all=True, verbose = False)
    except:
        assert("error running smoothing_function for a random forest classifier")


def test_smooth_regressor():
    """
    test for smooth- regressor, only runs on example dataset, checks for errs
    """
    try:
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
        fit_reg = rf_class_known.fit(X = np.array(X_trained)[:amount,:],
                                      y = y_trained[:amount].ravel())
        forest = fit_rf_known.estimators_

        random_forest = fit_rf_known
        verbose = True
        parents_all = True
        distance_style = "standard"

        a,b,c,d = smooth_rf.smooth(random_forest, X_trained, y_trained,
                                    parents_all=True, verbose = False)

    except:
        assert("error running smoothing_function for a random forest classifier")


