import numpy as np
import pandas as pd
import scipy.sparse
import sparse
import sklearn
from sklearn.ensemble import RandomForestRegressor
from collections import Counter
import sys, os

import smooth_rf

def test_prune_tree_full():

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

    t = tree.tree_

    n_obs_trained = X_train.shape[0]
    random_state = tree.random_state
    oob_indices = \
        sklearn.ensemble.forest._generate_unsampled_indices(
                                                         random_state,
                                                         n_obs_trained)
    X_tune = X_train[oob_indices,:]
    y_tune = y_train[oob_indices]

    decision_path = tree.decision_path(X_tune)
    # need to get reg or class
    tree.tree_.value

def test_inner_prune():
    """
    static test only of _inner_prune
    """

    #                       num obs     class 1   class 2   |T_t|
    #    |--1                   10          5       5        1
    # -0-|                       34         21      13       4
    #    |   |--3              9            9       0        1
    #    |-2-|                  24          16      8        3
    #        |   |--5         8             7       1        1
    #        |-4-|             15           7       8        2
    #            |--6         7             0       7        1

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
    nn = weighted_n_node_samples
    children_left = np.array([2,-1,4,-1,6,-1,-1], dtype = np.int)
    children_right = np.array([1,-1,3,-1,5,-1,-1], dtype = np.int)
    value = np.array([[21, 13],
                      [5, 5],
                      [16, 8],
                      [9, 0],
                      [7, 8],
                      [7, 1],
                      [0, 7]], dtype = np.float).reshape((-1,1,2))

    T_t = np.array([4,1,3,1,2,1,1], dtype = np.int)

    test = fake_tree(weighted_n_node_samples,
                     children_left,
                     children_right,
                     value)

    p_df = smooth_rf.tune_prune._initialize_prune_df(test)


    # root node
    prune_idx = 0

    updated_df = smooth_rf.tune_prune._inner_prune(p_df, prune_idx)

    assert np.all(updated_df.c_left <= -1),\
        "prune to root: all nodes either leaf or pruned."

    assert np.all(updated_df["R({t})"] == updated_df["R(T_t)"]),\
        "prune to root: all nodes R({t}) equals R(T_t)."

    # check after already been pruned/leaf:
    try:
        prune_idx = 0
        smooth_rf.tune_prune._inner_prune(updated_df, prune_idx)
        assert False,\
            "prune at leaf should raise an error"
    except ValueError:
        pass

    try:
        prune_idx = 2
        smooth_rf.tune_prune._inner_prune(updated_df, prune_idx)
        assert False,\
            "prune at node already pruned should raise an error"
    except ValueError:
        pass


    # interior node
    prune_idx = 4

    updated_df = smooth_rf.tune_prune._inner_prune(p_df, prune_idx)

    prune_idx_children_bool = [x in {5,6} for x in updated_df.idx]
    prune_idx_not_children_bool = [x not in {5,6} for x in updated_df.idx]

    assert (updated_df.c_left[updated_df.idx == prune_idx] == -1).values[0],\
        "prune to node: prune node should now be a leaf."

    assert (updated_df.c_left[prune_idx_children_bool] == -2).values[0],\
        "prune to node: descendants of prune node should now be pruned."

    assert (updated_df.c_left[prune_idx_not_children_bool] != -2).values[0],\
        "prune to node: non-desecendants of prune node should not be pruned."


def test_inner_prune_update_upward():
    """
    static test only of _inner_prune_update_upward
    """

    #                       num obs     class 1   class 2   |T_t|
    #    |--1                   10          5       5        1
    # -0-|                       34         21      13       4
    #    |   |--3              9            9       0        1
    #    |-2-|                  24          16      8        3
    #        |   |--5         8             7       1        1
    #        |-4-|             15           7       8        2
    #            |--6         7             0       7        1

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
    nn = weighted_n_node_samples
    children_left = np.array([2,-1,4,-1,6,-1,-1], dtype = np.int)
    children_right = np.array([1,-1,3,-1,5,-1,-1], dtype = np.int)
    value = np.array([[21, 13],
                      [5, 5],
                      [16, 8],
                      [9, 0],
                      [7, 8],
                      [7, 1],
                      [0, 7]], dtype = np.float).reshape((-1,1,2))

    T_t = np.array([4,1,3,1,2,1,1], dtype = np.int)

    test = fake_tree(weighted_n_node_samples,
                     children_left,
                     children_right,
                     value)

    p_df = smooth_rf.tune_prune._initialize_prune_df(test)


    # root node
    prune_idx = 0
    out_df = smooth_rf.tune_prune._inner_prune_down(p_df, prune_idx)
    out_df2 = smooth_rf.tune_prune._inner_prune_update_upward(out_df,
                                                              prune_idx)

    # nothing should change from _inner_prune_down to
    # _inner_prune_update_upward if using root node
    assert np.all(out_df == out_df2),\
        "prune to root: _inner_prune_update_upward shouldn't update " +\
        "out_df if pruned to root."

    # interior node:
    prune_idx = 4
    out_df = smooth_rf.tune_prune._inner_prune_down(p_df, prune_idx)
    out_df2 = smooth_rf.tune_prune._inner_prune_update_upward(out_df,
                                                              prune_idx)


    prune_idx_ancestors = [x in {0,2} for x in out_df2.idx]
    prune_idx_not_ancestors = [x not in {0,2} for x in out_df2.idx]


    assert np.all(out_df.loc[prune_idx_not_ancestors,] == \
                  out_df2.loc[prune_idx_not_ancestors,]),\
        "prune to node 4: only change ancestor's information " +\
        "(for _inner_prune_update_upward)"

    assert np.all(out_df.loc[prune_idx_ancestors, "R(T_t)"] <= \
                  out_df2.loc[prune_idx_ancestors, "R(T_t)"]),\
        "prune to node 4: all parent nodes' R(T_t) values should increase"

    assert np.all(out_df.loc[prune_idx_ancestors, "|T_t|"] > \
                  out_df2.loc[prune_idx_ancestors, "|T_t|"]),\
        "prune to node 4: all parent nodes' |T_t| values should decrease"



def test_inner_prune_down():
    """
    static test only of _inner_prune_down
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
            self.__class__ = sklearn.tree.tree.DecisionTreeClassifier

    weighted_n_node_samples = np.array([34,10,24,9,15,8,7], dtype = np.int)
    nn = weighted_n_node_samples
    children_left = np.array([2,-1,4,-1,6,-1,-1], dtype = np.int)
    children_right = np.array([1,-1,3,-1,5,-1,-1], dtype = np.int)
    value = np.array([[21, 13],
                      [5, 5],
                      [16, 8],
                      [9, 0],
                      [7, 8],
                      [7, 1],
                      [0, 7]], dtype = np.float).reshape((-1,1,2))

    T_t = np.array([4,1,3,1,2,1,1], dtype = np.int)

    test = fake_tree(weighted_n_node_samples,
                     children_left,
                     children_right,
                     value)

    p_df = _initialize_prune_df(test)


    # root node
    prune_idx = 0
    out_df = _inner_prune_down(p_df, prune_idx)

    assert np.all(out_df.loc[out_df.idx != 0, "c_left"] == -2) and \
        np.all(out_df.loc[out_df.idx != 0, "c_right"] == -2) and \
        np.all(out_df.loc[out_df.idx != 0, "|T_t|"] == 0),\
        "prune to root: recursive setting of lost nodes is correct"

    assert np.all(out_df.loc[out_df.idx != 0, "R({t})"] == \
                  out_df.loc[out_df.idx != 0, "R(T_t)"]),\
        "prune to root: all nodes lost should R({t}) == R(T_t)"

    # interior node:
    prune_idx = 4
    out_df = smooth_rf.tune_prune._inner_prune_down(p_df, prune_idx)


    prune_idx_children_bool = [x in {5,6} for x in out_df.idx]

    assert np.all(out_df.loc[prune_idx_children_bool, "c_left"] == -2) and \
        np.all(out_df.loc[prune_idx_children_bool, "c_right"] == -2) and \
        np.all(out_df.loc[prune_idx_children_bool, "|T_t|"] == 0),\
        "prune to node 4: recursive setting of lost nodes is correct"

    assert np.all(out_df.loc[prune_idx_children_bool, "R({t})"] == \
                  out_df.loc[prune_idx_children_bool, "R(T_t)"]),\
        "prune to node: all nodes lost should R({t}) == R(T_t)"


    # random pruning
    for _ in np.arange(5):
        prune_idx = np.random.choice(4)
        out_df = _inner_prune(p_df, prune_idx)

        assert out_df.loc[prune_idx, "R(T_t)"] ==\
            p_df.loc[prune_idx, "R({t})"], \
            "R(T_t) should be R({t}) for pruned node"

        assert np.all(out_df.loc[prune_idx, ["c_left", "c_right"]] == -1) and\
            out_df.loc[prune_idx, "|T_t|"] == 1,\
            "t node should now be a 'leaf'"

        assert np.all(out_df.loc[p_df.loc[prune_idx, "c_left"], ["c_left", "c_right"]] == -2) and \
            np.all(out_df.loc[p_df.loc[prune_idx, "c_right"], ["c_left", "c_right"]] == -2), \
            "children of t should now have labels for their children as -2"

        assert np.all(out_df.loc[p_df.loc[prune_idx, "c_left"], "|T_t|"] == 0) and \
            np.all(out_df.loc[p_df.loc[prune_idx, "c_right"], "|T_t|"] == 0), \
            "children of t should now have |T_t| == 0"





def test_initialize_prune_df_classification():
    """
    test for initialize_prune_df for classification examples
    """
    # data creation
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

    tree = random_forest.estimators_[0]

    p_df = _initialize_prune_df(tree)

    assert p_df.shape == (tree.tree_.children_left.shape[0], 8), \
        "pruned_df should have 8 columns and num_nodes rows"

    assert np.all(p_df.columns == ["idx", "R({t})", "R(T_t)",
                                    "c_left", "c_right", "|T_t|",
                                    "parent", "n_obs"]), \
        "pruned_df column names are incorrect (or incorrectly ordered)"

    # static check

    # tree structure:
    # ~upper: left, lower: right~
    #                       num obs     class 1   class 2   |T_t|
    #    |--1                   10          5       5        1
    # -0-|                       34         21      13       4
    #    |   |--3              9            9       0        1
    #    |-2-|                  24          16      8        3
    #        |   |--5         8             7       1        1
    #        |-4-|             15           7       8        2
    #            |--6         7             0       7        1



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
    nn = weighted_n_node_samples
    children_left = np.array([2,-1,4,-1,6,-1,-1], dtype = np.int)
    children_right = np.array([1,-1,3,-1,5,-1,-1], dtype = np.int)
    value = np.array([[21, 13],
                      [5, 5],
                      [16, 8],
                      [9, 0],
                      [7, 8],
                      [7, 1],
                      [0, 7]], dtype = np.float).reshape((-1,1,2))

    T_t = np.array([4,1,3,1,2,1,1], dtype = np.int)

    test = fake_tree(weighted_n_node_samples,
                     children_left,
                     children_right,
                     value)

    r_single_t_val = r_single_t(test)

    r_T_t = r_single_t_val.copy()
    r_T_t[4] = (r_T_t[5]*nn[5] + r_T_t[6]*nn[6])/nn[4]
    r_T_t[2] = (r_T_t[3]*nn[3] + r_T_t[4]*nn[4])/nn[2]
    r_T_t[0] = (r_T_t[1]*nn[1] + r_T_t[2]*nn[2])/nn[0]


    p_df = _initialize_prune_df(test)

    assert np.any(p_df.idx == np.arange(p_df.shape[0], dtype = np.int)), \
        "idx should be ordered 0:num_nodes"

    assert np.all(p_df.loc[:,"|T_t|"] == T_t), \
        "static example T_t's values are incorrect"

    assert np.all(p_df.n_obs == weighted_n_node_samples), \
        "static example n_obs should grab weighted_n_node_samples"

    assert np.all(p_df.loc[:,"R({t})"] == r_single_t_val), \
        "static example: p_df's R({t}) should be same as r_single_t"

    assert np.all(p_df.loc[:,"R(T_t)"] == r_T_t), \
        "static example, p_df's R(T_t) incorrect"

    assert np.all(p_df.c_left == children_left), \
        "static example, children_left incorrectly stored"
    assert np.all(p_df.c_right == children_right), \
        "static example, children_right incorrectly stored"

    assert np.all(p_df.parent ==  calc_parent(test)), \
        "static example, p_df.parent is same as calc_parent"

def test_initialize_prune_df_regression():
    """
    test for initialize_prune_df for regression examples
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

    p_df = _initialize_prune_df(tree)

    assert p_df.shape == (tree.tree_.children_left.shape[0], 8), \
        "pruned_df should have 8 columns and num_nodes rows"

    assert np.all(p_df.columns == ["idx", "R({t})", "R(T_t)",
                                    "c_left", "c_right", "|T_t|",
                                    "parent", "n_obs"]), \
        "pruned_df column names are incorrect (or incorrectly ordered)"

    # static check

    # tree structure:
    # ~upper: left, lower: right~
    #                       num obs  |T_t|
    #    |--1                   10      1
    # -0-|                       34     4
    #    |   |--3              9        1
    #    |-2-|                  24      3
    #        |   |--5         8         1
    #        |-4-|             15       2
    #            |--6         7         1


    class inner_fake_tree():
        def __init__(self, nn, cl, cr, impurity):
            self.weighted_n_node_samples = nn
            self.children_left = cl
            self.children_right = cr
            self.impurity = impurity

    class fake_tree():
        def __init__(self, nn, cl, cr, impurity):
            self.tree_ = inner_fake_tree(nn, cl, cr, impurity)
            self.__class__ = sklearn.tree.tree.DecisionTreeRegressor

    weighted_n_node_samples = np.array([34,10,24,9,15,8,7], dtype = np.int)
    children_left = np.array([2,-1,4,-1,6,-1,-1], dtype = np.int)
    children_right = np.array([1,-1,3,-1,5,-1,-1], dtype = np.int)
    impurity = np.array([.7, .2, .3, .2, .1, 0, 0])

    T_t = np.array([4,1,3,1,2,1,1], dtype = np.int)

    test = fake_tree(weighted_n_node_samples,
                     children_left,
                     children_right,
                     impurity)

    r_single_t_val = r_single_t(test)

    r_T_t = r_single_t_val.copy()
    r_T_t[4] = (r_T_t[5]*nn[5] + r_T_t[6]*nn[6])/nn[4]
    r_T_t[2] = (r_T_t[3]*nn[3] + r_T_t[4]*nn[4])/nn[2]
    r_T_t[0] = (r_T_t[1]*nn[1] + r_T_t[2]*nn[2])/nn[0]


    p_df = _initialize_prune_df(test)

    assert np.any(p_df.idx == np.arange(p_df.shape[0], dtype = np.int)), \
        "idx should be ordered 0:num_nodes"

    assert np.all(p_df.loc[:,"|T_t|"] == T_t), \
        "static example T_t's values are incorrect"

    assert np.all(p_df.n_obs == weighted_n_node_samples), \
        "static example n_obs should grab weighted_n_node_samples"

    assert np.all(p_df.loc[:,"R({t})"] == r_single_t_val), \
        "static example: p_df's R({t}) should be same as r_single_t"

    assert np.all(p_df.loc[:,"R(T_t)"] == r_T_t), \
        "static example, p_df's R(T_t) incorrect"

    assert np.all(p_df.c_left == children_left), \
        "static example, children_left incorrectly stored"
    assert np.all(p_df.c_right == children_right), \
        "static example, children_right incorrectly stored"

    assert np.all(p_df.parent ==  calc_parent(test)), \
        "static example, p_df.parent is same as calc_parent"


def test_r_single_t_regression():
    """
    test for r_single_t for regression examples
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
    r_single_t_val = r_single_t(tree)

    assert r_single_t_val.shape == tree.tree_.children_left.shape, \
        "r_single_t output should be the same length as number of nodes"

    assert np.all((r_single_t_val >= r_single_t_val[tree.tree_.children_left]) +\
                  (r_single_t_val >= r_single_t_val[tree.tree_.children_right]) >= 1), \
        "parents should have more loss at least one child - in tree"

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
        def __init__(self, nn, cl, cr, impurity):
            self.weighted_n_node_samples = nn
            self.children_left = cl
            self.children_right = cr
            self.impurity = impurity

    class fake_tree():
        def __init__(self, nn, cl, cr, v):
            self.tree_ = inner_fake_tree(nn, cl, cr, v)
            self.__class__ = sklearn.tree.tree.DecisionTreeRegressor

    weighted_n_node_samples = np.array([34,10,24,9,15,8,7], dtype = np.int)
    children_left = np.array([2,-1,4,-1,6,-1,-1], dtype = np.int)
    children_right = np.array([1,-1,3,-1,5,-1,-1], dtype = np.int)
    impurity = np.array([.7, .2, .3, .2, .1, 0, 0])

    test = fake_tree(weighted_n_node_samples,
                     children_left,
                     children_right,
                     impurity)

    assert np.all(r_single_t(test) == impurity), \
        "static tree's r_single_t should be MSE per node"




def test_r_single_t_classification():
    """
    test for r_single_t for classification examples
    """
    # data creation
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

    tree = random_forest.estimators_[0]
    r_single_t_val = r_single_t(tree)

    assert r_single_t_val.shape == tree.tree_.children_left.shape, \
        "r_single_t output should be the same length as number of nodes"
    parents = calc_parent(tree)

    assert np.all((r_single_t_val >= r_single_t_val[tree.tree_.children_left]) +\
                  (r_single_t_val >= r_single_t_val[tree.tree_.children_right]) >= 1), \
        "parents should have more loss at least one child - in tree"

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

    expected_missclass = np.array([13/(21+13), 5/10,
                                  8/(8+16), 0, 7/(8+7), 1/(7+1), 0])


    test = fake_tree(weighted_n_node_samples,
                     children_left,
                     children_right,
                     value)

    assert np.all(r_single_t(test) == expected_missclass), \
        "static tree's r_single_t should be misclassification rate per node"



def test_calc_parent():
    """
    test for calc_parent
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

    parents = calc_parent(tree)

    assert parents.shape == tree.tree_.children_left.shape, \
        "size dimension of parents should be same as the number of nodes"

    assert np.all(np.floor(parents) == parents), \
        "output of calc_parent should be integer based"

    assert np.all(parents >= -1), \
        "values of parents should be >= -1"

    assert np.max(parents) < tree.tree_.children_left.shape[0], \
        "nodes for parents should be not have values above the max node"

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

    parents = calc_parent(test)

    expected_parents = np.array([-1,0,0,2,2,4,4], dtype = np.int)

    assert np.all(parents == expected_parents), \
        "static test failed to correct results for parent nodes"



def test_initialize_prune_df_classification():
    """
    test for initialize_prune_df for classification examples
    """
    # data creation
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

    tree = random_forest.estimators_[0]

    p_df = _initialize_prune_df(tree)

    assert p_df.shape == (tree.tree_.children_left.shape[0], 8), \
        "pruned_df should have 8 columns and num_nodes rows"

    assert np.all(p_df.columns == ["idx", "R({t})", "R(T_t)",
                                    "c_left", "c_right", "|T_t|",
                                    "parent", "n_obs"]), \
        "pruned_df column names are incorrect (or incorrectly ordered)"

    # static check

    # tree structure:
    # ~upper: left, lower: right~
    #                       num obs     class 1   class 2   |T_t|
    #    |--1                   10          5       5        1
    # -0-|                       34         21      13       4
    #    |   |--3              9            9       0        1
    #    |-2-|                  24          16      8        3
    #        |   |--5         8             7       1        1
    #        |-4-|             15           7       8        2
    #            |--6         7             0       7        1



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
    nn = weighted_n_node_samples
    children_left = np.array([2,-1,4,-1,6,-1,-1], dtype = np.int)
    children_right = np.array([1,-1,3,-1,5,-1,-1], dtype = np.int)
    value = np.array([[21, 13],
                      [5, 5],
                      [16, 8],
                      [9, 0],
                      [7, 8],
                      [7, 1],
                      [0, 7]], dtype = np.float).reshape((-1,1,2))

    T_t = np.array([4,1,3,1,2,1,1], dtype = np.int)

    test = fake_tree(weighted_n_node_samples,
                     children_left,
                     children_right,
                     value)

    r_single_t_val = r_single_t(test)

    r_T_t = r_single_t_val.copy()
    r_T_t[4] = (r_T_t[5]*nn[5] + r_T_t[6]*nn[6])/nn[4]
    r_T_t[2] = (r_T_t[3]*nn[3] + r_T_t[4]*nn[4])/nn[2]
    r_T_t[0] = (r_T_t[1]*nn[1] + r_T_t[2]*nn[2])/nn[0]


    p_df = _initialize_prune_df(test)

    assert np.any(p_df.idx == np.arange(p_df.shape[0], dtype = np.int)), \
        "idx should be ordered 0:num_nodes"

    assert np.all(p_df.loc[:,"|T_t|"] == T_t), \
        "static example T_t's values are incorrect"

    assert np.all(p_df.n_obs == weighted_n_node_samples), \
        "static example n_obs should grab weighted_n_node_samples"

    assert np.all(p_df.loc[:,"R({t})"] == r_single_t_val), \
        "static example: p_df's R({t}) should be same as r_single_t"

    assert np.all(p_df.loc[:,"R(T_t)"] == r_T_t), \
        "static example, p_df's R(T_t) incorrect"

    assert np.all(p_df.c_left == children_left), \
        "static example, children_left incorrectly stored"
    assert np.all(p_df.c_right == children_right), \
        "static example, children_right incorrectly stored"

    assert np.all(p_df.parent ==  calc_parent(test)), \
        "static example, p_df.parent is same as calc_parent"


def test_initialize_prune_df_regression():
    """
    test for initialize_prune_df for regression examples
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

    p_df = _initialize_prune_df(tree)

    assert p_df.shape == (tree.tree_.children_left.shape[0], 8), \
        "pruned_df should have 8 columns and num_nodes rows"

    assert np.all(p_df.columns == ["idx", "R({t})", "R(T_t)",
                                    "c_left", "c_right", "|T_t|",
                                    "parent", "n_obs"]), \
        "pruned_df column names are incorrect (or incorrectly ordered)"

    # static check

    # tree structure:
    # ~upper: left, lower: right~
    #                       num obs  |T_t|
    #    |--1                   10      1
    # -0-|                       34     4
    #    |   |--3              9        1
    #    |-2-|                  24      3
    #        |   |--5         8         1
    #        |-4-|             15       2
    #            |--6         7         1


    class inner_fake_tree():
        def __init__(self, nn, cl, cr, impurity):
            self.weighted_n_node_samples = nn
            self.children_left = cl
            self.children_right = cr
            self.impurity = impurity

    class fake_tree():
        def __init__(self, nn, cl, cr, impurity):
            self.tree_ = inner_fake_tree(nn, cl, cr, impurity)
            self.__class__ = sklearn.tree.tree.DecisionTreeRegressor

    weighted_n_node_samples = np.array([34,10,24,9,15,8,7], dtype = np.int)
    children_left = np.array([2,-1,4,-1,6,-1,-1], dtype = np.int)
    children_right = np.array([1,-1,3,-1,5,-1,-1], dtype = np.int)
    impurity = np.array([.7, .2, .3, .2, .1, 0, 0])

    T_t = np.array([4,1,3,1,2,1,1], dtype = np.int)

    test = fake_tree(weighted_n_node_samples,
                     children_left,
                     children_right,
                     impurity)

    r_single_t_val = r_single_t(test)

    r_T_t = r_single_t_val.copy()
    r_T_t[4] = (r_T_t[5]*nn[5] + r_T_t[6]*nn[6])/nn[4]
    r_T_t[2] = (r_T_t[3]*nn[3] + r_T_t[4]*nn[4])/nn[2]
    r_T_t[0] = (r_T_t[1]*nn[1] + r_T_t[2]*nn[2])/nn[0]


    p_df = _initialize_prune_df(test)

    assert np.any(p_df.idx == np.arange(p_df.shape[0], dtype = np.int)), \
        "idx should be ordered 0:num_nodes"

    assert np.all(p_df.loc[:,"|T_t|"] == T_t), \
        "static example T_t's values are incorrect"

    assert np.all(p_df.n_obs == weighted_n_node_samples), \
        "static example n_obs should grab weighted_n_node_samples"

    assert np.all(p_df.loc[:,"R({t})"] == r_single_t_val), \
        "static example: p_df's R({t}) should be same as r_single_t"

    assert np.all(p_df.loc[:,"R(T_t)"] == r_T_t), \
        "static example, p_df's R(T_t) incorrect"

    assert np.all(p_df.c_left == children_left), \
        "static example, children_left incorrectly stored"
    assert np.all(p_df.c_right == children_right), \
        "static example, children_right incorrectly stored"

    assert np.all(p_df.parent ==  calc_parent(test)), \
        "static example, p_df.parent is same as calc_parent"




