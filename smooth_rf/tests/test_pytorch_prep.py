import numpy as np
import pandas as pd
import scipy.sparse
import sparse
import sklearn
from sklearn.ensemble import RandomForestRegressor
from collections import Counter
import sys, os

import smooth_rf

def test_create_Gamma_eta_tree_more_regression():
    """
    test for create_Gamma_eta_tree_more, regression tree - standard depth only

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

    G, n, ln, ld, li = smooth_rf.create_Gamma_eta_tree_more(tree)

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

    # new tests
    assert ln.shape[0] == G.shape[0] and ld.shape[0] == G.shape[0] and \
        li.shape[0] == G.shape[0], \
        "leaf based outputs should have same number of leaves and Gamma"

    assert np.all(np.ceil(ln) == ln) and np.all(ln > 0), \
        "leaf counts should be  strictly positive and integers"

    assert np.all(ln ==
        tree.tree_.weighted_n_node_samples[tree.tree_.children_left == -1]), \
        "number of obs in each leaf not matching tree structure"

    assert np.all(np.ceil(ld) == ld) and np.all(ld >= 0), \
        "leaf depth should be positive and integers"

    assert np.all(li >= - 1e-10), \
        "leaf impurity (mse) should be non-negative"

    # static check

   # tree structure:
    # ~upper: left, lower: right~
    #                       num obs   depth
    #    |--1                   10      1
    # -0-|                       34     0
    #    |   |--3              9        2
    #    |-2-|                  24      1
    #        |   |--5         8         3
    #        |-4-|             15       2
    #            |--6         7         3


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
            self.impurity = np.zeros(v.shape[0]) # this isn't a good test

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

    g_static, n_static, ln_static, ld_static, li_static = \
        smooth_rf.create_Gamma_eta_tree_more(test)

    n_expected = np.array([[10,24,0,0],
                           [9,15,10,0],
                           [8,7,9,10],
                           [7,8,9,10]])
    g_expected = np.array([[10,70,0,0],
                           [18,52,10,0],
                           [24,28,18,10],
                           [28,24,18,10]])
    ln_expected = n_expected[:,0]
    ld_expected = np.array([1,2,3,3])

    assert np.all(g_static == g_expected), \
        "static test's Gamma failed to reproduce correct solutions"
    assert np.all(n_static == n_expected), \
        "static test's eta failed to reproduce correct solutions"
    assert np.all(ln_static == ln_expected), \
        "static test's leaf count failed to reproduce correct solutions"
    assert np.all(ld_static == ld_expected), \
        "static test's leaf depth failed to reproduce correct solutions"


def test_create_Gamma_eta_tree_more_classification():
    """
    test for create_Gamma_eta_tree_more, class - standard depth only

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

    G, n, ln, ld, li = smooth_rf.create_Gamma_eta_tree_more(tree)

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

    # new tests
    assert ln.shape[0] == G.shape[1] and ld.shape[0] == G.shape[1] and \
        li.shape[0] == G.shape[1], \
        "leaf based outputs should have same number of leaves and Gamma"

    assert np.all(np.ceil(ln) == ln) and np.all(ln > 0), \
        "leaf counts should be  strictly positive and integers"

    assert np.all(ln ==
        tree.tree_.weighted_n_node_samples[tree.tree_.children_left == -1]), \
        "number of obs in each leaf not matching tree structure"

    assert np.all(np.ceil(ld) == ld) and np.all(ld >= 0), \
        "leaf depth should be positive and integers"

    # static check

   # tree structure:
    # ~upper: left, lower: right~
    #                       num obs     class 1   class 2    depth
    #    |--1                   10          5       5           1
    # -0-|                       34         21      13          0
    #    |   |--3              9            9       0           2
    #    |-2-|                  24          16      8           1
    #        |   |--5         8             7       1           3
    #        |-4-|             15           7       8           2
    #            |--6         7             0       7           3


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

    def gini(vec):
        p = vec / vec.sum()
        return p.T @ (1-p)

    class inner_fake_tree():
        def __init__(self, nn, cl, cr, v):
            self.weighted_n_node_samples = nn
            self.children_left = cl
            self.children_right = cr
            self.value = v
            self.impurity = np.array([gini(v[i,:,:].ravel()) for i in range(v.shape[0])])

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

    g_static, n_static, ln_static, ld_static, li_static  =  \
        smooth_rf.create_Gamma_eta_tree_more(test)

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
    ln_expected = np.array([10,9,8,7])
    ld_expected = np.array([1,2,3,3])
    li_expected = np.array([gini(value[i,:,:].ravel()) for i in range(value.shape[0])])[np.array([1,3,5,6])]


    assert np.all(g_static == g_expected), \
        "static test's Gamma failed to reproduce correct solutions"
    assert np.all(n_static == n_expected), \
        "static test's eta failed to reproduce correct solutions"
    assert np.all(ln_static == ln_expected), \
        "static test's leaf count failed to reproduce correct solutions"
    assert np.all(ld_static == ld_expected), \
        "static test's leaf depth failed to reproduce correct solutions"
    assert np.all(li_static == li_expected), \
        "static test's leaf impurity failed to reproduce correct solutions"


def test_create_Gamma_eta_tree_more_per_regression():
    """
    test for create_Gamma_eta_tree_more_per, reg tree - standard depth only

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

    G, n, ln, ld, li, fd, fi = smooth_rf.create_Gamma_eta_tree_more_per(tree)

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

    # new tests (ln,ld,li)
    assert ln.shape[0] == G.shape[0] and ld.shape[0] == G.shape[0] and \
        li.shape[0] == G.shape[0], \
        "leaf based outputs should have same number of leaves and Gamma"

    assert np.all(np.ceil(ln) == ln) and np.all(ln > 0), \
        "leaf counts should be  strictly positive and integers"

    assert np.all(ln ==
        tree.tree_.weighted_n_node_samples[tree.tree_.children_left == -1]), \
        "number of obs in each leaf not matching tree structure"

    assert np.all(np.ceil(ld) == ld) and np.all(ld >= 0), \
        "leaf depth should be positive and integers"

    assert np.all(li >= - 1e-10), \
        "leaf impurity (mse) should be non-negative"

    # newest tests (fd, fi)
    assert fd.shape == G.shape and fi.shape == G.shape, \
        "shapes of full depth and impurity should make shape of Gamma"

    assert np.all(fd[:,0]  == ld) and np.all(np.ceil(fd) == fd) and \
        np.all(fd >= 0), \
        "full depth shape should mirror leaf depth structure"

    assert np.all(fi[:,0] == li) and np.all(fi >= - 1e-10), \
        "full impurity (mse) should mirror leaf impurity structure"

    # for c_idx in range(fi.shape[1] - 1):
    #     assert np.all(fi[:,c_idx] - fi[:,c_idx + 1] <= 1e-10), \
    #         "impurity should be increasing (mse)"

    # static check

   # tree structure:
    # ~upper: left, lower: right~
    #                       num obs   depth
    #    |--1                   10      1
    # -0-|                       34     0
    #    |   |--3              9        2
    #    |-2-|                  24      1
    #        |   |--5         8         3
    #        |-4-|             15       2
    #            |--6         7         3


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
            self.impurity = np.zeros(v.shape[0]) # this isn't a good test

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

    g_static, n_static, ln_static, ld_static, li_static, \
        fd_static, fi_static = \
        smooth_rf.create_Gamma_eta_tree_more_per(test)

    n_expected = np.array([[10,24,0,0],
                           [9,15,10,0],
                           [8,7,9,10],
                           [7,8,9,10]])
    g_expected = np.array([[10,70,0,0],
                           [18,52,10,0],
                           [24,28,18,10],
                           [28,24,18,10]])
    ln_expected = n_expected[:,0]
    ld_expected = np.array([1,2,3,3])
    fd_expected = np.array([[1,0,0,0],
                            [2,1,0,0],
                            [3,2,1,0],
                            [3,2,1,0]])

    assert np.all(g_static == g_expected), \
        "static test's Gamma failed to reproduce correct solutions"
    assert np.all(n_static == n_expected), \
        "static test's eta failed to reproduce correct solutions"
    assert np.all(ln_static == ln_expected), \
        "static test's leaf count failed to reproduce correct solutions"
    assert np.all(ld_static == ld_expected), \
        "static test's leaf depth failed to reproduce correct solutions"
    assert np.all(fd_static == fd_expected), \
        "static test's full depth failed to reproduce correct solutions"



def test_create_Gamma_eta_tree_more_per_classification():
    """
    test for create_Gamma_eta_tree_more, class tree - standard depth only

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

    G, n, ln, ld, li, fd, fi = smooth_rf.create_Gamma_eta_tree_more_per(tree)

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

    # new tests
    assert ln.shape[0] == G.shape[1] and ld.shape[0] == G.shape[1] and \
        li.shape[0] == G.shape[1], \
        "leaf based outputs should have same number of leaves and Gamma"

    assert np.all(np.ceil(ln) == ln) and np.all(ln > 0), \
        "leaf counts should be  strictly positive and integers"

    assert np.all(ln ==
        tree.tree_.weighted_n_node_samples[tree.tree_.children_left == -1]), \
        "number of obs in each leaf not matching tree structure"

    assert np.all(np.ceil(ld) == ld) and np.all(ld >= 0), \
        "leaf depth should be positive and integers"

    # newest tests (fd, fi)
    assert fd.shape == G.shape[1:] and fi.shape == G.shape[1:], \
        "shapes of full depth and impurity should make shape of Gamma"

    assert np.all(fd[:,0]  == ld) and np.all(np.ceil(fd) == fd) and \
        np.all(fd >= 0), \
        "full depth shape should mirror leaf depth structure"

    assert np.all(fi[:,0] == li), \
        "full impurity (gini) should mirror leaf impurity structure"


    # static check

   # tree structure:
    # ~upper: left, lower: right~
    #                       num obs     class 1   class 2    depth
    #    |--1                   10          5       5           1
    # -0-|                       34         21      13          0
    #    |   |--3              9            9       0           2
    #    |-2-|                  24          16      8           1
    #        |   |--5         8             7       1           3
    #        |-4-|             15           7       8           2
    #            |--6         7             0       7           3


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

    def gini(vec):
        p = vec / vec.sum()
        return p.T @ (1-p)

    class inner_fake_tree():
        def __init__(self, nn, cl, cr, v):
            self.weighted_n_node_samples = nn
            self.children_left = cl
            self.children_right = cr
            self.value = v
            self.impurity = np.array([gini(v[i,:,:].ravel()) for i in range(v.shape[0])])

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

    g_static, n_static, ln_static, ld_static, li_static, \
        fd_static, fi_static =  \
        smooth_rf.create_Gamma_eta_tree_more_per(test)

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
    ln_expected = np.array([10,9,8,7])
    ld_expected = np.array([1,2,3,3])
    fd_expected = np.array([[1,0,0,0],
                            [2,1,0,0],
                            [3,2,1,0],
                            [3,2,1,0]])
    li_expected = np.array([gini(value[i,:,:].ravel()) for i in range(value.shape[0])])[np.array([1,3,5,6])]
    li_expected2 = np.array([gini(value[i,:,:].ravel()) for i in range(value.shape[0])])[np.array([0,2,4,4])]
    li_expected3 = np.array([gini(value[i,:,:].ravel()) for i in range(value.shape[0])])[np.array([0,0,2,2])]
    li_expected4 = np.array([gini(value[i,:,:].ravel()) for i in range(value.shape[0])])[np.array([0,0,0,0])]

    fi_expected = np.array([li_expected,li_expected2,li_expected3,li_expected4],
                           ).T

    assert np.all(g_static == g_expected), \
        "static test's Gamma failed to reproduce correct solutions"
    assert np.all(n_static == n_expected), \
        "static test's eta failed to reproduce correct solutions"
    assert np.all(ln_static == ln_expected), \
        "static test's leaf count failed to reproduce correct solutions"
    assert np.all(ld_static == ld_expected), \
        "static test's leaf depth failed to reproduce correct solutions"
    assert np.all(li_static == li_expected), \
        "static test's leaf impurity failed to reproduce correct solutions"
    assert np.all(fd_static == fd_expected), \
        "static test's full depth failed to reproduce correct solutions"
    assert np.all(fi_static == fi_expected), \
        "static test's full impurity failed to reproduce correct solutions"




def test_depth_per_node_plus_parent():
    """
    test depth_per_node_plus_parent on random forest tree
    Tests for:
    1) depth_per_node function makes sure all children are 1 (and only 1) level
        deeper
    2) structure relative to parent_mat
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

    predicted_depth, parent_mat = smooth_rf.depth_per_node_plus_parent(tree)

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

    for c_idx in range(parent_mat.shape[1] - 1):
        assert np.all(parent_mat[:,c_idx] - parent_mat[:,c_idx + 1] >= 0), \
            "parents should naturally have lower index than children " + \
            "(error in parent_mat output)"
    assert np.all((parent_mat > 0).sum(axis = 1) == predicted_depth), \
        "parent_mat rows should have same number of non-zero entries "+\
        "as the depth value"

def test_create_Gamma_eta_forest_more_regression():
    """
    test create_Gamma_eta_forest_more, regression forests - standard depth only

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

    g, n, t, ln, ld, li, fd, fi = \
        smooth_rf.create_Gamma_eta_forest_more(random_forest)

    assert g.shape == n.shape, \
        "Gamma and eta matrices are not the correct shared size"
    assert g.shape[0] == t.shape[0], \
        "the tree index vector doesn't have the correct number of observations"

    # new checks
    assert t.shape == ln.shape and t.shape == ld.shape and t.shape == li.shape,\
        "the leaf number, depth, or impurity don't have the correct dim"

    assert g.shape == fd.shape and g.shape == fi.shape, \
        "the full depth or impurity doens't have the correct dim"
    # ----

    assert np.all(
        np.array(list(dict(Counter(t)).keys())) == np.arange(n_tree)),\
        "tree index doesn't contain expected tree index values"

    for t_idx, tree in enumerate(random_forest.estimators_):
        max_depth_range = np.int(np.max(smooth_rf.depth_per_node(tree)) + 1)
        G_tree, n_tree, ln_tree, ld_tree, li_tree, fd_tree, fi_tree = \
            smooth_rf.create_Gamma_eta_tree_more_per(tree)


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

        # new checks
        assert np.all(ln_tree == ln[t==t_idx]), \
            "attributes in leaf number should match the base function"
        assert np.all(ld_tree == ld[t==t_idx]), \
            "attributes in leaf depth should match the base function"
        assert np.all(li_tree == li[t==t_idx]), \
            "attributes in leaf impurity should match the base function"
        assert np.all(ln_tree == ln[t==t_idx]), \
            "attributes in leaf number should match the base function"
        assert np.all(fd_tree == fd[t==t_idx,:][:,:max_depth_range]), \
            "attributes in full depth should match the base function"
        assert np.all(fi_tree == fi[t==t_idx,:][:,:max_depth_range]), \
            "attributes in full impurity should match the base function"




def test_create_Gamma_eta_forest_more_classification():
    """
    test create_Gamma_eta_forestmore, class forests - standard depth only

    compares to what is expected to be returned from create_Gamma_eta_tree -
    mostly just structurally
    """
    n = 200
    n_tree = 10
    min_size_leaf = 1

    X = np.random.uniform(size = (n, 510), low = -1,high = 1)
    y = 10 * np.sin(np.pi * X[:,0]*X[:,1]) + 20 * ( X[:,2] - .5)**2 +\
        10 * X[:,3] + 5 * X[:,4] + np.random.normal(size = n)

    y_cat = np.array(pd.cut(y, bins = 5, labels = np.arange(5, dtype = np.int)),
                     dtype = np.int)

    y = y_cat

    num_classes = len(Counter(y_cat).keys())

    rf_class = sklearn.ensemble.RandomForestClassifier(n_estimators = n_tree,
                                            min_samples_leaf = min_size_leaf)
    random_forest = rf_class.fit(X = X,
                                 y = y.ravel())

    g, n, t, ln, ld, li, fd, fi = \
        smooth_rf.create_Gamma_eta_forest_more(random_forest)

    assert g.shape[1:] == n.shape, \
        "Gamma and eta matrices are not the correct shared size"
    assert g.shape[1] == t.shape[0], \
        "the tree index vector doesn't have the correct number of observations"
    assert g.shape[0] == num_classes, \
        "Gamma matrix dimensions don't match the number of classes correctly"
    # new checks
    assert t.shape == ln.shape and t.shape == ld.shape and t.shape == li.shape,\
        "the leaf number, depth, or impurity don't have the correct dim"

    assert g.shape[1:] == fd.shape and g.shape[1:] == fi.shape, \
        "the full depth or impurity doens't have the correct dim"
    # ----


    assert np.all(
        np.array(list(dict(Counter(t)).keys())) == np.arange(n_tree)),\
        "tree index doesn't contain expected tree index values"

    for t_idx, tree in enumerate(random_forest.estimators_):
        max_depth_range = np.int(np.max(smooth_rf.depth_per_node(tree)) + 1)
        G_tree, n_tree, ln_tree, ld_tree, li_tree, fd_tree, fi_tree = \
            smooth_rf.create_Gamma_eta_tree_more_per(tree)

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

        # new checks
        assert np.all(ln_tree == ln[t==t_idx]), \
            "attributes in leaf number should match the base function"
        assert np.all(ld_tree == ld[t==t_idx]), \
            "attributes in leaf depth should match the base function"
        assert np.all(li_tree == li[t==t_idx]), \
            "attributes in leaf impurity should match the base function"
        assert np.all(ln_tree == ln[t==t_idx]), \
            "attributes in leaf number should match the base function"
        assert np.all(fd_tree == fd[t==t_idx,:][:,:max_depth_range]), \
            "attributes in full depth should match the base function"
        assert np.all(fi_tree == fi[t==t_idx,:][:,:max_depth_range]), \
            "attributes in full impurity should match the base function"

