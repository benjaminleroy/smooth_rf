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

def test_generate_oob_info():
    """
    test generate_oob_info

    basic tests - just check structure
    """
    n_obs_class = 100
    X_trained = np.concatenate(
        (np.random.normal(loc = (1,2), scale = .6, size = (n_obs_class,2)),
        np.random.normal(loc = (-1.2, -.5), scale = .6, size = (n_obs_class,2))),
        axis = 0)
    y_trained = np.concatenate((np.zeros(n_obs_class, dtype = np.int),
                         np.ones(n_obs_class, dtype = np.int)))
    # creating a random forest
    n_trees = 5
    rf_reg = sklearn.ensemble.RandomForestRegressor(
                                                    n_estimators = n_trees,
                                                    min_samples_leaf = 1)
    fit_reg = rf_reg.fit(X = np.array(X_trained),
                                  y = y_trained.ravel())
    forest = fit_reg.estimators_

    idx_mat, oob_weights = smooth_rf.generate_oob_info(forest, X_trained)

    # check idx_mat
    cum_tree_size = np.cumsum(np.array([0] +
            [np.sum(t.tree_.children_left == -1) for t in forest]))

    assert np.all(idx_mat < cum_tree_size[1:]) and \
        np.all(idx_mat >= cum_tree_size[:n_trees]), \
        "all indices in idx_mat follow assumptions relative to index in Gamma"+\
        "ordering relative to tree input"

    assert idx_mat.shape == (X_trained.shape[0], n_trees), \
        "shape of idx_mat should be (n_obs, n_trees)"

    # check oob_weights
    assert np.all(oob_weights >= 0) and \
        np.all(oob_weights <= 1), \
        "oob_weights have values betwen 0 and 1"

    #below code allows for never being observed
    assert np.sum(oob_weights) <= X_trained.shape[0] and \
        np.sum(oob_weights) > 0 and \
        np.sum(oob_weights) == np.int(np.sum(oob_weights)) and \
        np.all([np.sum(oob_weights[(n_trees*ob):(n_trees*(ob+1))]) in {0,1}
                                       for ob in np.arange(X_trained.shape[0],
                                                           dtype = np.int)]),\
        "oob_weights should be scaled per observation"

    assert oob_weights.shape == (X_trained.shape[0]*n_trees,), \
        "shape of idx_mat should be (n_obs * n_trees,)"


def test_create_G_H():
    """
    test create_G_H

    semi-static test
    """
    for _ in np.arange(5):
        # assume trees of with # leaves: 3, 2, 1
        n_leaves = np.array([3,2,1])
        n_trees = n_leaves.shape[0]
        n_obs = 20
        K = 5

        idx_mat_prime = np.array([list(np.random.choice(3, size = n_obs)),
                                  list(np.random.choice(2, size = n_obs)),
                                  [0]*n_obs]).T
        idx_mat = idx_mat_prime + np.cumsum([0] + list(n_leaves))[:n_leaves.shape[0]]

        Gamma = np.random.uniform(size = (n_leaves.sum(),K))
        eta = np.array(np.random.uniform(size = (n_leaves.sum(),K)) * 100, dtype = np.int)

        for _ in np.arange(5):
            inner_size = 10
            obs_idx = np.random.choice(n_obs,
                                       size = inner_size,
                                       replace = False)

            G, H = smooth_rf.create_G_H(Gamma, eta, idx_mat, obs_idx)

            assert G.shape == (inner_size*n_trees, K), \
                "shape of G should be (mT, K)"

            assert H.shape == (inner_size*n_trees, K), \
                "shape of G should be (mT, K)"

            assert np.all(Gamma[idx_mat[obs_idx[0],],:] == G[:n_trees,:]), \
                "values in G relative to first obs incorrect"

            assert np.all(eta[idx_mat[obs_idx[0],],:] == H[:n_trees,:]), \
                "values in H relative to first obs incorrect"


def test_calc_y_oob():
    """
    test calc_y_oob

    super basic test
    """

    n_obs_class = 100
    X_trained = np.concatenate(
        (np.random.normal(loc = (1,2), scale = .6, size = (n_obs_class,2)),
        np.random.normal(loc = (-1.2, -.5), scale = .6, size = (n_obs_class,2))),
        axis = 0)
    y_trained = np.concatenate((np.zeros(n_obs_class, dtype = np.int),
                         np.ones(n_obs_class, dtype = np.int))) + 100
    # creating a random forest
    n_trees = 5
    rf_reg = sklearn.ensemble.RandomForestRegressor(
                                                    n_estimators = n_trees,
                                                    min_samples_leaf = 1)
    fit_reg = rf_reg.fit(X = np.array(X_trained),
                                  y = y_trained.ravel())
    forest = fit_reg.estimators_

    idx_mat, oob_weights = smooth_rf.generate_oob_info(forest, X_trained)

    Gamma, eta, _ = smooth_rf.create_Gamma_eta_forest(fit_reg, parents_all=True)

    lamb = np.zeros(Gamma.shape[1])
    lamb[0] = 1

    for _ in np.arange(5):
        obs_idx = np.random.choice(X_trained.shape[0], size = 20, replace = False)
        y_oob, ever_oob = smooth_rf.calc_y_oob(lamb, Gamma, eta, idx_mat, oob_weights, obs_idx)

        assert np.all(y_oob[ever_oob] <= 101.001) and \
            np.all(y_oob[ever_oob] >= 100 -.001), \
            "test case values of y_oob should be around 101 to 100"

        assert (np.sum(ever_oob) == 0) or np.all(y_oob[ever_oob == False] == 0), \
            "every_oob logically relates to y_oob (returns y_oob == 0 "+\
            "when every_oob = False"



def test_calc_y_oob_grad():
    """
    test calc_y_oob_grad
    """
    n_obs_class = 100
    X_trained = np.concatenate(
        (np.random.normal(loc = (1,2), scale = .6, size = (n_obs_class,2)),
        np.random.normal(loc = (-1.2, -.5), scale = .6, size = (n_obs_class,2))),
        axis = 0)
    y_trained = np.concatenate((np.zeros(n_obs_class, dtype = np.int),
                         np.ones(n_obs_class, dtype = np.int))) + 100
    # creating a random forest
    n_trees = 5
    rf_reg = sklearn.ensemble.RandomForestRegressor(
                                                    n_estimators = n_trees,
                                                    min_samples_leaf = 1)
    fit_reg = rf_reg.fit(X = np.array(X_trained),
                                  y = y_trained.ravel())
    forest = fit_reg.estimators_

    idx_mat, oob_weights = smooth_rf.generate_oob_info(forest, X_trained)

    Gamma, eta, _ = smooth_rf.create_Gamma_eta_forest(fit_reg, parents_all=True)

    lamb = np.zeros(Gamma.shape[1])
    lamb[0] = 1

    for _ in np.arange(5):
        obs_idx = np.random.choice(X_trained.shape[0], size = 20, replace = False)
        grad = smooth_rf.calc_y_oob_grad(lamb, Gamma, eta, idx_mat, oob_weights, obs_idx)

        assert grad.shape == (20, Gamma.shape[1]), \
            "gradient should be (m x K)"

        _, ever_oob = smooth_rf.calc_y_oob(lamb, Gamma, eta, idx_mat, oob_weights, obs_idx)

        assert np.sum(ever_oob == False) == 0 or \
            np.all(grad[ever_oob == False, ] == 0), \
            "gradient relative to any y_oob never out of bag is 0"


def test_calc_l2_grad():
    """
    test calc_l2_grad

    super basic (just dimension) check for gradient
    """
    n_obs_class = 100
    X_trained = np.concatenate(
        (np.random.normal(loc = (1,2), scale = .6, size = (n_obs_class,2)),
        np.random.normal(loc = (-1.2, -.5), scale = .6, size = (n_obs_class,2))),
        axis = 0)
    y_trained = np.concatenate((np.zeros(n_obs_class, dtype = np.int),
                         np.ones(n_obs_class, dtype = np.int))) + 100
    # creating a random forest
    n_trees = 5
    rf_reg = sklearn.ensemble.RandomForestRegressor(
                                                    n_estimators = n_trees,
                                                    min_samples_leaf = 1)
    fit_reg = rf_reg.fit(X = np.array(X_trained),
                                  y = y_trained.ravel())
    forest = fit_reg.estimators_

    idx_mat, oob_weights = smooth_rf.generate_oob_info(forest, X_trained)

    Gamma, eta, _ = smooth_rf.create_Gamma_eta_forest(fit_reg, parents_all=True)

    lamb = np.zeros(Gamma.shape[1])
    lamb[0] = 1

    for _ in np.arange(5):
        obs_idx = np.random.choice(X_trained.shape[0], size = 20, replace = False)
        grad = smooth_rf.calc_l2_grad(lamb, Gamma, eta, idx_mat, oob_weights, obs_idx, y_trained)

        assert grad.shape[0] == Gamma.shape[1], \
            "expected dimensions of gradient should be (K,)"


def test_l2_s_grad_for_adam_wrapper_clean():
    """
    test l2_s_grad_for_adam_wrapper
    """
    n_obs_class = 100
    X_trained = np.concatenate(
        (np.random.normal(loc = (1,2), scale = .6, size = (n_obs_class,2)),
        np.random.normal(loc = (-1.2, -.5), scale = .6,
                         size = (n_obs_class,2))),
        axis = 0)
    y_trained = np.concatenate((np.zeros(n_obs_class, dtype = np.int),
                         np.ones(n_obs_class, dtype = np.int))) + 100
    # creating a random forest
    n_trees = 5
    rf_reg = sklearn.ensemble.RandomForestRegressor(
                                                    n_estimators = n_trees,
                                                    min_samples_leaf = 1)
    fit_reg = rf_reg.fit(X = np.array(X_trained),
                                  y = y_trained.ravel())
    forest = fit_reg.estimators_

    idx_mat, oob_weights = smooth_rf.generate_oob_info(forest, X_trained)

    Gamma, eta, _ = smooth_rf.create_Gamma_eta_forest(fit_reg,
                                                        parents_all=True)

    for _ in range(5):
        obs_idx = np.random.choice(X_trained.shape[0], size = 30)
        wrapper_funct = smooth_rf.l2_s_grad_for_adam_wrapper_clean(Gamma, eta,
                                         idx_mat, oob_weights, obs_idx,
                                         y_trained)

        for _ in range(10):
            lamb = np.random.uniform(size = Gamma.shape[1])
            lamb = lamb / lamb.sum()

            g_straight = smooth_rf.calc_l2_grad(lamb, Gamma, eta,
                                               idx_mat, oob_weights, obs_idx,
                                               y_trained)
            g_wrapper = wrapper_funct(lamb)

            assert np.all(g_straight == g_wrapper), \
                "l2 wrapper function should return same values at object it wraps"


def test_smooth_clean_regressor():
    """
    test for smooth_clean- regressor, only runs on example dataset, checks for errs
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
    verbose = False
    parents_all = True
    dist_mat_style = "standard"
    n_steps = 100

    # general check for erroring
    try:
        a,b = smooth_rf.smooth_clean(random_forest,
                 X_trained, y_trained,
                 verbose=verbose,
                 sgd_max_num=n_steps,
                 parents_all=parents_all,
                 dist_mat_style=dist_mat_style)

    except:
        assert False, \
            "error running smoothing_clean for a random forest regressor"

    # sanity check
    a,b = smooth_rf.smooth_clean(random_forest,
                 X_trained, y_trained,
                 verbose=verbose,
                 sgd_max_num=n_steps,
                 sanity_check=True,
                 parents_all=parents_all,
                 dist_mat_style=dist_mat_style)

    no_update_pred = a.predict(X_trained)
    base_pred = random_forest.predict(X_trained)

    assert np.all(no_update_pred == base_pred), \
        "sanity check for rf regressor in smoother failed"

    try:
        a,b = smooth_rf.smooth_clean(random_forest, X_trained, y_trained,
                                    parents_all=parents_all, verbose=verbose,
                                    dist_mat_style=dist_mat_style,
                                    sgd_max_num=n_steps,
                                    adam = {"alpha": .001, "beta_1": .9,
                                            "beta_2": .999,"eps": 1e-8})
    except:
        assert False, \
            "error running smoothing_function for a random forest "+\
            "regressor with adam"


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
    dist_mat_style = "standard"

    # general check for erroring
    try:
        a,b = smooth_rf.smooth_clean(random_forest, X_trained, y_trained,
                                    sgd_max_num=n_steps,
                                    parents_all=parents_all, verbose=verbose,
                                    dist_mat_style=dist_mat_style,
                                    adam={"alpha": .001, "beta_1": .9,
                                            "beta_2": .999,"eps": 1e-8})

    except:
        assert False, \
            "error running smoothing_function for a random forest regressor"



