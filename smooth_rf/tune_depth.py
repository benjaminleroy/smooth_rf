import numpy as np
import pandas as pd

import smooth_rf
import sklearn
import sklearn.ensemble
import scipy
import scipy.sparse

import copy
import progressbar

import smooth_rf
from collections import Counter

def leaf_predicted_values(tree):
    """
    Create dictionary that contains predicted values for leafs in tree
    conditional on constraining max depth.

    Argument:
    ---------
    tree : sklearn style tree (DecisionTreeClassifier or DecisionTreeRegressor)
        grown tree

    Returns:
    --------
    predicted_vals_dict : dict
        dictionary of predicted values for each leaf condititional on
        max depth allowed
    """
    vals = tree.tree_.value
    leaf_decision, _ = smooth_rf.create_decision_per_leafs(tree)
    n_nodes = leaf_decision.shape[1]

    depth_tree = smooth_rf.depth_per_node(tree)
    max_depth = np.int(np.max(depth_tree))

    predicted_vals_dict = dict()

    for max_depth_selected in range(max_depth):

        depth_logic = depth_tree <= max_depth_selected

        leaf_decision_updated = leaf_decision @ \
                                    scipy.sparse.diags(
                                        depth_logic*np.arange(n_nodes),
                                                       dtype=int)

        lowest_viable_node = leaf_decision_updated.max(axis = 1)

        shape_vals = list(vals.shape)
        shape_leaf = shape_vals
        shape_leaf[0] = leaf_decision.shape[0]

        predicted_vals_dict[max_depth_selected] = \
             vals[lowest_viable_node.todense().ravel()].reshape(shape_leaf)

    return predicted_vals_dict

def depth_tune(random_forest, X_trained=None, y_trained=None,
               X_tune=None, y_tune=None, verbose=True,
               resample_tune=False):
    """
    Update a random forest by tuning the maximum optimal depth

    Args:
    ----
    random_forest : sklearn forest
            (sklearn.ensemble.forest.RandomForestRegressor or
            sklearn.ensemble.forest.RandomForestClassifier)
        pre-trained classification or regression based random forest
    X_trained : array (n, p)
        X data array used to create the inputted random_forest. Note that this
        is assumed to be the correct data - and is used if the smoothing is
        preformed with either the oob sample(done by default if X_tune,
        y_tune are None and resample_tune is False), or another bootstrap
        sample (done when resample_tune is True).
        (default is none)
    y_trained : array (n, c)
        y data array used to create the inputted random_forest, (c classes)
        (default is none)
    X_tune : array (m, p)
        X data array to use in smoothing the random forest (default is none)
    y_tune : array (m, c)
        y data array to use in smoothing the random forest (default is none)
    verbose : bool
        logic to show tree analysis process
    resample_tune: bool
        logic to tune / optimize with another bootstrap same from X

    Returns:
    --------
    inner_rf : sklearn forest
        updated random forest (just leave node values are updated) and
        has additional attribute: .loss_vec_depth, which contains the
        loss vector ranging across different maximum depths.
    """
    if type(random_forest) is sklearn.ensemble.RandomForestClassifier:
        rf_type = "class"
    elif type(random_forest) is sklearn.ensemble.RandomForestRegressor:
        rf_type = "reg"
    else:
        ValueError("random_forest needs to be either a " +\
                   "sklearn.ensemble.RandomForestClassifier " +\
                   "or a sklearn.ensemble.RandomForestRegressor")

    if (X_tune is None or y_tune is None) and not resample_tune:
        oob_logic = True
    else:
        oob_logic = False


    if (oob_logic or resample_tune) and \
        (X_trained is None or y_trained is None):
        raise TypeError("X_trained and y_trained need to be inserted for "+\
                        "provided input of X_tune/y_tune and resample_tune "+\
                        "parameters.")

    if oob_logic or resample_tune:
        n_obs_trained = X_trained.shape[0]


    inner_rf = copy.deepcopy(random_forest)
    forest = inner_rf.estimators_

    first_iter = forest
    if verbose:
        bar = progressbar.ProgressBar()
        first_iter = bar(first_iter)

    _, max_depth = smooth_rf.calc_depth_for_forest(random_forest,verbose=False)
    max_depth = np.int(max_depth)

    forest_loss = np.zeros(max_depth)

    for t in first_iter:
        tree = t.tree_

        # observed information
        if oob_logic:
            random_state = t.random_state
            oob_indices = \
                sklearn.ensemble.forest._generate_unsampled_indices(
                                                                 random_state,
                                                                 n_obs_trained)
            X_tune = X_trained[oob_indices,:]
            y_tune = y_trained[oob_indices]

        if resample_tune:
            resample_indices = \
                sklearn.ensemble.forest._generate_sample_indices(None,
                                                                 n_obs_trained)
            X_tune = X_trained[resample_indices,:]
            y_tune = y_trained[resample_indices]

        pred_leaf = leaf_predicted_values(t)

        tree_loss = np.ones(max_depth)*-1
        for i, pred in pred_leaf.items():
            tree.value[tree.children_left == -1] = pred

            if rf_type == "reg":
                yhat = t.predict(X_tune)
                tree_loss[i] = np.sum((y_tune - yhat)**2)
            else:
                yhat = t.predict(X_tune)

                tree_loss[i] = np.sum(yhat != y_tune)

        tree_loss[tree_loss == -1] = tree_loss[i]
        forest_loss += tree_loss

    best_depth = np.int(np.argmin(forest_loss))


    # updating a random forest
    inner_rf = copy.deepcopy(random_forest)

    for t in inner_rf.estimators_:
        tree = t.tree_
        pred_leaf = leaf_predicted_values(t)
        tree.value[tree.children_left == -1] = pred_leaf[best_depth].reshape(tree.value[tree.children_left == -1].shape)

    inner_rf.loss_vec_depth = forest_loss

    return inner_rf



