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

# some built functions libraries
# sys.path.append("../../filament_and_rfdepth/code/functions/")
# import projected_grad_lamb_update as projgrad
import smooth_rf
from smooth_rf import depth_per_node, create_Gamma_eta_forest

##################################
# Base Tree and Forest Structure #
##################################
# Comments:
# this following 3 functions were
# originally developed by me for a different project and then used in a
# class project at CMU, this is don't copied illegally, I just haven't
# linking all the older versions.


def make_Vt_mat(random_forest, data, verbose = True, depth_dict = None):
    """
    makes set of V_{t,\lambda} matrices for a random forest
    """
    n_tree = random_forest.n_estimators
    forest = random_forest.estimators_

    if verbose:
        bar = progressbar.ProgressBar()
        first_iter = bar(np.arange(n_tree))
    else:
        first_iter = range(n_tree)

    dict_out = dict()
    for tree_idx in first_iter:
        if depth_dict is None:
            dict_out[tree_idx] = _make_Vt_mat_tree(forest[tree_idx], data)
        else:
            dict_out[tree_idx] = _make_Vt_mat_tree(forest[tree_idx], data,
                                            depth_vec = depth_dict[tree_idx])

    return dict_out

def _make_Vt_mat_tree(tree, data, depth_vec = None):
    """
    makes a set of V_{t,\lambda} matrices for a specific tree
    in a random forest (index t)
    """
    if depth_vec is None:
        depth_vec = depth_per_node(tree)

    Vt_full = tree.decision_path(data)

    unique_depth_values = np.array(list(dict(Counter(depth_vec)).keys()))
    unique_depth_values.sort()

    dict_out = {i: Vt_full[:,depth_vec == i] for i in unique_depth_values}

    return dict_out

def make_Ut_prime_mat_no_sym(Vt_dict_new, Vt_dict,
    max_depth = None, verbose = True):
    """
    calculates the Ut_prime_not_symmetric set of matrices (for each Vt, Vt_new),
    returns dictionary with Ut_prime array for each lambda

    assume not points are in both areas (otherwise ii entry would be 0)
    """

    assert len(Vt_dict_new) == len(Vt_dict), \
     "Both dictionaries should be the same length (number of trees)."

    n_tree = len(Vt_dict)
    n_obs_train = Vt_dict[list(Vt_dict.keys())[0]][0].shape[0]
    n_obs_test = Vt_dict_new[list(Vt_dict_new.keys())[0]][0].shape[0]

    # max_depth calculation
    if max_depth is None:
        max_depth_value = 0
        for idx in range(n_tree):
            max_depth_value = np.max([max_depth_value,
                                      np.max(list(Vt_dict_new.keys())),
                                      np.max(list(Vt_dict.keys()))])
        max_depth = max_depth_value

    assert np.floor(max_depth) == np.ceil(max_depth), \
     "max_depth is not a integer"

    max_depth = np.int(max_depth)


    if verbose:
        bar = progressbar.ProgressBar()
        first_iter = bar(np.arange(max_depth, dtype = np.int))
    else:
        first_iter = range(max_depth)

    Ut_prime_dict = dict() # by |lambda|


    for lambda_idx in first_iter:
        #Ut_prime_inner_array = sparse.DOK(shape = (n_tree, n_obs, n_obs))

        tt_all = np.zeros(shape = (0,))
        xx_all = np.zeros(shape = (0,))
        yy_all = np.zeros(shape = (0,))
        values_all = np.zeros(shape = (0,))
        for tree_idx in range(n_tree):
            if lambda_idx in Vt_dict[tree_idx].keys():
                Ut = Vt_dict_new[tree_idx][lambda_idx].dot(
                                            Vt_dict[tree_idx][lambda_idx].T
                                                      )

                # Ut_prime_inner_array[tree_idx,:,:] = Ut_prime.toarray()
                xx, yy, values = scipy.sparse.find(Ut)
                tt = np.array([tree_idx]*xx.shape[0], dtype = np.int)

                tt_all = np.concatenate((tt_all, tt))
                xx_all = np.concatenate((xx_all, xx))
                yy_all = np.concatenate((yy_all, yy))
                values_all = np.concatenate((values_all, values))


        Ut_prime_dict[lambda_idx] = sparse.COO(coords = [tt_all, xx_all, yy_all],
                                               data = values_all,
                                               shape = (n_tree, n_obs_test,
                                                                n_obs_train))

    return Ut_prime_dict


def remove_0_from_Ut_prime(Ut_prime_dict):
    """
    removes lambda index 0 from the Ut_prime dictionary and reorders elements
    so that you now have 0 to (|lambdas| - 1) indices.
    """
    Ut_prime_out = dict()

    keys = np.array(list(Ut_prime_dict.keys()))

    assert np.all(np.ceil(keys) == np.floor(keys)), \
     "keys are integer values."

    assert np.any(keys == 0), \
     "at least 1 key needs to be '0'."

    for l_idx in keys[keys != 0]:
        Ut_prime_out[l_idx - 1] = Ut_prime_dict[l_idx ]

    return Ut_prime_out



# Kernel creation
def make_kernel(Ut_prime_dict, lamb_vec = None):
    """
    Makes a kernel from a given Ut_prime_dict and lambda vector.
    """
    if lamb_vec is None:
        lamb_vec = np.ones(len(Ut_prime_dict))
    assert len(Ut_prime_dict) == lamb_vec.shape[0], \
        "error in dimensions of lamb vector relative to Ut_prime_dict -" +\
        "should be same length"

    t_mean = dict()

    for t_idx in Ut_prime_dict.keys():
        t_mean[t_idx] = Ut_prime_dict[t_idx].sum(axis = 0)/\
                            Ut_prime_dict[t_idx].shape[0]

    Kmat = 0 # not useful, but tests don't like Kmat not being defined
    # before the "for" statement below

    for t_idx in t_mean.keys():

        if t_idx == 0:
            K_mat = lamb_vec[t_idx] * t_mean[t_idx]
        else:
            K_mat = K_mat + lamb_vec[t_idx] * t_mean[t_idx]

    return K_mat

# depth distance calculation

def depth_dist(K_mat):
    """
    Calculates the "depth distance" relative to a kernel (this allows for
    any weighting of the depths).
    This function does DD_ij = d(t(x_i)) - d(parent(x_i,x_j)) and is not
    symmetric.
    """
    own_depth = np.diag(K_mat.todense())
    n = own_depth.shape[0]

    assert not np.all(own_depth == 0), \
        "Warning: Inserting kernel that had diagonal depth set to 0, this " +\
        "is not desirable for applying this function."

    own_mat = np.repeat(own_depth, repeats = n, axis = 0).reshape((n,n))

    if type(K_mat) is sparse.coo.core.COO:
        return own_mat - K_mat.todense()
    else:
        return own_mat - K_mat

def categorical_depth_expand(D_mat):
    """
    expects Depth matrix with integer differences
    """

    assert(np.min(D_mat) >= 0)
    z_size = np.max(D_mat) + 1

    xx, yy = np.indices(D_mat.shape)

    full_indices = (D_mat.ravel(), xx.ravel(), yy.ravel())

    s_mat = sparse.coo.COO(coords = full_indices,data = 1,
                           shape = (z_size, D_mat.shape[0], D_mat.shape[1]))

    return s_mat

def max_depth_dist(K_mat = None, DD_mat = None):
    """
    Caculates maximum depth from parent to x_i or x_j

    This function does DD_ij = max{
                                d(t(x_i)) - d(parent(x_i,x_j)),
                                d(t(x_j)) - d(parent(x_i,x_j))}
    and is symmetric.
    """

    assert not np.all([K_mat is None,DD_mat is None]), \
        "K_mat and DD_mat can't both be None"
    if DD_mat is None:
        DD_mat = depth_dist(K_mat)

    n = DD_mat.shape[0]

    return np.max(np.stack((DD_mat, DD_mat.T), axis = 0), axis = 0)

def min_depth_dist(K_mat = None, DD_mat = None):
    """
    Caculates minimum depth from parent to x_i or x_j

    This function does DD_ij = min{
                                d(t(x_i)) - d(parent(x_i,x_j)),
                                d(t(x_j)) - d(parent(x_i,x_j))}
    and is symmetric.
    """
    assert not np.all([K_mat is None,DD_mat is None]), \
        "K_mat and DD_mat can't both be None"
    if DD_mat is None:
        DD_mat = depth_dist(K_mat)
    n = DD_mat.shape[0]

    return np.min(np.stack((DD_mat, DD_mat.T), axis = 0), axis = 0)


# depth distance in a tree
def depth_dist_tree(Vt_dict_single, y):
    min_v = 1
    max_v = 0
    average_dict = dict()
    for d_idx in Vt_dict_single.keys():
        average_inner = []
        for c_idx in np.arange(Vt_dict_single[d_idx].shape[1], dtype = np.int):
            logic_inner = (Vt_dict_single[d_idx][:,c_idx] == 1)
            local_mean = np.mean(y[np.array(logic_inner.todense()).ravel()])
            average_inner.append(local_mean)
            max_v = max(max_v, local_mean)
            min_v = min(min_v, local_mean)

        average_dict[d_idx] = average_inner

    return average_dict, min_v, max_v


# a, mi, ma = depth_dist_tree(Vt_dict[0], y_test)

## maybe plot color as true local mean (this is associated with leaves)

def dist_mat(X, verbose = True):
    n = X.shape[0]
    Dist_mat = np.zeros(shape = (n,n))

    if verbose:
        bar = progressbar.ProgressBar()
        itera = bar(np.arange(n))
    else:
        itera = np.arange(n)

    if len(X.shape) > 1:
        for i in itera:
            for j in np.arange(i,n):
                Dist_mat[i,j] = scipy.spatial.distance.euclidean(X[i,],X[j,])
                Dist_mat[j,i] = Dist_mat[i,j]
    else:
        for i in itera:
            Dist_mat[i,:] = np.abs(X[i]-X)
    return Dist_mat




def depth_vis_pairs(X, Mat, ax):
    # calculate distance between X_i (for now just need 1D)
    n = X.shape[0]
    assert (n, n) == Mat.shape, \
        "error in size assumptions"

    Dist_mat = np.zeros(shape = Mat.shape)
    for i in np.arange(n):
        for j in np.arange(i,n):
            Dist_mat[i,j] = scipy.spatial.distance.euclidean(X[i,],X[j,])
            Dist_mat[j,i] = Dist_mat[i,j]
    # ravel Dist_mat, Mat, plot
    ax.scatter(x = Dist_mat.ravel(), y = Mat.ravel())

    return Dist_mat


#### decision_path relative to nodes ---------------

def decision_path_nodes(children_right, children_left):
    """
    creates a decision path structure relating leaf nodes to tree structure
    """
    x, yy = _decision_list_nodes(children_right, children_left)

    n_leafs = np.sum(children_right == -1,dtype = np.int)
    n_nodes = children_right.shape[0]
    leaf_idx = np.cumsum(children_right == -1) - 1
    xx = leaf_idx[x]
    return sparse.COO(coords = [xx,np.array(yy)],
                      data = [1]*xx.shape[0],
                      shape = (n_leafs, n_nodes))



def _decision_list_nodes(children_right, children_left, idx=0, elders=list()):
    """
    recursive function to do the inner operations in decision_path_nodes
    """

    if children_left[idx] == -1: # leaf node
        n = len(elders) + 1
        return [[idx]*n, elders + [idx]]
    else:
        c_left = _decision_list_nodes(children_right, children_left,
                                        idx=children_left[idx],
                                        elders=elders + [idx])
        c_right = _decision_list_nodes(children_right, children_left,
                                        idx=children_right[idx],
                                        elders=elders + [idx])
        return [c_left[0] + c_right[0],
                c_left[1] + c_right[1]]


def smooth_all(random_forest, X_trained, y_trained, X_tune=None, y_tune=None,
               verbose=True,
               no_constraint=False, sanity_check=False, resample_tune=False,
               parents_all=False):
    """
    creates a smooth random forest (1 lambda set across all trees)

    Args:
    ----
    random_forest : RandomForestRegressor
        pre-trained regression based random forest
    X_trained : array (n, p)
        X data array used to create the inputted random_forest
    y_trained : array (n, )
        y data array used to create the inputted random_forest
    X_tune : array (m, p)
        X data array to use in smoothing the random forest (default is none)
    y_tune : array (m, )
        y data array to use in smoothing the random forest (default is none)
    verbose : bool
        logic to show tree analysis process
    no_constraint : bool
        logic to not constrain the weights
    sanity_check : bool
        logic to do full process but keep same weights
    resample_tune: bool
        logic to tune / optimize with another bootstrap same from X
    parents_all : bool
        logic to instead include all observations with parent of distance k
        away

    Returns:
    --------
    an updated RandomForestRegressor object with values for each node altered
    if desirable

    Comments:
    ---------

    If X_tune and/or y_tune is None then we will optimize each tree with oob
    samples.
    """

    n_obs_trained = X_trained.shape[0]
    eps = 1/n_obs_trained
    numeric_eps = 1e-5

    if (X_tune is None or y_tune is None) and not resample_tune:
        oob_logic = True
    else:
        oob_logic = False



    y_trained = y_trained.ravel()
    #^note that we should probably check that this is correct shape

    inner_rf = copy.deepcopy(random_forest)

    forest = inner_rf.estimators_

    _, max_depth = smooth_rf.calc_depth_for_forest(random_forest,verbose=False)
    max_depth = np.int(max_depth)

    # Gamma_all = np.zeros((0,max_depth + 1))
    obs_y_leaf_all = np.zeros(0)
    obs_weight_non_zero_all = np.zeros(0)

    first_iter = forest
    if verbose:
        bar = progressbar.ProgressBar()
        first_iter = bar(first_iter)


    for t in first_iter:
        tree = t.tree_
        num_leaf = np.sum(tree.children_left == -1)

        # # node associated
        # node_V = decision_path_nodes(tree.children_right, tree.children_left)
        # node_dd = depth_dist(node_V @ node_V.T)
        # node_dd_expand = categorical_depth_expand(node_dd)

        # num_lamb = node_dd_expand.shape[0]
        # # hmmm - to grab the OOB we could do:
        # # _generate_sample_indices
        # # from https://github.com/scikit-learn/scikit-learn/blob/bac89c253b35a8f1a3827389fbee0f5bebcbc985/sklearn/ensemble/forest.py#L78
        # # just need to grab the tree's random state (t.random_state)

        # # trained "in bag observations"
        random_state = t.random_state
        sample_indices = \
            sklearn.ensemble.forest._generate_sample_indices(random_state,
                                                             n_obs_trained)
        # train_V = t.decision_path(X_trained[sample_indices,:])
        # train_V_leaf = train_V[:,tree.children_left == -1]

        # train_count = train_V_leaf.sum(axis = 0) # by column
        # train_weight = train_count / np.sum(train_count)
        # train_value_sum = (train_V_leaf.T @ y_trained[sample_indices])


        # assert np.sum(train_count) == sample_indices.shape[0], \
        #     "incorrect shape match"

        # np.testing.assert_allclose(
        #     np.array(tree.value[tree.children_left == -1,:,:]).ravel(),
        #     np.array(train_value_sum / train_count).ravel(),
        #     err_msg = "weirdly prediction value is expected to be...")

        # # observed information
        if oob_logic:
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

        obs_V = t.decision_path(X_tune)
        obs_V_leaf = obs_V[:,tree.children_left == -1]

        obs_count = obs_V_leaf.sum(axis = 0).ravel() # by column

        #---
        # for clean division without dividing by 0
        obs_count_div = obs_count.copy()
        obs_count_div[obs_count_div == 0] = 1
        #---
        obs_weight = obs_count / np.sum(obs_count)

        #---
        # for clean inverse even if no observations fall into a leaf (hmmm...)
        obs_weight_non_zero = obs_weight
        obs_weight_non_zero[obs_weight_non_zero == 0] = eps
        #---
        obs_y_leaf = (obs_V_leaf.T @ y_tune) / obs_count_div

        # Gamma_num = node_dd_expand @ train_value_sum
        # Gamma_deno = node_dd_expand @ np.array(train_count).ravel()

        # Gamma_deno_div = Gamma_deno.copy()
        # Gamma_deno_div[Gamma_deno_div == 0] = 1

        # Gamma = (Gamma_num / Gamma_deno_div).T

        # if Gamma.shape[1] != (max_depth + 1):
        #     Gamma_structure = np.zeros((Gamma.shape[0], max_depth + 1))
        #     Gamma_structure[:,:Gamma.shape[1]] = Gamma
        # else:
        #     Gamma_structure = Gamma

        # Gamma_all = np.vstack((Gamma_all, Gamma_structure))
        obs_weight_non_zero_all = np.hstack((obs_weight_non_zero_all,
                                    np.array(obs_weight_non_zero).ravel()))
        obs_y_leaf_all = np.hstack((obs_y_leaf_all,
                                    np.array(obs_y_leaf).ravel()))

    ga, et, _ = smooth_rf.create_Gamma_eta_forest(inner_rf,
                                                    verbose=verbose,
                                                    parents_all=parents_all)
    #pdb.set_trace()

    et[et == 0] = eps

    Gamma_all = ga/et

    # optimization:
    G = Gamma_all.T @ \
        scipy.sparse.diags(np.array(obs_weight_non_zero_all).ravel()) @ \
        Gamma_all # (n,n)

    a = 2 * Gamma_all.T @ np.array(obs_y_leaf_all).ravel() # (n,)
    C = np.hstack((1 * np.ones((max_depth + 1,1)),
                   1 * np.identity(max_depth + 1))) # (n, m)
    b = np.array([1] + [0]*(max_depth + 1)) #(m)

    # COMMENT: FOR ERROR: Gamma can have linearly dependent columns...
    # how to think about (pinv?) - should have learned this implimentation

    reattempt = True
    while reattempt:
        reattempt = False
        try:
            opt = quadprog.solve_qp(G = G.astype(np.float),
                                    a = a.astype(np.float),
                                    C = C.astype(np.float),
                                    b = b.astype(np.float),
                                    meq = 1)
        except:
            G = G + np.diag(np.ones(G.shape[0]) * np.finfo(float).eps * 1000)
            reattempt = True

    lamb = opt[0]

    # no constraints
    if no_constraint:
        lamb = np.linalg.inv(G) @ Gamma_all.T @ \
            scipy.sparse.diags(np.array(obs_weight_non_zero_all).ravel()) @ \
            np.array(obs_y_leaf_all).ravel()

    if sanity_check:
        lamb = np.zeros(lamb.shape)
        lamb[0] = 1

    y_leaf_new_all = Gamma_all @ lamb


    start_idx = 0
    for t in forest:
        tree = t.tree_
        num_leaf = np.sum(tree.children_left == -1)


        tree.value[tree.children_left == -1,:,:] = \
            y_leaf_new_all[start_idx:(start_idx + num_leaf)].reshape((-1,1,1))

        start_idx += num_leaf

    inner_rf.lamb = lamb

    return inner_rf
