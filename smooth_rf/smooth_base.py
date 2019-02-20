# Gamma = ny (per level)
# (m x k)
# eta = n (per level)
# (m x k)

# eta_fill = eta @ lamb
# (m x 1)

# Gamma_fill = Gamma @ lamb
# (m x 1)


# grad = - 2 residuals^t grad_yhat

# grad_yhat[l] = ( Gamma[l,]*eta_fill[l] - eta[l,]*Gamma_fill[l] ) / eta_fill[l]**2


# grad_yhat = scipy.sparse.diag(1/eta_fill) @ Gamma - \
#                 scipy.sparse.diag(Gamma_fill/eta_fill**2)

import numpy as np
import pandas as pd
import scipy.sparse
import sparse
import progressbar
import copy
import sklearn.ensemble
import sklearn
import matplotlib.path as mpltPath
import pdb

##################################
# Base Tree and Forest Structure #
##################################
# Comments:
# this following 2 functions "depth_per_node" and "calc_depth_for_forest" were
# originally developed by me for a different project and then used in a
# class project at CMU, this is don't copied illegally, I just haven't
# linking all the older versions. Additionally, function "prox_project" was
# originally writeen for the class project at CMU.

def depth_per_node(tree):
    """
    calculcates depth per node in binary tree

    Arguments:
    ----------
    tree :
        tree object with the same structure as
        `sklearn.ensemble.DecisionTreeClassifier`

    Returns:
    --------
    depth_vec : int array
        vector of depth for each node
    """
    c_left  = tree.tree_.children_left
    c_right = tree.tree_.children_right
    T = len(c_left)

    depth_vec = np.zeros(T)
    for split in np.arange(T, dtype = np.int):
        if c_left[split] != -1:
            depth_vec[c_left[split]] += depth_vec[split] + 1
        if c_right[split] != -1:
            depth_vec[c_right[split]] += depth_vec[split] + 1

    return depth_vec

def calc_depth_for_forest(input_forest, verbose=True):
    """
    calculates depth values for all nodes per each tree, and calculates forest
    maximum depth

    Arguments:
    ----------
    input_forest :
        forest object in the format `sklearn.ensemble.RandomForestClassifier`
        The tree needs to have been grown.
    verbose : bool
        Controls the verbosity when calculating

    Returns:
    --------
    dict_out : dict
        Dictionary (indexed by tree index) of arrays that contains depth for
        each node for the given tree.
    max_depth : int
        Maximum depth of all trees in the forest
    """

    n_tree = input_forest.n_estimators
    forest = input_forest.estimators_

    if verbose:
        bar = progressbar.ProgressBar()
        first_iter = bar(np.arange(n_tree))
    else:
        first_iter = range(n_tree)

    dict_out = dict()
    max_depth = 0
    for tree_idx in first_iter:
        dict_out[tree_idx] = depth_per_node(forest[tree_idx])
        max_depth = np.max([max_depth, np.max(dict_out[tree_idx])])

    return dict_out, max_depth

def prox_project(y):
    """
    Projection function onto the space defined by {x:x_i >= 0, sum_i x_i =1}.
    Arguments:
    ----------
    y : array (n, )
        array vector to be projected into the above space
    Returns:
    --------
    y_prox : array (n, )
        y projected into the above space
    Notes:
    ------
    This algorithm comes from
    - https://arxiv.org/pdf/1101.6081.pdf
    - https://www.mathworks.com/matlabcentral/fileexchange/30332-projection-onto-simplex
    """
    m = y.shape[0]
    bget = False

    s = np.sort(y)[::-1]
    tmpsum = 0

    for idx in np.arange(m-1):
        tmpsum = tmpsum + s[idx]
        tmax = (tmpsum -1)/(idx + 1)

        if tmax >= s[idx + 1]:
            bget = True
            break

    if not bget:
        tmax = (tmpsum + s[m-1] - 1)/m

    y_prox = (y-tmax)*(y-tmax > 0)

    return y_prox

# start of new code ---------------

def create_decision_per_leafs(tree):
    """
    Create a decision path matrix for leaf nodes - relative to all nodes of
    single tree

    Argument:
    ---------
    tree : sklearn style tree object (has attribute .tree_)
        generated tree to create decision path matricies from
    Returns:
    --------
    v_leaf : scipy.sparse matrix
        decision path matrix for all leaves in tree
    v_all : scipy.sparse matrix
        decision path matrix for all nodes in tree
    """
    t = tree.tree_
    cr = t.children_right
    cl = t.children_left

    n_nodes = cr.shape[0]

    storage = np.zeros((n_nodes,n_nodes), dtype = np.int)
    for n_idx in np.arange(n_nodes):
        storage[n_idx,n_idx] = 1
        if cr[n_idx] != -1:
            storage[cr[n_idx], storage[n_idx,:] == 1] = 1
            storage[cl[n_idx], storage[n_idx,:] == 1] = 1

    v_all = scipy.sparse.coo.coo_matrix(storage)
    v_leaf = scipy.sparse.coo.coo_matrix(storage[cr == -1,:])

    return v_leaf, v_all


def create_distance_mat_leaves(tree = None,
                               decision_mat_leaves = None,
                               style = ["standard", "max", "min"]):
    """
    create inner-tree based distance matrix for leaves from a tree or
    precomputed decision matrix for the leaves. The inner-tree distance is
    defined as in the *details* section.

    Arguments:
    ----------
    tree : sklearn style tree
        tree to create distance matrix relative to
    decision_mat_leaves : array (n_leaves, n_nodes)
        decision_matrix relative to leaves of a tree (columns have all nodes,
        whereas rows are just relative to the leaves)
    style : string
        style of inner-tree distance to use, see *details* for more

    Details:
    --------
    For the different styles of inner-tree distance, there is currently 3
    options: "standard", "max", and "min".

    1. For "standard", we define the inner-tree distance as:

        D_ij = depth(obs_i) - depth(parent(i,j)),

        note that this "distance" is *not* symmetric - and should really be
        thought of as a semi-metric.

    2. For "max", we define the inner-tree distance as:

        D_ij^{max} = max( D_ij , D_ji )

    3. For "min", we define the inner-tree distance as:

        D_ij^{min} = min( D_ij , D_ji )

    Returns:
    --------
    distance_matrix : array (n_leaves, n_leaves)
        non-symetric "distance" matrix relative to leaves in the tree - to be
        read "distance from row leave to column leave"

    """


    if tree is None and decision_mat_leaves is None:
        return
    if decision_mat_leaves is None:
        decision_mat_leaves, _ = create_decision_per_leafs(tree)

    if (type(style) is list) and (len(style) > 1):
        style = style[0]

    Q = decision_mat_leaves @ decision_mat_leaves.T

    if type(Q) is scipy.sparse.coo.coo_matrix or \
        type(Q) is scipy.sparse.csr.csr_matrix:
        d = np.diagonal(Q.todense())
    else:
        d = np.diagonal(Q)

    standard_distance = (d - Q.T).T

    if style == "standard":
        return standard_distance
    elif style == "max":
        standard_d_T = standard_distance.T
        both = np.append(np.array(standard_distance).reshape(
                                        (standard_distance.shape[0],
                                         standard_distance.shape[1],
                                         1)),
                         np.array(standard_d_T).reshape(
                                        (standard_d_T.shape[0],
                                         standard_d_T.shape[1],
                                         1)), axis = 2)
        return np.max(both, axis = 2)
    elif style == "min":
        standard_d_T = standard_distance.T
        both = np.append(np.array(standard_distance).reshape(
                                        (standard_distance.shape[0],
                                         standard_distance.shape[1],
                                         1)),
                         np.array(standard_d_T).reshape(
                                        (standard_d_T.shape[0],
                                         standard_d_T.shape[1],
                                         1)), axis = 2)
        return np.min(both, axis = 2)
    else:
        ValueError("style parameter needs to be 1 of the 3 choices.")


def create_Gamma_eta_tree(tree,
                      dist_mat_leaves=None,
                      parents_all=False,
                      dist_mat_style=["standard", "max", "min"]):
    """
    creates the Gamma and eta matrices for a single tree, where these two
    matrices are defined:

    Gamma_il = sum_j II(D_ij = l) n_j y_j
    eta_il = sum_j II(D_ij = l) n_j

    where n_j, y_j are the number of observations in leaf j and the predicted
    value associated with  leaf j for regression, or where y_j = p_j, a
    probably vector associated with leaf j in classification. Note that D_ij
    is the tree based distance between leaf i and j.

    Arguments:
    ----------
    tree : sklearn style tree (DecisionTreeClassifier or DecisionTreeRegressor)
        grown tree to create distance matrix relative to
    dist_mat_leaves : array (n_leaves, n_leaves)
        non-symetric "distance" matrix relative to leaves in the tree - to be
        read "distance from row leave to column leave" The inner-tree
        "distance" has 3 options to be defined (as described by
        create_distance_mat_leaves)
    parents_all : bool
        logic to instead include all observations with parent of distance k
        away
    dist_mat_style : string
        style of inner-tree distance to use, see *details* in the
        create_distance_mat_leaves doc-string.

    Returns:
    --------
    Gamma : array (n_leaves, maximum depth of tree + 1)
            (number of classes, n_leaves, maximum depth of tree + 1)
        see description above
    eta : array (n_leaves, maximum depth of tree + 1)
        see description above
    """

    if type(tree) is sklearn.tree.tree.DecisionTreeRegressor:

        n_leaf_original = tree.tree_.weighted_n_node_samples[
                                    tree.tree_.children_left == -1]
        yhat_leaf_original = tree.tree_.value.ravel()[
                                    tree.tree_.children_left == -1]
        ny_leaf_original = n_leaf_original * yhat_leaf_original

    elif type(tree) is sklearn.tree.tree.DecisionTreeClassifier:

        n_leaf_original = tree.tree_.weighted_n_node_samples[
                                tree.tree_.children_left == -1]
        ny_leaf_original = tree.tree_.value[
                                tree.tree_.children_left == -1,:,:].reshape(
                                                    n_leaf_original.shape[0],
                                                    tree.tree_.value.shape[-1])
        # yhat_leaf_original = np.diag(1/n_leaf_original) @ ny_leaf_original

    else:
        ValueError("tree needs either be a "+\
                   "sklearn.tree.tree.DecisionTreeClassifier "+\
                   "or a sklearn.tree.tree.DecisionTreeRegressor")

    ##############################
    ## Distance Matrix Creation ##
    ##############################

    if dist_mat_leaves is None:
        dist_mat_leaves = create_distance_mat_leaves(tree,
                                                     style = dist_mat_style)

    # creating a 3d sparse array
    xx_all = np.zeros(shape = (0,))
    yy_all = np.zeros(shape = (0,))
    kk_all = np.zeros(shape = (0,))


    for k_idx in np.arange(np.min(dist_mat_leaves), np.max(dist_mat_leaves)+1):
        if parents_all:
            xx, yy = np.where(dist_mat_leaves <= k_idx)
        else:
            xx, yy = np.where(dist_mat_leaves == k_idx)

        xx_all = np.concatenate((xx_all, xx))
        yy_all = np.concatenate((yy_all, yy))
        kk_all = np.concatenate((kk_all,
                                 np.array([k_idx]*np.int(xx.shape[0]),
                                          dtype = np.int)
                                 ))

    inner_sparse = sparse.COO(
                              coords = [kk_all, xx_all, yy_all],
                              data = 1,
                              shape = (np.max(dist_mat_leaves)+1,
                                       dist_mat_leaves.shape[0],
                                       dist_mat_leaves.shape[1]))
    Gamma = (inner_sparse @ ny_leaf_original).T
    eta = (inner_sparse @ n_leaf_original).T

    return Gamma, eta


def create_Gamma_eta_forest(forest, verbose=False, parents_all=False,
                            dist_mat_style=["standard", "max", "min"]):
    """
    creates the Gamma and eta matrices for a forest (aka set of trees, where
    these two matrices are defined (per tree):

    Gamma_il = sum_j II(D_ij = l) n_j y_j
    eta_il = sum_j II(D_ij = l) n_j

    where n_j, y_j are the number of observations in leaf j and the predicted
    value associated with with leaf j for regression, or where y_j = p_j, a
    probably vector associated with leaf j in classification. Note that
    D_ij is the tree based distance between leaf i and j.

    Arguments:
    ----------
    forest : sklearn forest (sklearn.ensemble.forest.RandomForestRegressor or
             sklearn.ensemble.forest.RandomForestClassifier)
        grown forest
    verbose : bool
        logic to show tree analysis process
    parents_all : bool
        logic to instead include all observations with parent of distance k
        away
   dist_mat_style : string
        style of inner-tree distance to use, see *details* in the
        create_distance_mat_leaves doc-string.

    Returns:
    --------
    Gamma_all : array (sum_{t=1}^T n_leaves(t), maximum depth of forest + 1) or
                (num_classes, sum_{t=1}^T n_leaves(t),
                 maximum depth of forest + 1)
        A horizontal stacking of Gamma matrices as defined above for all trees
        in the forest. (Regression and Classification is different)
    eta_all : array (sum_{t=1}^T n_leaves(t), maximum depth of forest + 1)
        A horizontal stacking of eta matrices as defined above for all trees
        in the forest.
    t_idx_all : int array (sum_{t=1}^T n_leaves(t)
        integer array will holds which tree in the forest the associated row
        of Gamma_all or eta_all comes from
    """

    _, max_depth = calc_depth_for_forest(forest, verbose = False)

    if type(forest) is sklearn.ensemble.forest.RandomForestRegressor:
        Gamma_all = np.zeros(shape=(0, np.int(max_depth + 1)))
        g_idx = 1
    elif type(forest) is sklearn.ensemble.forest.RandomForestClassifier:
        num_class = forest.n_classes_
        Gamma_all = np.zeros(shape=(num_class, 0, np.int(max_depth + 1)))
        g_idx = 2
    else:
        ValueError("random_forest needs to be either a " +\
                   "sklearn.ensemble.forest.RandomForestClassifier " +\
                   "or a sklearn.ensemble.forest.RandomForestRegressor")

    eta_all = np.zeros(shape = (0, np.int(max_depth + 1)))
    t_idx_all = np.zeros(shape = (0,))

    first_iter = enumerate(forest.estimators_)

    if verbose:
        bar = progressbar.ProgressBar()
        first_iter = bar(list(first_iter))

    for t_idx, tree in first_iter:
        g, n = create_Gamma_eta_tree(tree, parents_all=parents_all,
                                     dist_mat_style=dist_mat_style)
        tree_n_leaf = g.shape[g_idx - 1]
        if g.shape[g_idx] != Gamma_all.shape[g_idx]:
            diff = Gamma_all.shape[g_idx] - g.shape[g_idx]

            if g_idx == 1: # regressor
                g = np.hstack((g, np.zeros((tree_n_leaf, diff))))
            else: # classifier
                g = np.concatenate((g,
                                      np.zeros((num_class,
                                                tree_n_leaf,
                                                diff))),
                                    axis = 2)

            n = np.hstack((n, np.zeros((tree_n_leaf, diff))))

        if g_idx == 1: # regressor
            Gamma_all = np.concatenate((Gamma_all, g))
        else: # classifier
            Gamma_all = np.concatenate((Gamma_all, g), axis = 1)

        eta_all = np.concatenate((eta_all, n))
        t_idx_all = np.concatenate((t_idx_all,
                                    t_idx * np.ones(tree_n_leaf,
                                                    dtype = np.int)))

    return Gamma_all, eta_all, t_idx_all

def take_gradient(y, Gamma, eta, weights, lamb):
    """
    Calculates gradient of l2 norm for y-yhat (|y-yhat|^2_2)
    where

    yhat_l = sum_k lam_k sum_j II(D_ij = k) n_j y_j
            ----------------------------------------
             sum_k lam_k sum_j II(D_ij = k) n_j

    Arguments:
    ----------
    y : array (m,)
        true y-values (the average) per node of the tree
    Gamma : array (m, k)
        a matrix (rows for leafs, columns for depth) where each value

        Gamma[i,k]  = sum_l II(D_il = k) n_l y_l

        (i.e. sum of training observations y values for those obs with tree
        dist k away from new node i)
    eta : array (m, k)
        a matrix (rows for leafs, columns for depth) where each value

        eta[i,k]  = sum_l II(D_il = k) n_l

        (i.e. the total number of training observations that are tree dist k
        away from the new node i)
    weights : array (m,)
        weights associated with node i. For our analysis this weight = number
        of observations in the ith node
    lamb : array (k,)
        weights associated with how important different distance away from the
        true node should effect the prediction
    """
    Gamma_fill = Gamma @ lamb
    eta_fill = eta @ lamb
    eta_fill[eta_fill == 0] = 1 # to avoid divide by 0.
    residuals = y - Gamma_fill / eta_fill

    assert np.any(eta_fill != 0), \
        "some eta * lamb are zero - which doesn't make sense - "+\
        "try all positive lamb" +\
        "\n Ben: this may happen now - when using global structure"

    grad_yhat = scipy.sparse.diags(1/eta_fill) @ Gamma - \
                    scipy.sparse.diags(Gamma_fill/eta_fill**2) @ eta

    grad = -2 * residuals.T @ scipy.sparse.diags(weights) @ grad_yhat

    return grad


def calc_cost(y, Gamma, eta, weights, lamb):
    """
    l2 cost
    """
    Gamma_fill = Gamma @ lamb
    eta_fill = eta @ lamb
    eta_fill[eta_fill == 0] = 1 # to avoid divide by 0.
    residuals = y - Gamma_fill / eta_fill

    return np.sum( (residuals**2) * weights)


def smooth_regression_old(random_forest, X_trained=None, y_trained=None,
               X_tune=None, y_tune=None, verbose=True,
               no_constraint=False, sanity_check=False,
               resample_tune=False,
               subgrad_max_num=1000, subgrad_t_fix=1,
               all_trees=False,
               initial_lamb_seed=None,
               parents_all=False,
               distance_style=["standard","max", "min"]):
    """
    creates a smooth random forest (1 lambda set across all trees)

    this version uses the scaling relative to each observation

    Args:
    ----
    random_forest : RandomForestRegressor
        pre-trained regression based random forest
    X_trained : array (n, p)
        X data array used to create the inputted random_forest. Note that this
        is assumed to be the correct data - and is used if the smoothing is
        preformed with either the oob sample(done by default if X_tune,
        y_tune are None and resample_tune is False), or another bootstrap
        sample (done when resample_tune is True).
        (default is none)
    y_trained : array (n, )
        y data array used to create the inputted random_forest
        (default is none)
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
    subgrad_max_num : int
        number of steps to take for the subgradient optimization
    subgrad_t_fix : scalar
        value for fixed t step size for gradient descent
    all_trees : bool
        logic to use all trees (and therefore do full gradient descent)
    initial_lamb_seed : scalar
        initial value for seed (default is None) to randomly select the
        starting point of lambda
    parents_all : bool
        logic to instead include all observations with parent of distance k
        away
   distance_style : string
        style of inner-tree distance to use, see *details* in the
        create_distance_mat_leaves doc-string.
    Returns:
    --------
    an updated RandomForestRegressor object with values for each node altered
    if desirable

    Comments:
    ---------

    If X_tune and/or y_tune is None then we will optimize each tree with oob
    samples.
    """

    #n_obs_trained = X_trained.shape[0]
    #eps = 1/n_obs_trained
    #numeric_eps = 1e-5

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
    inner_rf2 = copy.deepcopy(random_forest)

    forest = inner_rf.estimators_
    forest2 = inner_rf2.estimators_

    _, max_depth = calc_depth_for_forest(random_forest,verbose=False)
    max_depth = np.int(max_depth)

    # getting structure from built trees
    Gamma, eta, t_idx_vec = create_Gamma_eta_forest(random_forest,
                                                verbose=verbose,
                                                parents_all=parents_all,
                                                dist_mat_style=distance_style)

    first_iter = forest
    if verbose:
        bar = progressbar.ProgressBar()
        first_iter = bar(first_iter)

    # getting structure for tuning y_leaves and weights for each leaf
    y_all = np.zeros((0,))
    weight_all = np.zeros((0,))

    for t in first_iter:
        tree = t.tree_

        num_leaf = np.sum(tree.children_left == -1)

        # node associated
        # hmmm - to grab the OOB we could do:
        # _generate_sample_indices
        # from https://github.com/scikit-learn/scikit-learn/blob/bac89c253b35a8f1a3827389fbee0f5bebcbc985/sklearn/ensemble/forest.py#L78
        # just need to grab the tree's random state (t.random_state)


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

        # create y_leaf and weights for observed
        obs_V = t.decision_path(X_tune)
        obs_V_leaf = obs_V[:,tree.children_left == -1]
        obs_weight = obs_V_leaf.sum(axis = 0).ravel() # by column (leaf)

        #---
        # for clean division without dividing by 0
        obs_weight_div = obs_weight.copy()
        obs_weight_div[obs_weight_div == 0] = 1

        obs_y_leaf = (obs_V_leaf.T @ y_tune) / obs_weight_div

        y_all = np.concatenate((y_all, np.array(obs_y_leaf).ravel()))
        weight_all = np.concatenate((weight_all,
                                     np.array(obs_weight).ravel()))

    #---
    # Optimization
    #---
    if initial_lamb_seed is None:
        lamb = np.zeros(Gamma.shape[1]) #sanity check
        lamb[0] = 1
    else:
        np.random.seed(initial_lamb_seed)
        lamb = np.random.uniform(size = Gamma.shape[1])
        lamb = lamb / lamb.sum()

    if all_trees:
        n = t_idx_vec.shape
        t_idx_vec = np.zeros(n, dtype = np.int)

    if not sanity_check:
        lamb,lamb_last,c = subgrad_descent(y_all, weight_all,
                               Gamma, eta, t_idx_vec,
                               lamb_init=lamb, # no change
                               t_fix=subgrad_t_fix,
                               num_steps=subgrad_max_num,
                               constrained=not no_constraint,
                               verbose=verbose)

    #---
    # update random forest object (correct estimates from new lambda)
    #---
    y_leaf_new_all = (Gamma @ lamb) / (eta @ lamb)

    # to avoid divide by 0 errors (this may be a problem relative to the
    #   observed values)
    eta_fill2 = (eta @ lamb_last)
    eta_fill2[eta_fill2 == 0] = 1
    y_leaf_new_all2 = (Gamma @ lamb_last) / eta_fill2
    y_leaf_new_all2[(eta @ lamb_last) == 0] = 0

    start_idx = 0
    for t in forest:
        tree = t.tree_
        num_leaf = np.sum(tree.children_left == -1)


        tree.value[tree.children_left == -1,:,:] = \
            y_leaf_new_all[start_idx:(start_idx + num_leaf)].reshape((-1,1,1))

        start_idx += num_leaf

    inner_rf.lamb = lamb

    start_idx2 = 0
    for t in forest2:
        tree = t.tree_
        num_leaf = np.sum(tree.children_left == -1)


        tree.value[tree.children_left == -1,:,:] = \
            y_leaf_new_all[start_idx2:(start_idx2 + \
                                       num_leaf)].reshape((-1,1,1))

        start_idx += num_leaf

    inner_rf2.lamb = lamb_last

    return inner_rf, inner_rf2, lamb_last, c

def smooth(random_forest, X_trained=None, y_trained=None,
               X_tune=None, y_tune=None, verbose=True,
               no_constraint=False, sanity_check=False,
               resample_tune=False,
               subgrad_max_num=1000, subgrad_t_fix=1,
               all_trees=False,
               initial_lamb_seed=None,
               parents_all=False,
               distance_style=["standard","max", "min"]):
    """
    creates a smooth random forest (1 lambda set across all trees)

    this version uses the scaling relative to each observation

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
    no_constraint : bool
        logic to not constrain the weights
    sanity_check : bool
        logic to do full process but keep same weights
    resample_tune: bool
        logic to tune / optimize with another bootstrap same from X
    subgrad_max_num : int
        number of steps to take for the subgradient optimization
    subgrad_t_fix : scalar
        value for fixed t step size for gradient descent
    all_trees : bool
        logic to use all trees (and therefore do full gradient descent)
    initial_lamb_seed : scalar
        initial value for seed (default is None) to randomly select the
        starting point of lambda
    parents_all : bool
        logic to instead include all observations with parent of distance k
        away
   distance_style : string
        style of inner-tree distance to use, see *details* in the
        create_distance_mat_leaves doc-string.

    Returns:
    --------
    inner_rf : RandomForestClassifier or RandomForestRegressor
        updated smoothed random forest with optional lambda weighting, also has
        a .lamb parameter that stores weighting
    inner_rf2 : RandomForestClassifier or RandomForestRegressor
        updated smoothed random forest with lambda weighting from the final
        step of the stocastic gradient descent, also has a.lamb parameter that
        stores weighting
    lamb_last : numpy array
        last lambda value selected from the stocastic gradient descent
    c : list
        list of cost value for each stocastic gradient descent

    Comments:
    ---------

    If X_tune and/or y_tune is None then we will optimize each tree with oob
    samples.
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
    inner_rf2 = copy.deepcopy(random_forest)

    forest = inner_rf.estimators_
    forest2 = inner_rf2.estimators_

    _, max_depth = calc_depth_for_forest(random_forest,verbose=False)
    max_depth = np.int(max_depth)

    # getting structure from built trees
    Gamma, eta, t_idx_vec = create_Gamma_eta_forest(random_forest,
                                                verbose=verbose,
                                                parents_all=parents_all,
                                                dist_mat_style=distance_style)

    first_iter = forest
    if verbose:
        bar = progressbar.ProgressBar()
        first_iter = bar(first_iter)

    # getting structure for tuning y_leaves and weights for each leaf
    if rf_type == "reg":
        y_all = np.zeros((0,))
    else:
        y_all = np.zeros((0,Gamma.shape[0]))


    weight_all = np.zeros((0,))

    for t in first_iter:
        tree = t.tree_

        num_leaf = np.sum(tree.children_left == -1)

        # to grab the OOB:
        # _generate_sample_indices
        # from https://github.com/scikit-learn/scikit-learn/blob/bac89c253b35a8f1a3827389fbee0f5bebcbc985/sklearn/ensemble/forest.py#L78
        # just need to grab the tree's random state (t.random_state)


        # observed information
        if oob_logic:
            random_state = t.random_state
            oob_indices = \
                sklearn.ensemble.forest._generate_unsampled_indices(
                                                                 random_state,
                                                                 n_obs_trained)
            X_tune = X_trained[oob_indices,:]
            y_tune = y_trained[oob_indices]

            if rf_type == "class":
                y_tune = np.array(pd.get_dummies(y_tune))

        if resample_tune:
            resample_indices = \
                sklearn.ensemble.forest._generate_sample_indices(None,
                                                                 n_obs_trained)
            X_tune = X_trained[resample_indices,:]
            y_tune = y_trained[resample_indices]

            if rf_type == "class":
                y_tune = np.array(pd.get_dummies(y_tune))

        # create y_leaf and weights for observed
        obs_V = t.decision_path(X_tune)
        obs_V_leaf = obs_V[:,tree.children_left == -1]
        obs_weight = obs_V_leaf.sum(axis = 0).ravel() # by column (leaf)

        #---
        # for clean division without dividing by 0
        obs_weight_div = obs_weight.copy()
        obs_weight_div[obs_weight_div == 0] = 1

        # obs_y_leaf is either \hat{y}_obs or \hat{p}_obs
        if rf_type == "reg":
            obs_y_leaf = (obs_V_leaf.T @ y_tune) / obs_weight_div
            y_all = np.concatenate((y_all, np.array(obs_y_leaf).ravel()))
        else:
            obs_y_leaf = (obs_V_leaf.T @ y_tune) / obs_weight_div.T
            y_all = np.concatenate((y_all, np.array(obs_y_leaf)))


        weight_all = np.concatenate((weight_all,
                                     np.array(obs_weight).ravel()))

    #---
    # Optimization
    #---
    if initial_lamb_seed is None:
        lamb = np.zeros(Gamma.shape[-1]) #sanity check
        lamb[0] = 1
    else:
        np.random.seed(initial_lamb_seed)
        lamb = np.random.uniform(size = Gamma.shape[-1])
        lamb = lamb / lamb.sum()

    if all_trees:
        n = t_idx_vec.shape
        t_idx_vec = np.zeros(n, dtype = np.int)


    if rf_type == "class":
        Gamma_shape = Gamma.shape
        num_classes = Gamma.shape[0]
        Gamma = Gamma.reshape((Gamma.shape[0]*Gamma.shape[1],
                                     Gamma.shape[2]))

        eta = np.tile(eta, (num_classes,1))
        y_all = y_all.T.reshape((-1,))
        weight_all = np.tile(weight_all, num_classes)
        t_idx_vec = np.tile(t_idx_vec, num_classes)


    if not sanity_check:
        lamb,lamb_last,c = subgrad_descent(y_all, weight_all,
                               Gamma, eta, t_idx_vec,
                               lamb_init=lamb, # no change
                               t_fix=subgrad_t_fix,
                               num_steps=subgrad_max_num,
                               constrained=not no_constraint,
                               verbose=verbose)

    #---
    # update random forest object (correct estimates from new lambda)
    #---
    # to avoid divide by 0 errors (this may be a problem relative to the
    #   observed values)


    eta_fill = (eta @ lamb)
    eta_fill[eta_fill == 0] = 1
    y_leaf_new_all = (Gamma @ lamb) / eta_fill
    y_leaf_new_all[(eta @ lamb) == 0] = 0

    eta_fill2 = (eta @ lamb_last)
    eta_fill2[eta_fill2 == 0] = 1
    y_leaf_new_all2 = (Gamma @ lamb_last) / eta_fill2
    y_leaf_new_all2[(eta @ lamb_last) == 0] = 0

    if rf_type == "class":
        y_leaf_new_all = y_leaf_new_all.reshape((-1,num_classes))
        y_leaf_new_all2 = y_leaf_new_all2.reshape((-1,num_classes))

    start_idx = 0
    for t in forest:
        tree = t.tree_
        num_leaf = np.sum(tree.children_left == -1)

        if rf_type == "reg":
            tree.value[tree.children_left == -1,:,:] = \
                y_leaf_new_all[start_idx:(start_idx + num_leaf)].reshape((-1,1,1))
        else:
            tree.value[tree.children_left == -1,:,:] = \
                y_leaf_new_all[start_idx:(start_idx + num_leaf)].reshape((-1,1,num_classes))

        start_idx += num_leaf

    inner_rf.lamb = lamb


    start_idx2 = 0
    for t in forest2:
        tree = t.tree_
        num_leaf = np.sum(tree.children_left == -1)

        if rf_type == "reg":
            tree.value[tree.children_left == -1,:,:] = \
                y_leaf_new_all2[start_idx2:(start_idx2 + num_leaf)].reshape((-1,1,1))
        else:
            tree.value[tree.children_left == -1,:,:] = \
                y_leaf_new_all2[start_idx2:(start_idx2 + num_leaf)].reshape((-1,1,num_classes))

        start_idx += num_leaf



    inner_rf2.lamb = lamb_last

    return inner_rf, inner_rf2, lamb_last, c

def subgrad_descent(y_leaf, weights_leaf,
                    Gamma, eta, tree_idx_vec,
                    lamb_init, t_fix=1, num_steps=1000,
                    constrained=True, verbose=True):
    """
    Preform subgradient descent to minimize the l2 defined by

    |(y_leaf - Gamma @ lamb / eta @ lamb) * diag(weight_leaf**(1/2))|^2

    The subgradient steps randomly select a tree for each step to estimate the
    gradient with.

    Arguments:
    ----------
    y_leaf : array (Tn,)
        average y values for each leaf (observed values)
    weight_leaf : array (Tn,)
        number of observations observed at certain leaf
    Gamma : array (Tn, K)
        A horizontal stacking of Gamma matrices as defined above for all trees
        in the forest. See details below
    eta : array (Tn, K)
        A horizontal stacking of eta matrices as defined above for all trees
        in the forest.
    tree_idx_vec : int array (Tn,)
        integer array will holds which tree in the forest the associated row
        of Gamma or eta comes from
    lamb_init : array (K,)
        inital lambda value
    t_fix : scalar
        value for fixed t step size for gradient descent (default 1)
    num_steps : int
        number of steps for stocastic gradient descent to take (default 1000)
    constrained : bool
        logic for if the space for lamb values is constrainted onto the simplex
    verbose : bool
        logic to show steps of subgradient descent

    Returns:
    --------
    lamb : array (K,)
        optimial lambda value

    Details:
    --------
    *Gamma* and *eta*
    Gamma and eta matrices are from a forest (aka set of trees, where
    these two matrices are defined (per tree):

    Gamma_il = sum_j II(D_ij = l) n_j y_j
    eta_il = sum_j II(D_ij = l) n_j

    where n_j, y_j are the number of observations in leaf j and the predicted
    value associated with with leaf j. Note that D_ij is the tree based
    distance between leaf i and j.
    """

    if Gamma.shape[1] != lamb_init.shape[0]:
        raise TypeError("lamb_init needs to be the same length as the "+\
                        "number of columns in Gamma and eta")
    if constrained and (np.sum(lamb_init)!=1 or np.any(lamb_init < 0)):
        raise TypeError("For simplicity please initialize lamb_init with "+\
                        "a feasible value \n(ex: np.ones(Gamma.shape[1])/"+\
                        "Gamma.shape[1] )")

    if verbose:
        bar = progressbar.ProgressBar()
        first_iter = bar(np.arange(num_steps))
    else:
        first_iter = range(num_steps)

    lamb = lamb_init

    lamb_best = lamb
    cost_best = calc_cost(y_leaf, Gamma, eta, weights_leaf, lamb)

    num_trees = np.int(np.max(tree_idx_vec) + 1)

    cost_all = [cost_best]

    for s_step in first_iter:
        # select tree idx
        tree_idx = np.random.choice(num_trees)

        y_inner = y_leaf[tree_idx_vec == tree_idx]
        weights_inner = weights_leaf[tree_idx_vec == tree_idx]
        Gamma_inner = Gamma[tree_idx_vec == tree_idx,:]
        eta_inner = eta[tree_idx_vec == tree_idx,:]

        grad = take_gradient(y_inner, Gamma_inner, eta_inner,
                             weights_inner, lamb)

        lamb = lamb - t_fix * grad

        if constrained:
           lamb = prox_project(lamb)

        cost_new = calc_cost(y_leaf, Gamma, eta, weights_leaf, lamb)

        if cost_new < cost_best:
            cost_best = cost_new
            lamb_best = lamb

        cost_all = cost_all +[cost_new]

    return lamb_best, lamb, cost_all





def generate_data(large_n = 650):
    """
    generate data structure

    reproducing example on page 114 (Microsoft)
    """
    n = np.int(large_n / (4 + 1/3))
    unif_prob = .6
    norm_prob = 1 - unif_prob
    rotate_angle = -10
    delta = .05

    d1_n = np.random.normal(size = (np.int(n * norm_prob),2),
                            loc = [.5,.9],
                            scale = [.33,.05])
    d1_u = np.vstack( ( np.random.uniform(size = np.int(n * unif_prob)),
                        np.random.uniform(size = np.int(n * unif_prob) ,
                                          low = .8)) ).T


    d2_n = np.random.normal(size = (np.int(n * norm_prob),2),
                            loc = [.9,.5],
                            scale = [.05,.33])
    d2_u = np.vstack( ( np.random.uniform(size = np.int(n * unif_prob), low = .8),
                        np.random.uniform(size = np.int(n * unif_prob))) ).T

    d3_u_pre = np.random.uniform(size = (n,2), high = .5)
    d3_u_rotate = d3_u_pre @ \
                    np.array([[np.cos(rotate_angle * (np.pi/180)),
                                    -np.sin(rotate_angle * (np.pi/180))],
                              [np.sin(rotate_angle * (np.pi/180)),
                                    np.cos(rotate_angle * (np.pi/180))]])

    d3_u = d3_u_rotate + np.array([.175,.075])


    d4_1_n = np.random.normal(size = (np.int(n * 2/3 * norm_prob),2),
                              loc = [.5,.1],
                              scale = [.33,.05])
    d4_1_u = np.vstack( ( np.random.uniform(size = np.int(n * 2/3 * unif_prob)),
                        np.random.uniform(size = np.int(n * 2/3 * unif_prob),
                                          high = .2)) ).T

    d4_2_n = np.random.normal(size = (np.int(n * norm_prob),2),
                              loc = [.1,.5], scale = [.05,.33])
    d4_2_u = np.vstack( ( np.random.uniform(size = np.int(n * 2/3 * unif_prob),
                                            high = .2),
                        np.random.uniform(size = np.int(n * 2/3 * unif_prob) )) ).T


    data = np.vstack((d1_n, d1_u,
                      d2_n, d2_u,
                      d3_u,
                      d4_1_n, d4_1_u, d4_2_n, d4_2_u))

    n_square = np.sum([x.shape[0] for x in [d4_1_n, d4_1_u, d4_2_n, d4_2_u]],
                      dtype = np.int)


    y = np.array([2]*(d1_n.shape[0] + d1_u.shape[0]) +\
                 [3]*(d2_n.shape[0] + d2_u.shape[0]) +\
                 [0]*(d3_u.shape[0]) + [1]*n_square, dtype = np.int)

    # removing points:
    rm_logic = (data[:,0] > 1 + delta) + (data[:,0] < 0 - delta) +\
               (data[:,1] > 1 + delta) + (data[:,1] < 0 - delta)
    keep_logic = rm_logic == 0

    data = data[keep_logic,:]
    y = y[keep_logic]

    return data, y

def generate_data_knn(n = 650, p = np.array([.4,.6])):

  # data structure:

  data = pd.DataFrame(
    data = {"x" :
    [0.008754602, 0.013429834, 0.027455531, 0.050831692, 0.122518587,
    0.187971838, 0.230048928, 0.261217143, 0.259658733, 0.251866679,
    0.279918072, 0.317319930, 0.353163378, 0.378097950, 0.374981128,
    0.376539539, 0.409266165, 0.440434380, 0.463810541, 0.487186702,
    0.535497435, 0.549523132, 0.583808169, 0.593158633, 0.593158633,
    0.588483401, 0.561990418, 0.540172668, 0.521471739, 0.526146971,
    0.588483401, 0.600950687, 0.600950687, 0.608742741, 0.728740368,
    0.748999708, 0.759908583, 0.775492691, 0.792635209, 0.806660906,
    0.823803424, 0.845621175, 0.845621175, 0.872114157, 0.900165551,
    0.915749658, 0.915749658, 0.921983301, 0.931333766, 0.939125820,
    0.962501981, 0.985878142],
    "y":
    [0.04020202, 0.08815571, 0.10680437, 0.12545303, 0.16008626,
    0.11213256, 0.09348390, 0.09348390, 0.16008626, 0.20537586,
    0.21603224, 0.18672720, 0.14676579, 0.14676579, 0.20271177,
    0.25599365, 0.27997050, 0.27997050, 0.24267318, 0.22402452,
    0.25066547, 0.27997050, 0.37587790, 0.41051112, 0.48244167,
    0.52773127, 0.57568497, 0.62363867, 0.65560780, 0.72221016,
    0.74352291, 0.75417929, 0.81545346, 0.85008669, 0.83676621,
    0.80746118, 0.77282795, 0.70089740, 0.63695914, 0.54904403,
    0.52773127, 0.51174671, 0.51174671, 0.51174671, 0.51973899,
    0.51973899, 0.49043395, 0.40784703, 0.37054971, 0.35190105,
    0.33325239, 0.31193963]}, columns = ["x","y"])


  upper = pd.DataFrame(
    data = {"x" :
    [-0.013152278,  0.008168012,  0.052331469,  0.096494926,  0.136089750,
    0.174161696,  0.210710764,  0.242691198, 0.235076809,  0.235076809,
    0.268580121,  0.302083434,  0.326449479,  0.346246891,  0.359952791,
    0.367567180, 0.373658692,  0.402593371,  0.433050927,  0.457416973,
    0.472645751,  0.492443163,  0.519854964,  0.539652376, 0.560972665,
    0.571632810,  0.570109932,  0.562495543,  0.545743887,  0.528992231,
    0.510717697,  0.501580430, 0.501580430,  0.500057552,  0.519854964,
    0.541175253,  0.559449787,  0.574678566,  0.772652684,  0.793972974,
    0.809201752,  0.818339019,  0.827476286,  0.847273698,  0.874685499,
    0.903620178,  0.924940468,  0.941692124, 0.947783635,  0.950829391,
    0.964535291,  0.979764070,  0.987378459],
    "y":
    [0.0542169, 0.1300291, 0.1695833, 0.1893605, 0.1893605, 0.1662872,
    0.1465101, 0.1366215, 0.1959528, 0.2256185, 0.2420994, 0.2388032,
    0.2223223, 0.2058414, 0.2025452, 0.2552841, 0.2981345, 0.3113193,
    0.3212078, 0.3047269, 0.2915422, 0.2849498, 0.3014307, 0.3409849,
    0.3871315, 0.4266857, 0.4794247, 0.5024980, 0.5288674, 0.5618293,
    0.5914949, 0.6244568, 0.6837881, 0.7365270, 0.7793774, 0.8123393,
    0.8420049, 0.8650782, 0.8749668, 0.8123393, 0.7497118, 0.6936766,
    0.6574186, 0.6277530, 0.6079759, 0.6112720, 0.6112720, 0.5684216,
    0.5387560, 0.5024980, 0.4530552, 0.4233895, 0.4102048]},
    columns = ["x","y"])

  lower = pd.DataFrame(
    data = {"x" :
    [0.04319420, 0.07212888, 0.11324658, 0.14370414, 0.18177608,
    0.21984803, 0.26705724, 0.29751480, 0.29903768, 0.32340372,
    0.35690704, 0.40107049, 0.40411625, 0.41934503, 0.43761956,
    0.46198561, 0.48787453, 0.54269813, 0.58381583, 0.60209037,
    0.62950217, 0.62493353, 0.61427339, 0.58533871, 0.56097267,
    0.55640403, 0.59447598, 0.62188778, 0.62188778, 0.66148260,
    0.70412318, 0.73610362, 0.75285527, 0.76351542, 0.77417556,
    0.79397297, 0.83509068, 0.86097960, 0.88382277, 0.89752867,
    0.91123457, 0.94169212, 0.97976407],
    "y":
    [0.03443980, 0.06740164, 0.09047493, 0.07399401, 0.05092072,
    0.04432835, 0.04432835, 0.08388256, 0.12673295, 0.11684440,
    0.10695585, 0.11025203, 0.15639861, 0.21243374, 0.20584138,
    0.19595282, 0.18606427, 0.20584138, 0.26517269, 0.32120782,
    0.41350098, 0.46953611, 0.52886743, 0.58490256, 0.63434532,
    0.67060335, 0.69038046, 0.71674993, 0.77608125, 0.79915453,
    0.81233927, 0.76948888, 0.68708427, 0.58819875, 0.48931322,
    0.44975901, 0.42009335, 0.42009335, 0.41350098, 0.35416967,
    0.31791164, 0.27176506, 0.24869177]}, columns = ["x","y"])

  left_add = np.min([lower.x.min(),
                     upper.x.min(),
                     data.x.min()])

  right_add = np.max([lower.x.max(),
                     upper.x.max(),
                     data.x.max()])

  above = upper.y.max() + .1
  below = lower.y.min() - .1



  data_np = np.concatenate((np.array([[left_add,data.y[0]]]),
                            np.array(data),
                            np.array([[right_add,data.y[len(data.y) -1]]]),
                            np.array([[right_add, above],
                                      [left_add, above]]))
                            )
  lower_np = np.concatenate((np.array([[left_add,lower.y[0]]]),
                            np.array(lower),
                            np.array([[right_add,lower.y[len(lower.y) -1]]]),
                            np.array([[right_add, above],
                                      [left_add, above]]))
                            )
  upper_np = np.concatenate((np.array([[left_add,upper.y[0]]]),
                            np.array(upper),
                            np.array([[right_add,upper.y[len(upper.y) -1]]]),
                            np.array([[right_add, above],
                                      [left_add, above]]))
                            )

  # actual creation:
  new = np.hstack((np.random.uniform(low = left_add,
                                     high = right_add,
                                     size = n).reshape((-1,1)),
                   np.random.uniform(low = below,
                                     high = above,
                                     size = n).reshape((-1,1))))

  path_l = mpltPath.Path(lower_np)
  path_c = mpltPath.Path(data_np)
  path_u = mpltPath.Path(upper_np)

  inside_l = path_l.contains_points(new)
  inside_c = path_c.contains_points(new)
  inside_u = path_u.contains_points(new)

  value = inside_l * 1 + \
          inside_c * 1 + \
          inside_u * 1

  y = np.zeros(new.shape[0])
  for v in np.arange(1,4, dtype = np.int):
    if v == 1:
      y[value == v] = np.random.binomial(n = 1,
                                         size = np.sum(value == v),
                                         p = p[0])
    if v == 2:
      y[value == v] = np.random.binomial(n = 1,
                                         size = np.sum(value == v),
                                         p = p[1])
    if v == 3:
      y[value == v] = 1

  # plt.plot(data_np[:,0], data_np[:,1])
  # plt.plot(lower_np[:,0], lower_np[:,1])
  # plt.plot(upper_np[:,0], upper_np[:,1])
  # plt.scatter(new[:,0], new[:,1],
  #             c = np.array(["r","b"])[(y).astype(np.int)])

  return new, y, value


#### generating spiral function

def spirals(n_total = 600, n_classes = 3, noise_sd = .1, t_shift = 0):
    """
    create spirals dataset

    Arguments:
    ----------
    n_total: int
        total number of observations (up to rounding - all classes get the
        same number of observations)
    n_classes: int
        number of classes
    noise_sd: scalar
        standard deviation of student t noise added in (x,y) dimension
    t_shift: scalar
        shifts start of t from 0 to t_shift
        (so that classes don't connect at center)

    Returns
    -------
    data_all: pd dataframe, with columns: t, x, y, class (class is the label)
    """
    shifts = np.linspace(0, 2*np.pi, num = n_classes, endpoint = False)
    n_each = np.int(n_total/n_classes)


    data_all = pd.DataFrame()

    for c_idx, shift in enumerate(shifts):
        tt =  np.random.uniform(low = 0, high = 2*np.pi, size = n_each) +\
            shift + t_shift
        #np.linspace(0, 2*np.pi, num = n_each) + shift + t_shift
        inner_data = pd.DataFrame(
                        data = {"t": tt,
                                "x": np.sqrt(tt - shift) * np.cos(tt) +\
                                    # np.random.normal(size = n_each,
                                    #                 scale = noise_sd),
                                    noise_sd *\
                                        np.random.standard_t(df = 1,
                                                             size = n_each),
                                "y": np.sqrt(tt - shift) * np.sin(tt)+\
                                    # np.random.normal(size = n_each,
                                    #                  scale = noise_sd),
                                    noise_sd *\
                                        np.random.standard_t(df = 1,
                                                             size = n_each),
                                "class": c_idx
                                })

        data_all = data_all.append(inner_data)

    return data_all


#### Calculating center of nodes

def bound_box_tree(tree, X):
    """
    calculate bounding box (min/max for each variable) of each node/split of a
    tree

    Arguments:
    ----------
    tree : sklearn style binary decision tree (split on features)
        tree to calculate the bounding boxes on
    X : array (n_obs, n_col)
        data used to train the tree To initialize the bounding box

    Returns:
    --------
    storage_bound_box : array (n_node, n_col, 2)
        array that storages bounding box for each node - min and max for each
        features
    """
    n_nodes = tree.tree_.children_left.shape[0]
    n_col = X.shape[1]


    storage_bound_box = np.zeros((n_nodes, n_col, 2))
    storage_bound_box[0] = np.array([np.min(X, axis = 0),
                                     np.max(X, axis = 0)]).T

    _bound_box_tree(storage_bound_box, tree, idx = 0)

    return storage_bound_box # if we update the actual object...

def _bound_box_tree(storage_bound_box, tree, idx):
    """
    inner function for calculating a bounding box (min/max for each variable)
    of each node/split of a tree

    Arguments:
    ----------

    tree : sklearn style binary decision tree (split on features)
        tree to calculate the bounding boxes on
    idx : int
        integer associated with the idx for the current node you are looking at

    Returns:
    --------
    None, updates the storage_bound_box as defined in *details*

    Details:
    --------
    Updates the storage_bound_box for the children nodes of node relatived to
    provided idx, where storage_bound_box is defined as follows:

    storage_bound_box : array (n_node, n_col, 2)
        array that storages bounding box for each node - min and max for each
        features
    """
    if tree.tree_.children_left[idx] == -1: # leaf
        return None

    split_feature = tree.tree_.feature[idx]
    split_threshold = tree.tree_.threshold[idx]

    idx_left = tree.tree_.children_left[idx]
    idx_right = tree.tree_.children_right[idx]


    # obs_value <= threshold  for left
    # https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html

    # left child
    storage_bound_box[idx_left] = storage_bound_box[idx].copy()
    storage_bound_box[idx_left, split_feature, 1] = split_threshold
    _bound_box_tree(storage_bound_box, tree, idx_left)

    # right child
    storage_bound_box[idx_right] = storage_bound_box[idx].copy()
    storage_bound_box[idx_right, split_feature, 0] = split_threshold
    _bound_box_tree(storage_bound_box, tree, idx_right)

    return None

def center_tree(tree, X):
    """
    Calculates the center of each node of a decision tree

    Arguments:
    ----------
    tree : sklearn style binary decision tree (split on features)
        tree to calculate the centers on
    X : array (n_obs, n_col)
        data used to train the tree to initialize the centers

    Returns:
    --------
    storage : array (n_node, n_col)
        array that storages center for each node for each feature
    """
    tree_box = bound_box_tree(tree, X)

    return tree_box.mean(axis = 2)


def center_forest(forest, X):
    """
    Calculates the center of each node of a decision tree

    Arguments:
    ----------
    forest : list of sklearn style binary decision trees (split on features).
            think random_forest.estimators_
        list of trees to calculate the centers on.
    X : array (n_obs, n_col)
        data used to train the tree to initialize the centers

    Returns:
    --------
    mean_all : array (total_nodes, n_col)
        array that storages center for each node for each feature, stacking
        relative to trees
    t_idx_all : array (total_nodes)
        vector that storage the index of which tree the node belongs to
    """
    n_col = X.shape[1]

    mean_all = np.zeros((0,n_col))
    t_idx_all = np.zeros((0,))

    for t_idx, t in enumerate(forest):
        inner_mean = center_tree(t, X)
        mean_all = np.concatenate((mean_all, inner_mean), axis = 0)
        t_idx_all = np.concatenate((t_idx_all,
                                    [t_idx]*inner_mean.shape[0]))

    return mean_all, t_idx_all



