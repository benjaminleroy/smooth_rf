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
import pdb

from . import adam_sgd

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

def prox_project_ce(y, class_eps = 1e-4):
    """
    Approximate projection function onto the space defined by
    {x:x_i >= (class_eps / n)/(class_eps + 1), sum_i x_i =1}_i=1^n.

    This spaces is desireable for the cross-entropy loss and derivative (which
    includes log(x) or 1/x for an x that could be zero without such weighting).

    Arguments:
    ----------
    y : array (n, )
        array vector to be projected into the above space
    class_eps : scalar
        positive scalar (much smaller than 1) that adds small positive value
        to all entries in returned array
        ( value added is actually = (class_eps/n)/(1+class_eps) )
    Returns:
    --------
    y_prox : array (n, )
        y projected into the above space
    Notes:
    ------
    This algorithm comes from
    - https://arxiv.org/pdf/1101.6081.pdf
    - https://www.mathworks.com/matlabcentral/fileexchange/30332-projection-onto-simplex

    The algorithm (from the above paper) provides the projection to
    {x:x_i >= 0, sum_i x_i =1}_i=1^n, we provide a non-mathematically sound
    "correction" to the returned value that does get it in the desired space
    but doesn't necessarily get it to the most correct location.  If class_eps
    is small, it should be close geometrically.

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

    # update for cross-entropy
    y_prox = y_prox + np.ones(y_prox.shape[0]) * class_eps / y_prox.shape[0]
    y_prox = y_prox / y_prox.sum()
    y_prox = y_prox / y_prox.sum() # for some reason - needs to be done twice

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
                               decision_mat_leaves=None,
                               change_in_impurity_vec=None,
                               style=["standard", "max", "min"],
                               distance_style=["depth", "impurity"],
                               levels=None):
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
    change_in_impurity_vec : array (n_nodes)
        array containing change impurity for all nodes in the tree compare
        to their parents (see change_in_impurity function)
    style : string
        style of inner-tree distance to use, see *details* for more
    distance_style : string
        distance style (use depth or difference in impurity)
    levels : int
        if distance_style = "impurity", this decides the number of discrete
        levels of differences in impurity we will look at. 1 Level is "0", to
        include the standard random forest structure. If None, will return
        true difference in impurity - not levels. **NOTE: this is defined
        relative to quantiles, so the actual number of levels could be much
        lower.**

    Details:
    --------
    Written relative to `distance_style = "depth"` - similar when using other
    distances.

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

    if (type(distance_style) is list) and (len(distance_style) > 1):
        distance_style = distance_style[0]

    if distance_style not in ["depth", "impurity"]:
        ValueError("distance_style parameter needs to be 1 of the 2 choices.")

    if distance_style == "impurity":
        if tree is None and \
            (decision_mat_leaves is None or change_in_impurity_vec is None):
            ValueError("you must either provide a tree or a (decision_mat_leaves, change_in_impurity_vec) pair for impurity distance")
    else:
        if tree is None and decision_mat_leaves is None:
            ValueError("you must either provide a tree or a decision_mat_leaves for depth distance")


    if decision_mat_leaves is None:
        decision_mat_leaves, _ = create_decision_per_leafs(tree)


    if (type(style) is list) and (len(style) > 1):
        style = style[0]


    if levels is not None:
        if type(levels) is not int:
            ValueError("levels parameter needs to be an integer or None.")


    if distance_style == "depth":
        Q = decision_mat_leaves @ decision_mat_leaves.T

        if type(Q) is scipy.sparse.coo.coo_matrix or \
            type(Q) is scipy.sparse.csr.csr_matrix:
            d = np.diagonal(Q.todense())
        else:
            d = np.diagonal(Q)

        standard_distance = (d - Q.T).T
    elif distance_style == "impurity":

        if change_in_impurity_vec is None:
            impurity_diff = change_in_impurity(tree)
        else:
            impurity_diff = change_in_impurity_vec

        Q = decision_mat_leaves @ \
            scipy.sparse.diags(impurity_diff) @ decision_mat_leaves.T

        if type(Q) is scipy.sparse.coo.coo_matrix or \
            type(Q) is scipy.sparse.csr.csr_matrix:
            d = np.diagonal(Q.todense())
        else:
            d = np.diagonal(Q)

        standard_distance = -(d - Q.T).T
    else:
        ValueError("distance_style parameter needs to be 1 of the 2 choices.")

    if style == "standard":
        out = standard_distance
    elif style == "max":
        if type(standard_distance) is scipy.sparse.coo.coo_matrix or \
            type(standard_distance) is scipy.sparse.csr.csr_matrix:
            standard_distance = standard_distance.todense()

        standard_d_T = standard_distance.T

        both = np.append(np.array(standard_distance).reshape(
                                        (standard_distance.shape[0],
                                         standard_distance.shape[1],
                                         1)),
                         np.array(standard_d_T).reshape(
                                        (standard_d_T.shape[0],
                                         standard_d_T.shape[1],
                                         1)), axis = 2)
        out = np.max(both, axis = 2)
    elif style == "min":
        if type(standard_distance) is scipy.sparse.coo.coo_matrix or \
            type(standard_distance) is scipy.sparse.csr.csr_matrix:
            standard_distance = standard_distance.todense()

        standard_d_T = standard_distance.T
        both = np.append(np.array(standard_distance).reshape(
                                        (standard_distance.shape[0],
                                         standard_distance.shape[1],
                                         1)),
                         np.array(standard_d_T).reshape(
                                        (standard_d_T.shape[0],
                                         standard_d_T.shape[1],
                                         1)), axis = 2)
        out = np.min(both, axis = 2)
    else:
        ValueError("style parameter needs to be 1 of the 3 choices.")



    if distance_style != "impurity" or levels is None:
        return out
    else:
        # need to discretize
        if type(out) is scipy.sparse.coo.coo_matrix or \
            type(out) is scipy.sparse.csr.csr_matrix:
            out = out.todense()

        breaks = np.quantile(out, q = np.arange(levels + 2)/\
                    (levels + 1))

        upper_zero = np.min([np.min(out[out > 0]), 2])/2

        breaks = np.array(sorted(list(set(breaks).union({0}))))
        breaks[breaks.shape[0]-1] = np.inf
        breaks[0] = - np.inf

        if np.sum(breaks == 0) == 0: # dealing with 0 being replaced with -inf
            which_zero = np.int(np.arange(breaks.shape[0])[breaks == -np.inf])
        else:
            which_zero = np.int(np.arange(breaks.shape[0])[breaks == 0])

        breaks = np.array(list(breaks[:(which_zero+1)]) +\
                          [upper_zero] + list(breaks[(which_zero + 1):]))

        actual_levels = breaks.shape[0] - 1

        out2 = np.array(pd.cut(np.array(out).ravel(),
                               bins=breaks, labels=False)).reshape(out.shape)

        return out2

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


            if parents_all:
                if g_idx == 1: # regressor
                    extra = np.tile(g[:,g.shape[1] - 1].reshape((-1,1)),
                                    (1,diff))

                    g = np.hstack((g, extra))
                else: # classifier
                    extra = np.tile(
                        g[:,:,g.shape[2] - 1].reshape((g.shape[0], g.shape[1],1)),
                            (1,1,diff))
                    g = np.concatenate((g, extra),
                                        axis = 2)
            else:
                if g_idx == 1: # regressor
                    g = np.hstack((g, np.zeros((tree_n_leaf, diff))))
                else: # classifier
                    g = np.concatenate((g,
                                          np.zeros((num_class,
                                                    tree_n_leaf,
                                                    diff))),
                                        axis = 2)
            if parents_all:
                extra = np.tile(n[:,n.shape[1] - 1].reshape((-1,1)),
                                    (1,diff))
                n = np.hstack((n, extra))
            else:
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

    assert np.all(Gamma_fill[eta_fill == 0] == 0), \
        "whenever Gamma @ lamb == 0, then eta @ lamb should equal 0"

    eta_fill[eta_fill == 0] = 1 # to avoid divide by 0.
    residuals = y - Gamma_fill / eta_fill

    grad_yhat = scipy.sparse.diags(1/eta_fill) @ Gamma - \
                    scipy.sparse.diags(Gamma_fill/eta_fill**2) @ eta

    grad = -2 * residuals.T @ scipy.sparse.diags(weights) @ grad_yhat

    return grad

def take_gradient_ce(p, Gamma, eta, weights, lamb):
    """
    Calculates gradient of cross entropy
    where

    phat_ld = sum_k lam_k sum_j II(D_ij = k) n_j p_jd
             ----------------------------------------
              sum_k lam_k sum_j II(D_ij = k) n_j

    all elements' class structure is unraveled

    Arguments:
    ----------
    p : array (m*d,)
        true y-values (the average) per node of the tree
    Gamma : array (m*d, k)
        a matrix (rows for leafs, columns for depth) where each value

        Gamma[i,k]  = sum_l II(D_il = k) n_l p_ld

        (i.e. sum of training observations p values for those obs with tree
        dist k away from new node i, class d)
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
    eta_fill2 = eta_fill
    assert np.all(Gamma_fill[eta_fill == 0] == 0), \
        "whenever Gamma @ lamb == 0, then eta @ lamb should equal 0"

    eta_fill2[eta_fill == 0] = 1 # to avoid divide by 0.
    phat = Gamma_fill / eta_fill2


    left_npp = -weights*p / phat
    right_crazy = (Gamma.T @ np.diag(eta_fill) - eta.T @ np.diag(Gamma_fill)) \
                    / (eta_fill)**2

    grad = right_crazy @ left_npp
    return grad



def l2_s_grad_for_adam_wrapper(y, Gamma, eta, weights):
    """
    Stocastic gradient for l2 loss to be inserted into adam_step

    The inner function calculates gradient of l2 norm for y-yhat (|y-yhat|^2_2)
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

    Returns
    -------
    l2_sg_adam : lambda function
        lambda function of take_gradient function whos only is input is lamb
    """

    l2_sg_adam = lambda lamb: take_gradient(y, Gamma, eta, weights, lamb)

    return l2_sg_adam

def ce_s_grad_for_adam_wrapper(p, Gamma, eta, weights):
    """
    Stocastic gradient for ce loss to be inserted into adam_step

    The inner function calculates the gradient of cross entropy
    where

    phat_ld = sum_k lam_k sum_j II(D_ij = k) n_j p_jd
             ----------------------------------------
              sum_k lam_k sum_j II(D_ij = k) n_j

    all elements' class structure is unraveled

    Arguments:
    ----------
    p : array (m*d,)
        true y-values (the average) per node of the tree
    Gamma : array (m*d, k)
        a matrix (rows for leafs, columns for depth) where each value

        Gamma[i,k]  = sum_l II(D_il = k) n_l p_ld

        (i.e. sum of training observations p values for those obs with tree
        dist k away from new node i, class d)
    eta : array (m, k)
        a matrix (rows for leafs, columns for depth) where each value

        eta[i,k]  = sum_l II(D_il = k) n_l

        (i.e. the total number of training observations that are tree dist k
        away from the new node i)
    weights : array (m,)
        weights associated with node i. For our analysis this weight = number
        of observations in the ith node

    Returns
    -------
    ce_sg_adam : lambda function
        lambda function of take_gradient_ce function whos only is input is lamb
    """

    ce_sg_adam = lambda lamb: take_gradient_ce(p, Gamma, eta,
                                               weights, lamb)

    return ce_sg_adam

def calc_cost(y, Gamma, eta, weights, lamb):
    """
    calculates weighted l2 loss function

    Loss = sum_i^n w_i*|y_i - hat{y_i}|^2

    where

    hat{y} = Gamma @ lamb
             ------------
              eta @ lamb

    Arguments:
    ----------
    y : array (n, )
        mean value of leaf node (from test set)
    Gamma : array (n, k)
        Gamma matrix (sum_j n_j * bar{p}_j from training set for all
        leaf nodes at most k distance away for current leaf node.)
    eta : array (n, k)
        eta matrix (sum_j n_j  from training set for all
        leaf nodes at most k distance away for current leaf node.)
    weights : array (n,)
        weights array (number of observations in leaf from testing set)
    lamb : array (k,)
        weights relative to distance away for prediction node

    Return:
    -------
    loss : scalar
        weighted l2 loss
    """
    Gamma_fill = Gamma @ lamb
    eta_fill = eta @ lamb
    assert np.all(Gamma_fill[eta_fill == 0] == 0), \
        "whenever Gamma @ lamb == 0, then eta @ lamb should equal 0"

    eta_fill[eta_fill == 0] = 1 # to avoid divide by 0.

    residuals = y - Gamma_fill / eta_fill

    return np.sum( (residuals**2) * weights)

def calc_cost_ce(p, Gamma, eta, weights, lamb):
    """
    calculates weighted cross-entropy loss

    Loss = - sum_{l=1}^{# leaves} n_l sum_{d=1}^D p_{ld} log(hat{p}_{ld})

    where

    hat{p} = Gamma @ lamb
             ------------
              eta @ lamb

    Arguments:
    ----------
    p : array (l*d,)
        raveled probability vector
    Gamma : array (l*d, k)
        raveled Gamma matrix (sum_j n_j * bar{p}_j from training set for all
        leaf nodes at most k distance away for current leaf node.)
    eta : array (l*d, k)
        raveled eta matrix (sum_j n_j  from training set for all
        leaf nodes at most k distance away for current leaf node.)
    weights : array (l*d,)
        raveled weights array (number of observations in leaf from testing
        set)
    lamb : array (k,)
        weights relative to distance away for prediction node

    Return:
    -------
    loss : scalar
        weighted cross-entropy loss

    Notes:
    ------
    We expect all objects (except lamb) to be raveled version, with raveling
    done across # of classes dimensions.
    """
    try:
        assert np.all(Gamma >= 0), \
            "for probabilty structure Gamma should be non-negative"
        assert np.all(lamb >= 0), \
            "for probability structure lambda should be non-negative"
    except:
        pdb.set_trace()

    Gamma_fill = Gamma @ lamb
    eta_fill = eta @ lamb
    eta_fill[eta_fill == 0] = 1 # to avoid divide by 0.
    p_hat = Gamma_fill / eta_fill

    only_use_nonzeros = (weights != 0) * (p != 0)

    return - np.sum(weights[only_use_nonzeros] *\
                    p[only_use_nonzeros] *\
                    np.log(p_hat[only_use_nonzeros]))


def smooth(random_forest, X_trained=None, y_trained=None,
               X_tune=None, y_tune=None, verbose=True,
               no_constraint=False, sanity_check=False,
               resample_tune=False,
               sgd_max_num=1000, sgd_t_fix=1,
               all_trees=False,
               initial_lamb_seed=None,
               parents_all=False,
               distance_style=["standard","max", "min"],
               class_eps=1e-4,
               class_loss=["ce","l2"],
               adam=None):
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
    sgdgrad_max_num : int
        number of steps to take for the stocastic gradient optimization
    sgdgrad_t_fix : scalar
        value for fixed t step size for stocastic gradient descent
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
    class_eps : scalar (default 1e-4)
        For classification modles only. This scalar value makes it such that
        we keep the lamb values stay in the space defined by
        {x:x_i >= (class_eps / n)/(class_eps + 1), sum_i x_i =1}_i=1^n. This
        is done to not let the cross-entropy loss or it's derivative explode.
    class_loss : str (default = "ce")
        loss function for classification regression. Either "l2" for l2 loss or
        "ce" for cross-entropy loss.
    adam : dict (default is None)
        dictionary for input parameters adam SGD (if None, regular SGD is used)
        Note expected structure looks like:
            {"alpha": .001, "beta_1": .9, "beta_2": .999,"eps": 1e-8}

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

    if type(class_loss) == list:
        class_loss = class_loss[0]

    if adam is None:
        reg_sgd = True
    else:
        reg_sgd = False

    if rf_type == "class" and no_constraint == True:
        raise ValueError("for probabilities for the smoothed classification "+\
                         "rf to be consistent, you need to constrain the " +\
                         "lambda in the simplex (no_constraint = False)")



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

    if rf_type == "class": # to avoid initial problems in cross-entropy loss
        lamb = lamb + np.ones(lamb.shape[0]) * class_eps / lamb.shape[0]
        lamb = lamb / lamb.sum()
        lamb = lamb / lamb.sum() # need to do twice...

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
        if rf_type == "reg" or class_loss == "l2":
            if reg_sgd:
                lamb,lamb_last,c = stocastic_grad_descent(y_all, weight_all,
                                   Gamma, eta, t_idx_vec,
                                   lamb_init=lamb, # no change
                                   t_fix=sgd_t_fix,
                                   num_steps=sgd_max_num,
                                   constrained=not no_constraint,
                                   verbose=verbose)
            else:
                lamb,lamb_last,c = stocastic_grad_descent(y_all, weight_all,
                                   Gamma, eta, t_idx_vec,
                                   lamb_init=lamb, # no change
                                   t_fix=sgd_t_fix,
                                   num_steps=sgd_max_num,
                                   constrained=not no_constraint,
                                   verbose=verbose,
                                   adam=adam)
        else:
            if reg_sgd:
                lamb,lamb_last,c = stocastic_grad_descent_ce(y_all, weight_all,
                               Gamma, eta, t_idx_vec,
                               lamb_init=lamb,
                               t_fix=sgd_t_fix,
                               num_steps=sgd_max_num,
                               constrained=not no_constraint,
                               verbose=verbose,
                               class_eps=class_eps)
            else:
                lamb,lamb_last,c = stocastic_grad_descent_ce(y_all, weight_all,
                               Gamma, eta, t_idx_vec,
                               lamb_init=lamb,
                               t_fix=sgd_t_fix,
                               num_steps=sgd_max_num,
                               constrained=not no_constraint,
                               verbose=verbose,
                               class_eps=class_eps,
                               adam=adam)
    else:
        lamb_last = lamb
        c = []

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
        y_leaf_new_all = y_leaf_new_all.reshape((-1,num_classes),
                                                order = "F")
                                                # ^order by column, no row
        y_leaf_new_all2 = y_leaf_new_all2.reshape((-1,num_classes),
                                                  order = "F")

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

def stocastic_grad_descent(y_leaf, weights_leaf,
                    Gamma, eta, tree_idx_vec,
                    lamb_init, t_fix=1, num_steps=1000,
                    constrained=True, verbose=True,
                    adam=None):
    """
    Preform stocastic gradient descent to minimize the l2 defined by

    |(y_leaf - Gamma @ lamb / eta @ lamb) * diag(weight_leaf**(1/2))|^2

    The stocastic gradient steps randomly select a tree for each step to
    estimate the gradient with.



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
        logic to show steps of stocastic gradient descent
    adam : dict (default is None)
        dictionary for input parameters adam SGD (if None, regular SGD is used)
        Note expected structure looks like:
            {"alpha": .001, "beta_1": .9, "beta_2": .999,"eps": 1e-8}
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

    if adam is None:
        reg_sgd = True
    else:
        reg_sgd = False

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

        if reg_sgd: # regular sgd
            grad = take_gradient(y_inner, Gamma_inner, eta_inner,
                                 weights_inner, lamb)

            lamb = lamb - t_fix * grad
        else: # adam sgd
            take_gradient_adam = l2_s_grad_for_adam_wrapper(
                                                            y_inner,
                                                            Gamma_inner,
                                                            eta_inner,
                                                            weights_inner)
            if s_step == 0:
                iv = None
            lamb, iv = adam_sgd.adam_step(grad_fun = take_gradient_adam,
                                          lamb_init = lamb,
                                          internal_values = iv,
                                          **adam)

        if constrained:
           lamb = prox_project(lamb)

        cost_new = calc_cost(y_leaf, Gamma, eta, weights_leaf, lamb)

        if cost_new < cost_best:
            cost_best = cost_new
            lamb_best = lamb

        cost_all = cost_all +[cost_new]

    return lamb_best, lamb, cost_all



def stocastic_grad_descent_ce(p_leaf, weights_leaf,
                    Gamma, eta, tree_idx_vec,
                    lamb_init, t_fix=1, num_steps=1000,
                    constrained=True, verbose=True,
                    class_eps=1e-4,
                    adam=None):
    """
    Preform stocastic gradient descent to minimize the cross-entropy defined by

    Loss = \sum_{l=1}^{# leaves} n_l \sum_{d=1}^D p_{ld} log(\hat{p}_{ld})

    The stocastic gradient steps randomly select a tree for each step to estimate the
    gradient with.


    Arguments:
    ----------
    p_leaf : array (Tn*d,)
        average y values for each leaf (observed values)
    weight_leaf : array (Tn*d,)
        number of observations observed at certain leaf
    Gamma : array (Tn*d, K)
        A horizontal stacking of Gamma matrices as defined above for all trees
        in the forest. See details below
    eta : array (Tn*d, K)
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
        logic to show steps of stocastic gradient descent
    class_eps : scalar (default 1e-4)
        scalar value such that the lamb values stay in the space defined by
        {x:x_i >= (class_eps / n)/(class_eps + 1), sum_i x_i =1}_i=1^n. This
        is done to not let the cross-entropy loss or it's derivative explode.
    adam : dict (default is None)
        dictionary for input parameters adam SGD (if None, regular SGD is used)
        Note expected structure looks like:
            {"alpha": .001, "beta_1": .9, "beta_2": .999,"eps": 1e-8}

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

    Tn is the number of trees, d is number of class labels, K is the length of
    lambda
    """
    if Gamma.shape[1] != lamb_init.shape[0]:
        raise TypeError("lamb_init needs to be the same length as the "+\
                        "number of columns in Gamma and eta")
    try:
        if constrained and \
            (np.testing.assert_approx_equal(np.sum(lamb_init),1) or \
            np.any(lamb_init < 0)):
            raise TypeError("For simplicity please initialize lamb_init with "+\
                            "a feasible value \n(ex: np.ones(Gamma.shape[1])/"+\
                            "Gamma.shape[1] )")
    except:
        pdb.set_trace()

    if verbose:
        bar = progressbar.ProgressBar()
        first_iter = bar(np.arange(num_steps))
    else:
        first_iter = range(num_steps)

    if adam is None:
        reg_sgd = True
    else:
        reg_sgd = False

    lamb = lamb_init

    lamb_best = lamb
    cost_best = calc_cost_ce(p_leaf, Gamma, eta, weights_leaf, lamb)

    num_trees = np.int(np.max(tree_idx_vec) + 1)

    cost_all = [cost_best]

    for s_step in first_iter:
        # select tree idx
        tree_idx = np.random.choice(num_trees)

        p_inner = p_leaf[tree_idx_vec == tree_idx]
        weights_inner = weights_leaf[tree_idx_vec == tree_idx]
        Gamma_inner = Gamma[tree_idx_vec == tree_idx,:]
        eta_inner = eta[tree_idx_vec == tree_idx,:]


        if reg_sgd: # regular sgd
            grad = take_gradient_ce(p_inner, Gamma_inner, eta_inner,
                             weights_inner, lamb)

            lamb = lamb - t_fix * grad
        else: # adam sgd
            take_gradient_adam = ce_s_grad_for_adam_wrapper(
                                                            p_inner,
                                                            Gamma_inner,
                                                            eta_inner,
                                                            weights_inner)
            if s_step == 0:
                iv = None
            lamb, iv = adam_sgd.adam_step(grad_fun = take_gradient_adam,
                                          lamb_init = lamb,
                                          internal_values = iv,
                                          **adam)

        if constrained:
           lamb = prox_project_ce(lamb, class_eps)
        cost_new = calc_cost_ce(p_leaf, Gamma, eta, weights_leaf, lamb)

        if cost_new < cost_best:
            cost_best = cost_new
            lamb_best = lamb

        cost_all = cost_all + [cost_new]

    return lamb_best, lamb, cost_all

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

#### pytorch prep


def create_Gamma_eta_tree_more(tree,
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
    n_leaf_original : (n leaves, )
        number of observations that fell into the leaf
    leaf_depth : array (n_leaves, )
        depth of each leaf
    leaf_impurity : array (n_leaves, )
        impurity for each leaf based on tree criterion (default is 'gini' for
        classification, 'mse' for regression))
    """

    if type(tree) is sklearn.tree.tree.DecisionTreeRegressor:

        n_leaf_original = tree.tree_.weighted_n_node_samples[
                                    tree.tree_.children_left == -1]
        yhat_leaf_original = tree.tree_.value.ravel()[
                                    tree.tree_.children_left == -1]
        ny_leaf_original = n_leaf_original * yhat_leaf_original

        leaf_impurity = tree.tree_.impurity[tree.tree_.children_left == -1]

    elif type(tree) is sklearn.tree.tree.DecisionTreeClassifier:

        n_leaf_original = tree.tree_.weighted_n_node_samples[
                                tree.tree_.children_left == -1]
        ny_leaf_original = tree.tree_.value[
                                tree.tree_.children_left == -1,:,:].reshape(
                                                    n_leaf_original.shape[0],
                                                    tree.tree_.value.shape[-1])
        leaf_impurity = tree.tree_.impurity[tree.tree_.children_left == -1]

        # yhat_leaf_original = np.diag(1/n_leaf_original) @ ny_leaf_original

    else:
        ValueError("tree needs either be a "+\
                   "sklearn.tree.tree.DecisionTreeClassifier "+\
                   "or a sklearn.tree.tree.DecisionTreeRegressor")

    ############################
    # Depth per leaf gathering #
    ############################

    decision_mat_leaves, _ = create_decision_per_leafs(tree)

    Q = decision_mat_leaves @ decision_mat_leaves.T

    if type(Q) is scipy.sparse.coo.coo_matrix or \
        type(Q) is scipy.sparse.csr.csr_matrix:
        leaf_depth = np.diagonal(Q.todense()) - 1
    else:
        leaf_depth = np.diagonal(Q) - 1

    ##############################
    ## Distance Matrix Creation ##
    ##############################


    dist_mat_leaves = create_distance_mat_leaves(tree,
                                    decision_mat_leaves = decision_mat_leaves,
                                    style = dist_mat_style)

    ##########################
    # Gamma and Eta Creation #
    ##########################

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

    return Gamma, eta, n_leaf_original, leaf_depth, leaf_impurity


def create_Gamma_eta_tree_more_per(tree,
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
        see description above (also the *level version* of n_leaf_original)
    n_leaf_original : (n leaves, )
        number of observations that fell into the leaf
    leaf_depth : array (n_leaves, )
        depth of each leaf
    leaf_impurity : array (n_leaves, )
        impurity for each leaf based on tree criterion (default is 'gini' for
        classification, 'mse' for regression)
    leaf_depth_full : array (n_leaves, maximum depth of tree + 1)
        the *level version* of leaf_depth
    leaf_impurity_full : array (n_leaves, maximum depth of tree + 1)
        the *level version* of leaf_impurity
    Notes:
    ------
    When we say "the level version", we mean the value of that metric at
    x-levels above. In this definition, if we want to look at any levels
    greater than the leaf's depth we examine the value relative to the root
    node
    """

    if type(tree) is sklearn.tree.tree.DecisionTreeRegressor:

        n_leaf_original = tree.tree_.weighted_n_node_samples[
                                    tree.tree_.children_left == -1]
        yhat_leaf_original = tree.tree_.value.ravel()[
                                    tree.tree_.children_left == -1]
        ny_leaf_original = n_leaf_original * yhat_leaf_original

        leaf_impurity = tree.tree_.impurity[tree.tree_.children_left == -1]

    elif type(tree) is sklearn.tree.tree.DecisionTreeClassifier:

        n_leaf_original = tree.tree_.weighted_n_node_samples[
                                tree.tree_.children_left == -1]
        ny_leaf_original = tree.tree_.value[
                                tree.tree_.children_left == -1,:,:].reshape(
                                                    n_leaf_original.shape[0],
                                                    tree.tree_.value.shape[-1])

        leaf_impurity = tree.tree_.impurity[tree.tree_.children_left == -1]
        # yhat_leaf_original = np.diag(1/n_leaf_original) @ ny_leaf_original

    else:
        ValueError("tree needs either be a "+\
                   "sklearn.tree.tree.DecisionTreeClassifier "+\
                   "or a sklearn.tree.tree.DecisionTreeRegressor")

    ############################
    # Depth per leaf gathering #
    ############################

    decision_mat_leaves, _ = create_decision_per_leafs(tree)

    Q = decision_mat_leaves @ decision_mat_leaves.T

    if type(Q) is scipy.sparse.coo.coo_matrix or \
        type(Q) is scipy.sparse.csr.csr_matrix:
        leaf_depth = np.diagonal(Q.todense()) - 1
    else:
        leaf_depth = np.diagonal(Q) - 1

    ##############################
    ## Distance Matrix Creation ##
    ##############################


    dist_mat_leaves = create_distance_mat_leaves(tree,
                                    decision_mat_leaves = decision_mat_leaves,
                                    style = dist_mat_style)

    ##########################
    # Gamma and Eta Creation #
    ##########################

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


    ######################
    # changing per level #
    ######################

    ### leaf_depth per level (leaf_depth_full)

    # note this is 1 less than those of the leaf_depth ...
    leaf_depth_full = np.zeros((leaf_depth.shape[0],Gamma.shape[-1]))
    for row_idx in range(leaf_depth.shape[0]):
        leaf_depth_full[row_idx,:np.int(leaf_depth[row_idx] + 1)] = \
            np.arange(np.int(leaf_depth[row_idx] + 1))[::-1]

    ### n_leaf_original per level (eta)

    ### leaf_impurity per level (leaf_impurity_full)
    _, parent_mat = depth_per_node_plus_parent(tree)
    impurity_full = tree.tree_.impurity[parent_mat]
    leaf_impurity_full = impurity_full[tree.tree_.children_left == -1,:]

    return Gamma, eta, n_leaf_original, leaf_depth, \
        leaf_impurity, leaf_depth_full, leaf_impurity_full



def depth_per_node_plus_parent(tree):
    """
    calculcates depth per node in binary tree and provides parents index

    Arguments:
    ----------
    tree :
        tree object with the same structure as
        `sklearn.ensemble.DecisionTreeClassifier`

    Returns:
    --------
    depth_vec : int array (n_nodes, )
        vector of depth for each node
    parent_mat : int array (n_nodes, max_depth + 1)
        array of parents of each node k levels up
    """

    c_left  = tree.tree_.children_left
    c_right = tree.tree_.children_right
    T = len(c_left)

    depth_vec = np.zeros(T)
    parent_list = [[0] for i in np.arange(T)]

    for split in np.arange(T, dtype = np.int):
        if split > 0:
            parent_list[split] = parent_list[split] + [split]
        if c_left[split] != -1:
            depth_vec[c_left[split]] += depth_vec[split] + 1
            if split > 0:
                parent_list[c_left[split]] = parent_list[split]
        if c_right[split] != -1:
            depth_vec[c_right[split]] += depth_vec[split] + 1
            if split > 0:
                parent_list[c_right[split]] = parent_list[split]

    max_depth = np.max(depth_vec)
    parent_mat = np.array([[0] * np.int(max_depth+1 -len(x)) + x
                                for x in parent_list])[:,::-1]

    return depth_vec, parent_mat





def create_Gamma_eta_forest_more(forest, verbose=False, parents_all=False,
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
        in the forest. (also the *level version* of n_leaf_original)
    t_idx_all : int array (sum_{t=1}^T n_leaves(t), )
        integer array will holds which tree in the forest the associated row
        of Gamma_all or eta_all comes from

    num_leaf : int array (sum_{t=1}^T n_leaves(t), )
        number of observations that fell into the leaf
    depth_leaf : int array (sum_{t=1}^T n_leaves(t), )
        depth of each leaf
    impurity_leaf: :  array (sum_{t=1}^T n_leaves(t), )
        impurity for each leaf based on tree criterion (default is 'gini' for
        classification, 'mse' for regression)
    depth_full : array (sum_{t=1}^T n_leaves(t), maximum depth of forest + 1)
        the *level version* of leaf_depth
    impurity_full : array (sum_{t=1}^T n_leaves(t), maximum depth of forest +1)
        the *level version* of leaf_impurity
    """

    _, max_depth = calc_depth_for_forest(forest, verbose = False) # techincally repeats 1/2 the calculations in depth_per_node_plus_parent

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
    depth_full = np.zeros(shape = (0, np.int(max_depth + 1)))
    impurity_full = np.zeros(shape = (0, np.int(max_depth + 1)))

    t_idx_all = np.zeros(shape = (0,))
    num_leaf = np.zeros(shape = (0,))
    depth_leaf = np.zeros(shape = (0,))
    impurity_leaf = np.zeros(shape = (0,))


    first_iter = enumerate(forest.estimators_)

    if verbose:
        bar = progressbar.ProgressBar()
        first_iter = bar(list(first_iter))

    for t_idx, tree in first_iter:
        g, n, ln, ld, li, fd, fi = create_Gamma_eta_tree_more_per(tree,
                                     parents_all=parents_all,
                                     dist_mat_style=dist_mat_style)

        tree_n_leaf = g.shape[g_idx - 1]
        if g.shape[g_idx] != Gamma_all.shape[g_idx]:
            #^ needs to elements before concat
            diff = Gamma_all.shape[g_idx] - g.shape[g_idx]

            if parents_all:
                if g_idx == 1: # regressor
                    extra = np.tile(g[:,g.shape[1] - 1].reshape((-1,1)),
                                    (1,diff))

                    g = np.hstack((g, extra))
                else: # classifier
                    extra = np.tile(
                        g[:,:,g.shape[2] - 1].reshape((g.shape[0], g.shape[1],1)),
                            (1,1,diff))
                    g = np.concatenate((g, extra),
                                        axis = 2)
            else:
                if g_idx == 1: # regressor
                    g = np.hstack((g, np.zeros((tree_n_leaf, diff))))
                else: # classifier
                    g = np.concatenate((g,
                                          np.zeros((num_class,
                                                    tree_n_leaf,
                                                    diff))),
                                        axis = 2)
            if parents_all:
                extra = np.tile(n[:,n.shape[1] - 1].reshape((-1,1)),
                                    (1,diff))
                n = np.hstack((n, extra))
            else:
                n = np.hstack((n, np.zeros((tree_n_leaf, diff))))

            fd = np.hstack((fd, np.zeros((tree_n_leaf, diff))))
            i_top = fi[0,0]
            fi = np.hstack((fi, i_top * np.ones((tree_n_leaf, diff))))


        if g_idx == 1: # regressor
            Gamma_all = np.concatenate((Gamma_all, g))
        else: # classifier
            Gamma_all = np.concatenate((Gamma_all, g), axis = 1)

        eta_all = np.concatenate((eta_all, n))
        depth_full = np.concatenate((depth_full, fd))
        impurity_full = np.concatenate((impurity_full, fi))

        t_idx_all = np.concatenate((t_idx_all,
                                    t_idx * np.ones(tree_n_leaf,
                                                    dtype = np.int)))
        num_leaf = np.concatenate((num_leaf,
                                   ln))
        depth_leaf = np.concatenate((depth_leaf,
                                   ld))
        impurity_leaf = np.concatenate((impurity_leaf,
                                   li))

    return Gamma_all, eta_all, t_idx_all, num_leaf, depth_leaf, impurity_leaf,\
        depth_full, impurity_full


def change_in_impurity(tree):
    """
    calculates the change in impurity per node and is immediate parent in a
    binary tree

    Arguments:
    ----------
    tree :
        tree object with the same structure as
        `sklearn.ensemble.DecisionTreeClassifier`

    Returns:
    --------
    impurity_diff : float array
        vector of change in impurity for each node. Node 0 (root node) contains
        actual impurity value

    Notes:
    ------
    In a CART tree, you'll find that impurity isn't a monotonically decreasing
    value as you go down the tree
    """
    impurity = tree.tree_.impurity

    c_left  = tree.tree_.children_left
    c_right = tree.tree_.children_right
    T = len(c_left)

    impurity_diff = np.zeros(T)

    for split in np.arange(T, dtype = np.int):
        impurity_diff[split] += impurity[split]
        if c_left[split] != -1:
            impurity_diff[c_left[split]] -= impurity[split]
        if c_right[split] != -1:
            impurity_diff[c_right[split]] -= impurity[split]

    return impurity_diff
