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
    makes set of V_{t,lambda} matrices for a random forest
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
    makes a set of V_{t,lambda} matrices for a specific tree
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

    K_mat = 0 # not useful, but tests don't like Kmat not being defined
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


def is_pos_def(A):
    """
    check if matrix is PSD

    Code from: https://stackoverflow.com/questions/16266720/find-out-if-matrix-is-positive-definite-with-numpy

    Argument:
    ---------
    A: array (n, n)
        symmetric matrix (though checks if it is)

    Returns:
    --------
    boolean logic if matrix is PSD

    """
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


def process_tuning_leaf_attributes_tree(t, eps, X_tune, y_tune):
    """
    create (defined from the tuning data) the weights per leaf and y_value
    per leaf

    This process is for regression only currently

    Arguments:
    ----------
    t: tree
        Assumed to be sklearn regression base version, for the leaf structure
    eps: float
        value to fill in for leaves with no observations observed
    X_tune: array (n, p)
        array with X data for the tuning data
    y_tune: array (n, )
        vector with tuning data's y values

    Returns:
    --------
    obs_weight_non_zero: array (T,)
        vector of number of observations in the tuning data that fall into each
        leaf (T total number of leaves). 0 is replaced with eps.
    obs_y_leaf: array (T,)
        vector of average y value for each leaf (T total number of leaves)
    """
    tree = t.tree_

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

    return np.array(obs_weight_non_zero).ravel(), \
        np.array(obs_y_leaf).ravel()

def update_til_psd(G, verbose=True):
    """
    If the symmetric matrix G is not PSD then we add diagonal * eps_star,
    in sequences of 2^k * machine_eps until it is PSD.

    Argument:
    ---------
    G : array (m, m)
        symmetric matrix
    verbose : bool
        boolean value if printout for number of updates

    Returns:
    --------
    G_updated : array (m, m)
        updated G matrix that is PSD (G + diag(w)) where w is lowest 2^k * eps
    """

    G_updated = copy.deepcopy(G)

    if not np.allclose(G, G.T):
        raise ValueError("G needs to be symmetric")

    reattempt = True
    two_power = -1
    while reattempt:
        reattempt = False
        if not is_pos_def(G_updated):
            reattempt = True
            two_power += 1

            if verbose:
                if two_power == 0:
                    sys.stdout.write('\n')
                sys.stdout.write(".")

            if two_power != 0:
                G_updated = G_updated +\
                                np.diag(np.ones(G.shape[0]) *\
                                np.finfo(float).eps *\
                                1000 * (2**two_power - 2**(two_power-1)))
            else:
                G_updated = G_updated +\
                                np.diag(np.ones(G.shape[0]) *\
                                np.finfo(float).eps *\
                                1000)
    return G_updated

def check_in_null(G, v, tol_pow = None):
    """
    checks if a vector is in the null space of a matrix
    Arguments:
    ----------
    G : array (n, m)
        matrix with null space
    v : array (n, )
        vector to check if within null space
    tol_pow : int (non-negative)  (or None)
        power for tolerance for deciding the rank of the matrix with
        both the null space of G and the v containdated by examining the
        singlar values (s) and looking at which are less than:
            max(s) * 2^tol_pow * np.finfo(float).eps
        this same value is also used in estimating the null space for G,
        which looks at the singular value as of G smaller in magnitude than
        the following as associated with the null space:
            max(n,m) * 2^tol_pow * np.finfo(float).eps
    Returns:
    --------
    value : bool
        boolean value if v is in the null space of G
    """

    if tol_pow is not None and (tol_pow < 0 or \
                                np.floor(tol_pow) != tol_pow):
        raise ValueError("tol_pow needs to be a non-negative integer")

    if False:#tol_pow is not None:
        nspace = scipy.linalg.null_space(G, rcond=np.max(G.shape) *\
                                                    2**tol_pow *\
                                                    np.finfo(float).eps)
    else:
        nspace = scipy.linalg.null_space(G)

    null_rank = nspace.shape[1]

    if null_rank == 0:
        return False

    large_mat = np.concatenate((nspace, v.reshape((-1,1))), axis=1)
    if tol_pow is not None:
        s = np.linalg.svd(large_mat,compute_uv=False)

        larger_rank = np.linalg.matrix_rank(large_mat,
                                            tol=s.max() *\
                                                2**tol_pow *\
                                                np.finfo(float).eps)
    else:
        larger_rank = np.linalg.matrix_rank(large_mat)
    return larger_rank == null_rank

def smooth_all(random_forest, X_trained, y_trained, X_tune=None, y_tune=None,
               verbose=True,
               no_constraint=False,
               sanity_check={"sanity check":False, "tol_pow":None},
               resample_tune=False,parents_all=False,
               inner_assess=False):
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
    sanity_check : bool or dictionary
        if the first, logic to do full process but keep same weights
        if the second, a dictionary with "santity check" as a bool defined
        above and "tol_pow" as the value to put into smooth_rf.check_in_null
        function relative to analysis inner_assess analysis.
    resample_tune: bool
        logic to tune / optimize with another bootstrap same from X
    parents_all : bool
        logic to instead include all observations with parent of distance k
        away
    inner_assess : bool
        if one should assess if the base rf lamb beats the current lamb value

    Returns:
    --------
    an updated RandomForestRegressor object with values for each node altered
    if desirable

    Comments:
    ---------

    1. If X_tune and/or y_tune is None then we will optimize each tree with oob
    samples.
    2. If the created matrix (Gamma/eta).T @ diag(weights) @ (Gamma/eta) is not
    PSD then we add diagonal * eps_star, in sequences of 2^k * machine_eps until
    it is PSD.

    """

    n_obs_trained = X_trained.shape[0]
    eps = 1/n_obs_trained
    numeric_eps = 1e-5

    if (X_tune is None or y_tune is None) and not resample_tune:
        oob_logic = True
    else:
        oob_logic = False

    y_trained = y_trained.ravel()

    if y_trained.shape[0] != X_trained.shape[0]:
        raise ValueError("y_trained and X_trained should have the same "+\
                         "number of observations")

    if type(sanity_check) is dict:
        try:
            tol_pow = sanity_check["tol_pow"]
            sanity_check = sanity_check["sanity check"]
        except:
            raise ValueError("a dictionary input for sanity_check did not "+\
                             "keys named named 'sanity check' and 'pow_tol'")
    else:
        tol_pow = None

    if type(sanity_check) is not bool:
        raise ValueError("sanity_check or sanity_check['sanity check'] must "+\
                         "must be a boolean value.")

    inner_rf = copy.deepcopy(random_forest)
    forest = inner_rf.estimators_

    _, max_depth = smooth_rf.calc_depth_for_forest(random_forest,verbose=False)
    max_depth = np.int(max_depth)

    obs_y_leaf_all = np.zeros(0)
    obs_weight_non_zero_all = np.zeros(0)

    first_iter = forest
    if verbose:
        bar = progressbar.ProgressBar()
        first_iter = bar(first_iter)


    for t in first_iter:
        tree = t.tree_
        num_leaf = np.sum(tree.children_left == -1)

        # # trained "in bag observations"
        random_state = t.random_state
        # sample_indices = \
        #     sklearn.ensemble.forest._generate_sample_indices(random_state,
        #                                                      n_obs_trained)
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


        obs_weight_non_zero, obs_y_leaf = \
            process_tuning_leaf_attributes_tree(t, eps = -1,
                                                X_tune = X_tune,
                                                y_tune = y_tune)

        obs_weight_non_zero_all = np.hstack((obs_weight_non_zero_all,
                                    np.array(obs_weight_non_zero).ravel()))
        obs_y_leaf_all = np.hstack((obs_y_leaf_all,
                                    np.array(obs_y_leaf).ravel()))

    ga, et, _ = smooth_rf.create_Gamma_eta_forest(inner_rf,
                                                    verbose=verbose,
                                                    parents_all=parents_all)

    assert np.all(ga[et == 0] == 0), \
        "All gamma values when eta = 0 should also equal 0"

    # this should only account for dividing by 0
    et[et == 0] = eps
    Gamma_all = ga/et

    # if we see no observations in the leaf we shouldn't look at it
    not_missing_idx = obs_weight_non_zero_all != -1

    Gamma_all_store = copy.deepcopy(Gamma_all)
    Gamma_all = Gamma_all[not_missing_idx,:]
    obs_weight_non_zero_all = obs_weight_non_zero_all[not_missing_idx]
    obs_y_leaf_all = obs_y_leaf_all[not_missing_idx]

    # optimization:
    G = Gamma_all.T @ \
        scipy.sparse.diags(np.array(obs_weight_non_zero_all).ravel()) @ \
        Gamma_all # (n,n)

    a = Gamma_all.T @ \
        scipy.sparse.diags(np.array(obs_weight_non_zero_all).ravel()) @ \
        np.array(obs_y_leaf_all).ravel() # (n,)
    C = np.hstack((1 * np.ones((max_depth + 1,1)),
                   1 * np.identity(max_depth + 1))) # (n, m)
    b = np.array([1] + [0]*(max_depth + 1)) #(m)

    # COMMENT: FOR ERROR: Gamma can have linearly dependent columns...
    # how to think about (pinv?) - should have learned this implimentation

    #pdb.set_trace()
    G = (G + G.T)/2
    G = update_til_psd(G, verbose=verbose) # apparently this isn't good enough...

    reattempt = True
    two_power = -1
    while reattempt:
        reattempt = False
        try:
            if no_constraint:
                lamb = np.linalg.inv(G) @ a
            else:
                opt = quadprog.solve_qp(G = G.astype(np.float),
                                                a = a.astype(np.float),
                                                C = C.astype(np.float),
                                                b = b.astype(np.float),
                                                meq = 1)
                lamb = opt[0]
        except:
            reattempt = True
            two_power += 1

            if verbose:
                if two_power == 0:
                    sys.stdout.write('\n')
                sys.stdout.write(".")

            if two_power != 0:
                G = G + np.diag(np.ones(G.shape[0]) *\
                        np.finfo(float).eps *\
                        1000 * (2**two_power - 2**(two_power-1)))
            else:
                G = G + np.diag(np.ones(G.shape[0]) *\
                        np.finfo(float).eps *\
                        1000)



    # internal sanity check --------
    cost_actual = 1/2 * lamb.T @ G @ lamb - a.T @ lamb

    lamb_base = np.zeros(lamb.shape)
    lamb_base[0] = 1

    cost_base = 1/2 * lamb_base.T @ G @ lamb_base - a.T @ lamb_base

    if np.any(np.abs(lamb - lamb_base) > 1e-7):
        #print(cost_base >= cost_actual)
        #print(np.abs(lamb - lamb_base))
        #pdb.set_trace()

        #if not (cost_base >= cost_actual):
        #    pdb.set_trace()
        if not check_in_null(G, lamb_base - lamb,tol_pow=tol_pow):
            if inner_assess:
                assert cost_base >= cost_actual, \
                    "the base lambda is inside the options of lambda, "+\
                    "so there is a problem with the minimization"


    if sanity_check:
    # end of interal sanity check --------
        lamb = np.zeros(lamb.shape)
        lamb[0] = 1

    y_leaf_new_all = Gamma_all_store @ lamb

    start_idx = 0
    for t in forest:
        tree = t.tree_
        num_leaf = np.sum(tree.children_left == -1)


        tree.value[tree.children_left == -1 ,:,:] = \
            y_leaf_new_all[start_idx:(start_idx + num_leaf)].reshape((-1,1,1))

        start_idx += num_leaf

    inner_rf.lamb = lamb

    return inner_rf


