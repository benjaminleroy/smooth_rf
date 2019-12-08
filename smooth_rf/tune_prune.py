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

import pdb


# def initialize_prune_df_old(tree):
#     c_left = tree.tree_.children_left
#     c_right = tree.tree_.children_right
#     parents = calc_parent(tree)
#     n_obs = tree.tree_.weighted_n_node_samples
#     n = c_left.shape[0]
#     r_single_t_val = r_single_t(tree)
#     prune_df = pd.DataFrame(data = {"idx":np.arange(n),
#                                     "R({t})":r_single_t_val,
#                                     "R(T_t)":r_single_t_val,
#                                     "c_left":c_left,
#                                     "c_right":c_right,
#                                     "|T_t|":np.zeros(n, dtype=np.int),
#                                     "parent":parents,
#                                     "n_obs":n_obs})

#     # initialize queue with leaf nodes
#     queue = list(prune_df.idx[prune_df.c_left == -1])
#     print(queue)

#     if len(queue) < 2:
#         raise ValueError("Tree is malformed or is only a root node (has < 2 leafs)")
#     current_idx = queue.pop(0)

#     while current_idx is not None:
#         print(queue)
#         # leaf
#         if (prune_df.loc[prune_df.idx == current_idx, "c_left"] == -1).values:
#             prune_df.loc[prune_df.idx == current_idx,"|T_t|"] = 1

#             parent_idx = prune_df.loc[prune_df.idx == current_idx, "parent"].values[0]
#             if parent_idx != -1:
#                 if parent_idx not in queue:
#                     queue.append(parent_idx)
#                 else:
#                     #pdb.set_trace()
#                     queue.pop(np.int(np.arange(len(queue), dtype = np.int)[queue == parent_idx]))
#                     queue.append(parent_idx)

#         else:
#             #pdb.set_trace()
#             prune_df.loc[prune_df.idx == current_idx, "|T_t|"] =\
#                 prune_df.loc[prune_df.c_left[prune_df.idx == current_idx], "|T_t|"] +\
#                 prune_df.loc[prune_df.c_right[prune_df.idx == current_idx], "|T_t|"]
#             prune_df.loc[prune_df.idx == current_idx,"R(T_t)"] =\
#                 (prune_df.loc[prune_df.c_left[prune_df.idx == current_idx], "R(T_t)"] *\
#                  prune_df.loc[prune_df.c_left[prune_df.idx == current_idx],"n_obs"] +\
#                  prune_df.loc[prune_df.c_right[prune_df.idx == current_idx], "R(T_t)"] *\
#                  prune_df.loc[prune_df.c_right[prune_df.idx == current_idx],"n_obs"]) /\
#                 prune_df.loc[prune_df.idx == current_idx, "n_obs"]

#         try:
#             current_idx = queue.pop(0)
#         except:
#             current_idx = None


def prune_tree(tree, data):
    """
    this function fully prunes the tree while keeping track of R_oob values
    provides alpha values and associated nodes for future prune

    thoughts:
    1. g_1 (critical value) is defined [R(t) - R(T_t)]/[|T_t| - 1]
    1.1. select minimum g_1 to be the new value
    1.1.1. will need to update g_1 throught the tree

    2. could keep track / precalculate oob MSE for each node (is this easy)? - R_oob({t})
    2.1. need to track n_oob
    2.2. then could calculate MSE_alpha from n_oob * MSE relative to only leaf nodes
    """

    random_state = t.random_state
    oob_indices = \
        sklearn.ensemble.forest._generate_unsampled_indices(
                                                         random_state,
                                                         n_obs_trained)
    X_tune = X_trained[oob_indices,:]
    y_tune = y_trained[oob_indices]

    if rf_type == "class":
        y_tune = np.array(pd.get_dummies(y_tune))



def _inner_prune(prune_df, prune_idx):
    """
    update prune_df to prune relative to node (prune_idx)

    Arguments:
    ----------
    prune_df : pd.DataFrame
        data frame with information about each node (before any pruning occurs)
        (see structure in _initialize_prune_df function)
    prune_idx : int
        which nodes to prune

    Returns:
    --------
    out_df : updated prune_df data frame

    Details:
    --------
    this function is broken into 2 helper functions: a _inner_prune_down and
    a _inner_prune_update_upward
    """

    if (prune_df.c_left[prune_df.idx == prune_idx].values[0] in {-1,-2}):
        raise ValueError("the node associated with prune_idx is already a "+\
                         "been pruned / is a leaf.")

    out_df = _inner_prune_down(prune_df, prune_idx)
    out_df = _inner_prune_update_upward(out_df, prune_idx)

    return out_df

def _inner_prune_update_upward(prune_df, prune_idx):
    """
    progate upwards (through parents) updates of |T_t| and R(T_t) values
    as well as g1 values assoicated with these updates

    Arguments:
    ----------
    prune_df : pd.DataFrame
        data frame with information about each node (before any pruning occurs)
        (see structure in _initialize_prune_df function)
    prune_idx : int
        which nodes to prune

    Returns:
    --------
    out_df : updated prune_df data frame
    """

    out_df = prune_df.copy()
    if (out_df.loc[out_df.idx == prune_idx, "parent"] == -1).values:
        # ^ if was root node
        return out_df


    queue = [np.int((out_df.loc[out_df.idx == prune_idx, "parent"]).values)]

    current_idx = queue.pop(0)

    while current_idx is not None:

        out_df.loc[out_df.idx == current_idx,"|T_t|"] =\
                out_df.loc[out_df.idx == out_df.loc[out_df.idx == current_idx,"c_left"].values[0], "|T_t|"].values[0] +\
                out_df.loc[out_df.idx == np.int(out_df.loc[out_df.idx == current_idx,"c_right"].values[0]), "|T_t|"].values[0]

        out_df.loc[out_df.idx == current_idx,"R(T_t)"] =\
            (out_df.loc[out_df.c_left[out_df.idx == current_idx].values[0], "R(T_t)"] *\
             out_df.loc[out_df.c_left[out_df.idx == current_idx].values[0],"n_obs"] +\
             out_df.loc[out_df.c_right[out_df.idx == current_idx].values[0], "R(T_t)"] *\
             out_df.loc[out_df.c_right[out_df.idx == current_idx].values[0],"n_obs"]) /\
            out_df.loc[out_df.idx == current_idx, "n_obs"].values[0]

        # calculation of g1 value
        out_df.loc[out_df.idx == current_idx, "g1"] = \
            (out_df.loc[out_df.idx == current_idx, "R({t})"].values[0] -\
             out_df.loc[out_df.idx == current_idx, "R(T_t)"].values[0]) /\
            (out_df.loc[out_df.idx == current_idx, "|T_t|"].values[0] - 1)

        # COME HERE
        # need to update OOB values
        raise ValueError("need update here")


        if (out_df.loc[out_df.idx == current_idx, "parent"] != -1).values[0]:
            # ^ if not root node
            queue.append(out_df.loc[out_df.idx == current_idx, "parent"].values[0])

        try:
            current_idx = queue.pop(0)
        except:
            current_idx = None

    return out_df

def _inner_prune_down(prune_df, prune_idx):
    """
    inner function to update prune_df below a single node. See details for
    what this function does.

    Arguments:
    ----------
    prune_df : pd.DataFrame
        data frame with information about each node (before any pruning occurs)
        (see structure in _initialize_prune_df function)
    prune_idx : int
        which nodes to prune

    Returns:
    --------
    out_df : updated prune_df data frame

    Details:
    --------
    - lost/pruned nodes have |T_t| = 0, c_left/c_right = -2
    - all effected nodes have R(T_t) = R({t}) (new leaf node and
    lost/pruned nodes)

    Comments:
    ---------
    doesn't check that prune_idx is not a leaf node already...
    """
    out_df = prune_df.copy()

    out_df.loc[out_df.idx == prune_idx, "|T_t|"] = 1
    out_df.loc[out_df.idx == prune_idx, "R(T_t)"] = prune_df.loc[out_df.idx == prune_idx, "R({t})"]
    out_df.loc[out_df.idx == prune_idx, "c_left"] = -1
    out_df.loc[out_df.idx == prune_idx, "c_right"] = -1

    if prune_df.loc[prune_df.idx == prune_idx, "c_left"].values != -1:
        out_df = _recursive_inner_prune(prune_df, out_df, np.int(prune_df.loc[prune_df.idx == prune_idx, "c_left"].values))
        out_df = _recursive_inner_prune(prune_df, out_df, np.int(prune_df.loc[prune_df.idx == prune_idx, "c_right"].values))

    # COME HERE
    # need to update OOB values
    raise ValueError("need update here")


    return out_df


def _recursive_inner_prune(prune_df, out_df, current_idx):
    #pdb.set_trace()
    if prune_df.loc[prune_df.idx == current_idx, "c_left"].values != -1:
        out_df = _recursive_inner_prune(prune_df, out_df, np.int(prune_df.loc[prune_df.idx == current_idx, "c_left"].values))
        out_df = _recursive_inner_prune(prune_df, out_df, np.int(prune_df.loc[prune_df.idx == current_idx, "c_right"].values))

    out_df.loc[out_df.idx == current_idx, "|T_t|"] = 0
    out_df.loc[out_df.idx == current_idx, "R(T_t)"] = prune_df.loc[out_df.idx == current_idx, "R({t})"]
    out_df.loc[out_df.idx == current_idx, "c_left"] = -2
    out_df.loc[out_df.idx == current_idx, "c_right"] = -2

    out_df.loc[out_df.idx == current_idx, "g1"] = np.nan

    return out_df

def _initialize_prune_df(tree, X_trained=None, y_trained=None):
    """
    inner function for initializing prune_df (no pruning)

    Arguments:
    ----------
    tree : sklearn style tree (DecisionTreeClassifier or DecisionTreeRegressor)
        grown tree
    X_trained : array (n, p)
        X data array used to create the inputted tree. Note that this
        is assumed to be the correct data
        (default is none)
    y_trained : array (n, )
        y data array used to create the inputted tree.


    Returns:
    --------
    prune_df : pd.DataFrame
        data frame with information about each node (before any pruning occurs)

    Details:
    --------
    prune_df contains the following columns:
    idx : int
        index of node in tree
    R({t}) : float
        Training risk associated with tree with just this node
    R(T_t) : float
        Training risk associated with tree with all non-pruned nodes below this
        node
    c_left : int
        idx of node that is current nodes left child (when a leaf = -1, when
        pruned = -2)
    c_right : int
        idx of node that is current nodes right child (when a leaf = -1, when
        pruned = -2)
    |T_t| : int
        number of leaf nodes below this node (0 if pruned, 1 if leaf, etc)
    parent : int
        idx for parent of current node (-1 if root node)
    n_obs : float (but should be thought of as an integer)
        number of training observations that fell into this node
    g1 : float
        g_1 = [R(t) - R(T_t)]/[|T_t| - 1]. Associated with pruning rule for
        CART trees. See CART chapter 2.

    Details:
    -------
    The last 2 columns only are returned if X_trained and y_trained are
    provided.
    """
    c_left = tree.tree_.children_left
    c_right = tree.tree_.children_right
    parents = calc_parent(tree)
    n_obs = tree.tree_.weighted_n_node_samples
    n = c_left.shape[0]
    r_single_t_val = r_single_t(tree)
    prune_df = pd.DataFrame(data = {"idx":np.arange(n),
                                    "R({t})":r_single_t_val,
                                    "R(T_t)":r_single_t_val,
                                    "c_left":c_left,
                                    "c_right":c_right,
                                    "|T_t|":np.zeros(n, dtype=np.int),
                                    "parent":parents,
                                    "n_obs":n_obs#,
                                    #
                                    },
                            columns = ["idx", "R({t})", "R(T_t)",
                                       "c_left", "c_right", "|T_t|",
                                       "parent", "n_obs"])


    prune_df = _recursive_initial_prune_df(prune_df, current_idx = 0)
    prune_df = _append_g1(prune_df)

    if X_trained is not None and y_trained is not None:
        prune_df = _append_oob_values(prune_df, tree, X_trained, y_trained)

    return prune_df


def _append_g1(prune_df):
    """
    add on a g1 column to prune_df with correct columns relative to R(t),
    R(T_t) and |T_t|. Assumes this is before pruning and all other values are
    correct. Recall from CART - chapter 2 that

    g1 = [R(t) - R(T_t)]/[|T_t| - 1]

    Arguments:
    ----------
    prune_df :
        data frame with information about each node (before any pruning occurs)
        which doesn't have a g1 column

    Returns:
    --------
    prune_df : pd.DataFrame
        data frame with information about each node (before any pruning occurs)
        with a g1 column
    """

    if "g1" in prune_df.columns:
        raise ValueError("prune_df shouldn't already have a g1 column")

    return prune_df.assign(g1 = lambda df: (df["R({t})"] - df["R(T_t)"])/\
                                                (df["|T_t|"] - 1))




def test_append_g1():
    """
    tests _append_g1

    static examples
    """
    prune_df = pd.DataFrame({"g1": np.arange(7)})

    error_obs = False
    try:
        _append_g1(prune_df)
    except:
        error_obs = True

    assert error_obs, \
        "should get an error if prune_df already has a g1 when using " + \
        "_append_g1"


    prune_df = pd.DataFrame({"R({t})": np.array([1,2,3]),
                            "R(T_t)": np.array([1, 1,3]),
                            "|T_t|" : np.array([1, 2,4], dtype = np.int)})

    out_df = _append_g1(prune_df)

    g1 = out_df.pop("g1")

    assert np.all(out_df == prune_df), \
        "all columns but 'g1' should be the same in input and output of "+\
        "_append_g1."

    assert np.all(g1[1:] == np.array([1,0])), \
        "g1 calculated incorrectly in static example (non NaN)"

    assert np.isnan(g1[0]), \
        "g1 calculated incorrectly in static example (NaN)"


def _append_oob_values(prune_df, tree, X_trained, y_trained):
    """
    "R_oob({t})" and "n_oob" only? - oob_risk will be calculated with these
    """
    if type(tree) is sklearn.tree.tree.DecisionTreeRegressor:
        rf_type = "reg"
    elif type(tree) is sklearn.tree.tree.DecisionTreeClassifier:
        rf_type = "class"
    else:
        raise ValueError("tree needs either be a "+\
                   "sklearn.tree.tree.DecisionTreeClassifier "+\
                   "or a sklearn.tree.tree.DecisionTreeRegressor")

    n_obs_trained = X_trained.shape[0]

    random_state = tree.random_state
    oob_indices = \
        sklearn.ensemble.forest._generate_unsampled_indices(
                                                         random_state,
                                                         n_obs_trained)
    X_tune = X_trained[oob_indices,:]
    y_tune = y_trained[oob_indices]

    if rf_type == "class":
        y_tune = np.array(pd.get_dummies(y_tune))

    oob_decision_path = tree.decision_path(X_tune)
    n_oob = np.array(oob_decision_path.sum(axis = 0)).ravel()

    if rf_type == "class":
        # l_0 loss
        oob_values = oob_decision_path.T @ y_tune

        tree_values = tree.tree_.value
        tree_values = tree_values.reshape((tree_values.shape[0],
                                           tree_values.shape[2]))
        column_choice = tree_values.argmax(axis = 1)
        oob_error = oob_values[np.arange(oob_values.shape[0]),
                              column_choice] / n_oob

        # cleaning up oob_error's nans
        oob_error[np.isnan(oob_error)] = 0

        oob_error = np.array(oob_error)

    else:
        # l_2 loss
        # decision_path (row times) y_tune (column subtraction) tree_values (column sum^2)
        # ^ this is dangerous due to differences with np.array, np.matrix,
        #   and scipy.sparse

        tree_values = tree.tree_.value.ravel()

        dense_decision_path = np.array(oob_decision_path.todense())
        diff = (dense_decision_path.T * y_tune).T - tree_values
        diff[dense_decision_path == 0] = 0
        diff2 = diff**2
        oob_error = diff2.mean(axis = 0)


    out_df = pd.concat((prune_df, pd.DataFrame({"n_oob": n_oob,
                                                "R_oob({t})" : oob_error})))

    return out_df




def test_append_oob_values_regression():
    #COME HERE
    "checking oob calculations..."

def test_append_oob_values_classification():
    #COME HERE
    # also should be updating testing for _initialize_prune_df
    #
    "checking oob calculations..."
    "oob values classification"

def _recursive_initial_prune_df(prune_df, current_idx):
    """
    recursively create the initial prune_df, by walking down the tree
    and updating children and then sharing information with parent

    Arguments:
    ----------
    prune_df : pd.DataFrame
        data frame with correct values except |T_t| and R(T_t) - to be updated
    current_idx : int
        which node we are currently processing

    Returns:
    --------
    prune_df : pd.DataFrame
        data frame with correct information about each node (before any pruning
        occurs) - relative to children of and node relative to current_idx
    """
    # leaf
    if (prune_df.loc[prune_df.idx == current_idx, "c_left"] == -1).values:
        prune_df.loc[prune_df.idx == current_idx,"|T_t|"] = 1
    else:  # non-leaf
        #pdb.set_trace()
        prune_df = _recursive_initial_prune_df(prune_df,
            np.int(prune_df.c_left[prune_df.idx == current_idx]))
        prune_df = _recursive_initial_prune_df(prune_df,
            np.int(prune_df.c_right[prune_df.idx == current_idx]))

        prune_df.loc[prune_df.idx == current_idx, "|T_t|"] =\
            prune_df.loc[prune_df.c_left[prune_df.idx == current_idx], "|T_t|"].values +\
            prune_df.loc[prune_df.c_right[prune_df.idx == current_idx], "|T_t|"].values
        prune_df.loc[prune_df.idx == current_idx,"R(T_t)"] =\
            (prune_df.loc[prune_df.c_left[prune_df.idx == current_idx], "R(T_t)"].values *\
             prune_df.loc[prune_df.c_left[prune_df.idx == current_idx],"n_obs"].values +\
             prune_df.loc[prune_df.c_right[prune_df.idx == current_idx], "R(T_t)"].values *\
             prune_df.loc[prune_df.c_right[prune_df.idx == current_idx],"n_obs"].values) /\
            prune_df.loc[prune_df.idx == current_idx, "n_obs"].values

    return prune_df

def r_single_t(tree):
    """
    Calculates R({t}) for each node

    Argument:
    ---------
    tree : sklearn style tree (DecisionTreeClassifier or DecisionTreeRegressor)
        grown tree

    Returns:
    --------
    r_single_t_val : array (num_nodes)
        see details (calculation of R({t}) per node)

    Details:
    --------
    R({t}) is the cost for for each node (classification or regression error)
    Notes this is either (MSE or missclassification rate)
    """

    if type(tree) is sklearn.tree.tree.DecisionTreeRegressor:
        n_nodes = tree.tree_.weighted_n_node_samples

        r_single_t = tree.tree_.impurity
    elif type(tree) is sklearn.tree.tree.DecisionTreeClassifier:
        n_nodes = tree.tree_.weighted_n_node_samples

        values = tree.tree_.value.copy()
        a_max = values.argmax(axis = 2).ravel()

        values[np.arange(a_max.shape[0]), 0, a_max] = 0

        r_single_t = np.sum(values, axis = 2).ravel() / n_nodes
    else:
        raise ValueError("tree needs either be a "+\
                   "sklearn.tree.tree.DecisionTreeClassifier "+\
                   "or a sklearn.tree.tree.DecisionTreeRegressor")


    return r_single_t


def calc_parent(tree):
    """
    calculate who's the parent of each node

    Argument:
    ---------
    tree : sklearn style tree (DecisionTreeClassifier or DecisionTreeRegressor)
        grown tree

    Returns:
    --------
    parents : array (num_nodes,)
        integer of who the parent of the node is (-1 means no parent/root)

    """
    n = tree.tree_.children_left.shape[0]
    c_left = pd.DataFrame(data = {"idx" : np.arange(n, dtype = np.int),
                                  "c" : tree.tree_.children_left})
    c_right = pd.DataFrame(data = {"idx" : np.arange(n, dtype = np.int),
                                   "c" : tree.tree_.children_right})

    c_right = c_right[c_right.c != -1]
    c_left = c_left[c_left.c != -1]

    parents = -1 * np.ones(n, dtype = np.int)
    parents[c_left.c] = c_left.idx
    parents[c_right.c] = c_right.idx

    return parents


#def update_pure_df(queue, df):
