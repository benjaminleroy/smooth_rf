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

    return out_df

def _initialize_prune_df(tree):
    """
    inner function for initializing prune_df (no pruning)

    Arguments:
    ----------
    tree : sklearn style tree (DecisionTreeClassifier or DecisionTreeRegressor)
        grown tree

    Returns:
    --------
    prune_df : pd.DataFrame
        data frame with information about each node (before any pruning occurs)
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
                                    "n_obs":n_obs},
                            columns = ["idx", "R({t})", "R(T_t)",
                                       "c_left", "c_right", "|T_t|",
                                       "parent", "n_obs"])


    prune_df = _recursive_initial_prune_df(prune_df, current_idx = 0)

    return prune_df



def _recursive_initial_prune_df(prune_df, current_idx):
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
    Notes this is either (MSE or missclaffication rate)
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
