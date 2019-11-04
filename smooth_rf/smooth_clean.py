import numpy as np
import pandas as pd
import scipy.sparse
import sparse
import progressbar
import copy
import sklearn.ensemble
import sklearn
import pdb
import itertools

#from . import smooth_base
from smooth_rf import smooth_base

def generate_oob_info(forest, X_trained):
    """
    This function creates the idx_mat and oob information (w_i^t)

    idx_mat_{it}' = leaf index in tree t that observation i falls into
        (relative to ordering in a Gamma or eta matrix)
    idx_mat_{it} = idx_mat +_{column} [0, # leaves in tree 1,
                                            # leaves in tree 1 and 2, ...]
    and

    w_i^t' = Indicator if observation i is oob to tree t (in code called
    oob_weights)
    w_i^t = w_i^t' / sum_t(w_i^t')

    Arguments:
    ----------
    forest : .estimators_ from sklearn forest
            (sklearn.ensemble.forest.RandomForestRegressor or
             sklearn.ensemble.forest.RandomForestClassifier)
        grown forest's trees
    X_trained : array (n, p)
        X data array used to create the inputted random_forest. Note that this
        is assumed to be the correct data.

    Returns:
    --------
    idx_mat : array (n, T)
        array with information relative to which leaf of a tree each
        observation (row) falls under (defined in more details above)
    oob_weights : array (nT, )
        indicator vector that informs if observation falls into tree i.
        Specifically, value T*i + t is 1 if observation i was oob for tree t,
        and 0 otherwise.
    """

    n_trees = len(forest)
    n_obs = X_trained.shape[0]
    oob_weights_prime = np.zeros((n_obs, n_trees), dtype = np.int)
    idx_mat_prime = np.zeros((n_obs, n_trees), dtype = np.int)
    num_leaves_per_tree = np.zeros(n_trees)

    for t_idx, t in enumerate(forest):
        tree = t.tree_

        # number of leaves
        num_leaves_per_tree[t_idx] = np.sum(tree.children_left == -1)
        num_nodes = tree.children_left.shape[0]

        # oob_weights_prime
        random_state = t.random_state
        oob_indices = \
            sklearn.ensemble.forest._generate_unsampled_indices(
                                                             random_state,
                                                             n_obs)
        oob_weights_prime[oob_indices, t_idx] = 1

        # idx_mat_prime
        d_path = t.decision_path(X_trained)
        d_path_leaf = d_path[:, tree.children_left == -1]

        idx_mat_prime[:, t_idx] = d_path_leaf @ \
                                        np.arange(num_leaves_per_tree[t_idx],
                                                  dtype = np.int)


    # idx_mat
    cum_tree_size = np.array(np.cumsum(
                                np.array([0] +\
                                    list(num_leaves_per_tree)))[:n_trees],
                            dtype = np.int)
    idx_mat = idx_mat_prime + cum_tree_size

    # oob_weights
    oob_weights_counts = np.array(oob_weights_prime).sum(axis = 1)
    oob_weights_counts[oob_weights_counts == 0] = 1 #if never oob
    oob_weights = (np.array(oob_weights_prime).T / oob_weights_counts).T.ravel()

    return idx_mat, oob_weights

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

    idx_mat, oob_weights = generate_oob_info(forest, X_trained)

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

def create_G_H(Gamma, eta, idx_mat, obs_idx):
    """
    create G and H matrices for observations with index in obs_idx

    Arguments:
    ----------
    Gamma : array (sum_t (leafs in tree t), K)
        A horizontal stacking of Gamma matrices as defined above for all trees
        in the forest. See details below.
    eta : array (sum_t (leafs in tree t), K)
        A horizontal stacking of eta matrices as defined above for all trees
        in the forest.
    idx_mat : int array (n, T)
        array with information relative to which leaf of a tree each
        observation (row) falls under (defined in more details in
        smooth_rf.generate_oob_info documentation)
    obs_idx : int array (m, )
        indices of observations would like to examine

    Returns:
    --------
    G : array (mT, K)
        array that contains row blocks of G_i (for i in obs_idx) where
        G_i is an (T, K) matrix with information the same as Gamma, just for
        the observation's leaf it falls into for the tree.
    H : array (mT, K)
        array that contains row blocks of H_i (for i in obs_idx) where
        H_i is an (T, K) matrix with information the same as eta, just for
        the observation's leaf it falls into for the tree.

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

    current_idx_mat = idx_mat[obs_idx,]
    leaf_idx_ravel = np.array(current_idx_mat).ravel()

    G = Gamma[leaf_idx_ravel,:]
    H = eta[leaf_idx_ravel,:]

    return G, H

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

            G, H = create_G_H(Gamma, eta, idx_mat, obs_idx)

            assert G.shape == (inner_size*n_trees, K), \
                "shape of G should be (mT, K)"

            assert H.shape == (inner_size*n_trees, K), \
                "shape of G should be (mT, K)"

            assert np.all(Gamma[idx_mat[obs_idx[0],],:] == G[:n_trees,:]), \
                "values in G relative to first obs incorrect"

            assert np.all(eta[idx_mat[obs_idx[0],],:] == H[:n_trees,:]), \
                "values in H relative to first obs incorrect"


def calc_y_oob(lamb, Gamma, eta, idx_mat, oob_weights, obs_idx):
    """
    Calculate y_oob(i) (smoothed) values for observations with indices in
    obs_idx

    Arguments:
    ----------
    lamb : array (K,)
        weights associated with how important different distance away from the
        true node should effect the prediction
    Gamma : array (sum_t (leafs in tree t), K)
        A horizontal stacking of Gamma matrices as defined above for all trees
        in the forest. See details below.
    eta : array (sum_t (leafs in tree t), K)
        A horizontal stacking of eta matrices as defined above for all trees
        in the forest.
    idx_mat : int array (n, T)
        array with information relative to which leaf of a tree each
        observation (row) falls under (defined in more details in
        smooth_rf.generate_oob_info documentation)
    oob_weights : array (nT, )
        indicator vector that informs if observation falls into tree i.
        Specifically, value T*i + t is 1 if observation i was oob for tree t,
        and 0 otherwise (defined in more details in smooth_rf.generate_oob_info
        documentation).
    obs_idx : int array (m, )
        indices of observations would like to examine


    Returns:
    --------
    y_oob : array (m, )
        estimated values for obs_idx relative to oob value of algorithm
        *note: that if the obs_idx value never is oob, then you'll get 0 as
        the return value*
    ever_oob : boolean array (m, )
        index if the observation is every out of bag.
    """
    n_trees = idx_mat.shape[1]
    n_obs = obs_idx.shape[0]

    oob_weights_inner_first = [oob_weights[np.arange(n_trees*ob,
                                          n_trees*(ob+1),dtype = np.int)]
                                           for ob in obs_idx]

    ever_oob = np.array([np.sum(item) == 1 for item in oob_weights_inner_first])

    oob_weights_inner = np.array(list(itertools.chain(*oob_weights_inner_first)) )

    G_fill, H_fill =\
        calc_G_H_fill(lamb, Gamma, eta, idx_mat, obs_idx)

    # not dividing by 0 (should see that G_fill is also zero...)
    H_fill[G_fill == 0] = 1

    left_part = G_fill / H_fill

    y_oob = scipy.sparse.kron(scipy.sparse.identity(n_obs),
                              np.ones((n_trees,))) @\
            scipy.sparse.diags(oob_weights_inner) @\
        left_part

    return y_oob, ever_oob

def calc_G_H_fill(lamb, Gamma, eta, idx_mat, obs_idx):
    """
    calculate G_fill and H_fill for selected observations

    Arguments:
    ----------
    lamb : array (K,)
        weights associated with how important different distance away from the
        true node should effect the prediction
    Gamma : array (sum_t (leafs in tree t), K)
        A horizontal stacking of Gamma matrices as defined above for all trees
        in the forest. See details below.
    eta : array (sum_t (leafs in tree t), K)
        A horizontal stacking of eta matrices as defined above for all trees
        in the forest.
    idx_mat : int array (n, T)
        array with information relative to which leaf of a tree each
        observation (row) falls under (defined in more details in
        smooth_rf.generate_oob_info documentation)
    oob_weights : array (nT, )
        indicator vector that informs if observation falls into tree i.
        Specifically, value T*i + t is 1 if observation i was oob for tree t,
        and 0 otherwise (defined in more details in smooth_rf.generate_oob_info
        documentation).
    obs_idx : int array (m, )
        indices of observations would like to examine

    Returns:
    --------
    G_fill : (m, )
        defined as (G_i)^T lamb for each observatio
    H_fill : (m, )
        defined as (H_i)^T lamb for each observatio
    """
    G, H = create_G_H(Gamma, eta, idx_mat, obs_idx)

    G_fill =  G @ lamb
    H_fill = H @ lamb

    return G_fill, H_fill

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

    idx_mat, oob_weights = generate_oob_info(forest, X_trained)

    Gamma, eta, _ = smooth_base.create_Gamma_eta_forest(fit_reg, parents_all=True)

    lamb = np.zeros(Gamma.shape[1])
    lamb[0] = 1

    for _ in np.arange(5):
        obs_idx = np.random.choice(X_trained.shape[0], size = 20, replace = False)
        y_oob, ever_oob = calc_y_oob(lamb, Gamma, eta, idx_mat, oob_weights, obs_idx)

        assert np.all(y_oob[ever_oob] <= 101.001) and \
            np.all(y_oob[ever_oob] >= 100 -.001), \
            "test case values of y_oob should be around 101 to 100"

        assert (np.sum(ever_oob) == 0) or np.all(y_oob[ever_oob == False] == 0), \
            "every_oob logically relates to y_oob (returns y_oob == 0 "+\
            "when every_oob = False"

def calc_y_oob_grad(lamb, Gamma, eta, idx_mat, oob_weights, obs_idx):
    """
    calculate grad y_oob/ grad lamb relative to specific obs

    Arguments:
    ----------
    lamb : array (K,)
        weights associated with how important different distance away from the
        true node should effect the prediction
    Gamma : array (sum_t (leafs in tree t), K)
        A horizontal stacking of Gamma matrices as defined above for all trees
        in the forest. See details below.
    eta : array (sum_t (leafs in tree t), K)
        A horizontal stacking of eta matrices as defined above for all trees
        in the forest.
    idx_mat : int array (n, T)
        array with information relative to which leaf of a tree each
        observation (row) falls under (defined in more details in
        smooth_rf.generate_oob_info documentation)
    oob_weights : array (nT, )
        indicator vector that informs if observation falls into tree i.
        Specifically, value T*i + t is 1 if observation i was oob for tree t,
        and 0 otherwise (defined in more details in smooth_rf.generate_oob_info
        documentation).
    obs_idx : int array (m, )
        indices of observations would like to examine

    Returns:
    --------
    grad : array (K,)
        gradient of y_oob / grad lamb
    """

    n_trees = idx_mat.shape[1]
    n_obs = obs_idx.shape[0]

    oob_weights_inner_first = [oob_weights[np.arange(n_trees*ob,
                                          n_trees*(ob+1),dtype = np.int)]
                                           for ob in obs_idx]

    ever_oob = np.array([np.sum(item) == 1 for item in oob_weights_inner_first])

    oob_weights_inner = np.array(list(itertools.chain(*oob_weights_inner_first)) )


    G_fill, H_fill =\
        calc_G_H_fill(lamb, Gamma, eta, idx_mat, obs_idx)
    G, H = create_G_H(Gamma, eta, idx_mat, obs_idx)


    star = scipy.sparse.block_diag([1/G_fill[np.arange(n_trees*i,
                                          n_trees*(i+1),dtype = np.int)]
                        for i in np.arange(obs_idx.shape[0], dtype = np.int)])

    star2 = scipy.sparse.block_diag([G_fill[np.arange(n_trees*i,
                                          n_trees*(i+1),dtype = np.int)]/\
                                     H_fill[np.arange(n_trees*i,
                                          n_trees*(i+1),dtype = np.int)]**2
                        for i in np.arange(obs_idx.shape[0], dtype = np.int)])

    grad = star @ scipy.sparse.diags(oob_weights_inner) @ G -\
        star2 @ scipy.sparse.diags(oob_weights_inner) @ H

    return grad

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

    idx_mat, oob_weights = generate_oob_info(forest, X_trained)

    Gamma, eta, _ = smooth_base.create_Gamma_eta_forest(fit_reg, parents_all=True)

    lamb = np.zeros(Gamma.shape[1])
    lamb[0] = 1

    for _ in np.arange(5):
        obs_idx = np.random.choice(X_trained.shape[0], size = 20, replace = False)
        grad = calc_y_oob_grad(lamb, Gamma, eta, idx_mat, oob_weights, obs_idx)

        assert grad.shape == (20, Gamma.shape[1]), \
            "gradient should be (m x K)"

        _, ever_oob = calc_y_oob(lamb, Gamma, eta, idx_mat, oob_weights, obs_idx)

        assert np.sum(ever_oob == False) == 0 or \
            np.all(grad[ever_oob == False, ] == 0), \
            "gradient relative to any y_oob never out of bag is 0"

def calc_l2_grad(lamb, Gamma, eta, idx_mat, oob_weights, obs_idx, y):
    """
    calculate grad l_2/ grad lamb relative to specific obs

    Arguments:
    ----------
    lamb : array (K,)
        weights associated with how important different distance away from the
        true node should effect the prediction
    Gamma : array (sum_t (leafs in tree t), K)
        A horizontal stacking of Gamma matrices as defined above for all trees
        in the forest. See details below.
    eta : array (sum_t (leafs in tree t), K)
        A horizontal stacking of eta matrices as defined above for all trees
        in the forest.
    idx_mat : int array (n, T)
        array with information relative to which leaf of a tree each
        observation (row) falls under (defined in more details in
        smooth_rf.generate_oob_info documentation)
    oob_weights : array (nT, )
        indicator vector that informs if observation falls into tree i.
        Specifically, value T*i + t is 1 if observation i was oob for tree t,
        and 0 otherwise (defined in more details in smooth_rf.generate_oob_info
        documentation).
    obs_idx : int array (m, )
        indices of observations would like to examine

    Returns:
    --------
    grad : array (K,)
        gradient of l_2 / grad lamb
    """

    y_inner = y[obs_idx]

    y_oob, ever_oob = calc_y_oob(lamb, Gamma, eta, idx_mat, oob_weights, obs_idx)
    n_oob = np.sum(ever_oob)

    y_oob_grad = calc_y_oob_grad(lamb, Gamma, eta, idx_mat, oob_weights, obs_idx)

    grad = 2/n_oob * (y_oob - y_inner) @ y_oob_grad

    return grad

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

    idx_mat, oob_weights = generate_oob_info(forest, X_trained)

    Gamma, eta, _ = smooth_base.create_Gamma_eta_forest(fit_reg, parents_all=True)

    lamb = np.zeros(Gamma.shape[1])
    lamb[0] = 1

    for _ in np.arange(5):
        obs_idx = np.random.choice(X_trained.shape[0], size = 20, replace = False)
        grad = calc_l2_grad(lamb, Gamma, eta, idx_mat, oob_weights, obs_idx, y_trained)

        assert grad.shape[0] == Gamma.shape[1], \
            "expected dimensions of gradient should be (K,)"



def smooth_clean():
    # 1. allow for keeping track of full loss if desired
    # 2. impliment adam
    # 3. do without constraint
    # 4. do sanity check
    # 5. allow for stocastic and full grad descent
    # (basically have same structure as smooth() from smooth_base)
    #
    # this looks like a major (even though it's just a copy of the last code basically)
    pass
