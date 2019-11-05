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
from smooth_rf import smooth_base, adam_sgd

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

    def get_G_fill(i, G_fill, n_trees):
        G_fill_val = G_fill[np.arange(n_trees*i,
                                      n_trees*(i+1), dtype = np.int)]
        G_fill_val[G_fill_val == 0] = 1
        return 1/G_fill_val

    def get_G_over_H2_fill(i, G_fill, H_fill, n_trees):
        G_fill_val = G_fill[np.arange(n_trees*i,
                                      n_trees*(i+1), dtype = np.int)]

        H_fill_val = H_fill[np.arange(n_trees*i,
                                      n_trees*(i+1), dtype = np.int)]

        H_fill_val[H_fill_val == 0] = 1
        return G_fill_val/H_fill_val**2

    star = scipy.sparse.block_diag([get_G_fill(i, G_fill, n_trees)
                        for i in np.arange(obs_idx.shape[0], dtype = np.int)])

    star2 = scipy.sparse.block_diag(
                [get_G_over_H2_fill(i, G_fill, H_fill, n_trees)
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

    idx_mat, oob_weights = generate_oob_info(forest, X_trained)

    Gamma, eta, _ = smooth_base.create_Gamma_eta_forest(fit_reg,
                                                        parents_all=True)

    lamb = np.zeros(Gamma.shape[1])
    lamb[0] = 1

    for _ in np.arange(5):
        obs_idx = np.random.choice(X_trained.shape[0], size = 20,
                                   replace = False)
        grad = calc_l2_grad(lamb, Gamma, eta, idx_mat,
                            oob_weights, obs_idx, y_trained)

        assert grad.shape[0] == Gamma.shape[1], \
            "expected dimensions of gradient should be (K,)"


def calc_cost_l2_straight(y_oob, y_trained, ever_oob):
    """
    calc sum_i 1/(n_oob) |y_oob(i) - y_trained(i)|^2
    """
    return np.mean((y_oob[ever_oob] - y_trained[ever_oob])**2)



def l2_s_grad_for_adam_wrapper_clean(Gamma, eta,
                                     idx_mat, oob_weights, obs_idx,
                                     y_trained):
    """
    Stocastic gradient for l2 loss to be inserted into adam_step (clean)

    gradient relative to mean_i |y_oob(i)-y(i)|^2

    Arguments:
    ----------
    Gamma : array (Tn, K)
        A horizontal stacking of Gamma matrices as defined above for all trees
        in the forest. See details below
    eta : array (Tn, K)
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
    y_trained : array (n, )
        y data array used to create the inputted random_forest

    Returns:
    --------

    l2_sg_adam : lambda function
        lambda function of take_gradient function whos only is input is lamb
    """

    l2_sg_adam = lambda lamb: calc_l2_grad(lamb, Gamma, eta,
                                           idx_mat, oob_weights, obs_idx,
                                           y_trained)

    return l2_sg_adam

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

    idx_mat, oob_weights = generate_oob_info(forest, X_trained)

    Gamma, eta, _ = smooth_base.create_Gamma_eta_forest(fit_reg,
                                                        parents_all=True)

    for _ in range(5):
        obs_idx = np.random.choice(X_trained.shape[0], size = 30)
        wrapper_funct = l2_s_grad_for_adam_wrapper_clean(Gamma, eta,
                                         idx_mat, oob_weights, obs_idx,
                                         y_trained)

        for _ in range(10):
            lamb = np.random.uniform(size = Gamma.shape[1])
            lamb = lamb / lamb.sum()

            g_straight = calc_l2_grad(lamb, Gamma, eta,
                                               idx_mat, oob_weights, obs_idx,
                                               y_trained)
            g_wrapper = wrapper_funct(lamb)

            assert np.all(g_straight == g_wrapper), \
                "l2 wrapper function should return same values at object it wraps"



def stocastic_grad_descent_clean(Gamma, eta, idx_mat, oob_weights, y_trained,
                                 lamb_init,
                                 n_obs=20, use_full_loss=False,
                                 t_fix=1, num_steps=10000,
                                 no_constraint=False,verbose=True,
                                 adam=None):
    """
    Preform stocastic gradient descent to minimize the l2 defined by

    sum_i (y_oob(i) - y)^2

    The stocastic gradient steps randomly select a subset of observations for
    each step to estimate the gradient with.

    Arguments:
    ----------
    Gamma : array (Tn, K)
        A horizontal stacking of Gamma matrices as defined above for all trees
        in the forest. See details below
    eta : array (Tn, K)
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
    y_trained : array (n, )
        y data array used to create the inputted random_forest
    lamb_init : array (K,)
        inital lambda value
    n_obs : int
        number of observations for each step of stocastic gradient descent
    use_full_loss : bool
        logic to use and track full loss (even if doing sgd). If all_obs is
        TRUE, then this will be corrected to being true.
    num_steps : int
        number of steps for stocastic gradient descent to take (default 1000)
    no_constraint : bool
        logic for if the space for lamb values is constrainted onto the simplex
    verbose : bool
        logic to show steps of stocastic gradient descent
    adam : dict (default is None)
        dictionary for input parameters adam SGD (if None, regular SGD is used)
        Note expected structure looks like:
            {"alpha": .001, "beta_1": .9, "beta_2": .999,"eps": 1e-8}
    Returns:
    --------
    lamb_best, lamb, cost_all

    Details:
    --------
    *Gamma* and *eta*
    Gamma and eta matrices are from a forest (aka set of trees, where
    these two matrices are defined (per tree):

    Gamma_il = sum_j II(D_ij =(or <=) l) n_j y_j
    eta_il = sum_j II(D_ij =(or <=) l) n_j

    where n_j, y_j are the number of observations in leaf j and the predicted
    value associated with with leaf j. Note that D_ij is the tree based
    distance between leaf i and j.
    """

    if Gamma.shape[1] != lamb_init.shape[0]:
        raise TypeError("lamb_init needs to be the same length as the "+\
                        "number of columns in Gamma and eta")
    if not no_constraint and (np.sum(lamb_init)!=1 or np.any(lamb_init < 0)):
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

    cost_all = []

    for s_step in first_iter:
        # select obs idx
        obs_idx = np.random.choice(idx_mat.shape[0],
                                   size=n_obs, replace=False)

        # initial cost calculation
        if s_step == 0:
            if use_full_loss:
                y_oob, ever_oob = calc_y_oob(lamb, Gamma, eta,
                                        idx_mat, oob_weights,
                                        np.arange(idx_mat.shape[0],
                                                  dtype = np.int))
                cost_best = calc_cost_l2_straight(y_oob,
                                      y_trained,
                                      ever_oob)
                cost_all.append(cost_best)
                lamb_best = lamb
            else:
                y_oob, ever_oob = calc_y_oob(lamb, Gamma, eta,
                                             idx_mat, oob_weights, obs_idx)
                cost_all.append(
                    calc_cost_l2_straight(y_oob,
                                          y_trained[obs_idx],
                                          ever_oob))


        if reg_sgd: # regular sgd
            grad = calc_l2_grad(lamb, Gamma, eta,
                                idx_mat, oob_weights, obs_idx,
                                y_trained)

            lamb = lamb - t_fix * grad
        else: # adam sgd
            take_gradient_adam = l2_s_grad_for_adam_wrapper_clean(
                                    Gamma, eta,
                                    idx_mat, oob_weights, obs_idx,
                                    y_trained)
            if s_step == 0:
                iv = None
            lamb, iv = adam_sgd.adam_step(grad_fun = take_gradient_adam,
                                          lamb_init = lamb,
                                          internal_values = iv,
                                          **adam)

        if not no_constraint:
           lamb = smooth_base.prox_project(lamb)

        if use_full_loss:
            y_oob, ever_oob = calc_y_oob(lamb, Gamma, eta,
                                    idx_mat, oob_weights,
                                    np.arange(idx_mat.shape[0],
                                              dtype = np.int))
            cost_new = calc_cost_l2_straight(y_oob,
                                  y_trained,
                                  ever_oob)
            cost_all.append(cost_new)

            if cost_new < cost_new:
                cost_best = cost_new
                lamb_best = lamb
        else:
            y_oob, ever_oob = calc_y_oob(lamb, Gamma, eta,
                                         idx_mat, oob_weights, obs_idx)
            cost_all.append(
                calc_cost_l2_straight(y_oob,
                                      y_trained[obs_idx],
                                      ever_oob))

    if not use_full_loss:
        lamb_best = lamb


    return lamb_best, lamb, cost_all


# 1. allow for keeping track of full loss if desired
# 2. impliment adam
# 3. do without constraint
# 4. do sanity check
# 5. allow for stocastic and full grad descent
# (basically have same structure as smooth() from smooth_base)
#
# this looks like a major (even though it's just a copy of the last code basically)

def smooth_clean(random_forest,
                 X_trained, y_trained,
                 verbose=True,
                 no_constraint=False, sanity_check=False,
                 sgd_max_num=10000, sgd_t_fix=1,
                 all_obs=False, use_full_loss=False,
                 sgd_n_obs=50,
                 initial_lamb_seed=None,
                 parents_all=False,
                 dist_mat_style=["standard","max", "min"],
                 distance_style=["depth", "impurity"],
                 levels=None,
                 adam=None):
    """
    creates a smooth random forest (1 lambda set across all trees)

    - CURRENTLY CODED FOR REGRESSION ONLY.

    this version uses y_oob vs y as it's loss

    Args:
    ----
    random_forest : sklearn forest
            (sklearn.ensemble.forest.RandomForestRegressor or
            sklearn.ensemble.forest.RandomForestClassifier)
        pre-trained classification or regression based random forest
    X_trained : array (n, p)
        X data array used to create the inputted random_forest. Note that this
        is assumed to be the correct data - and is used for the smoothing.
    y_trained : array (n, )
        y data array used to create the inputted random_forest
    verbose : bool
        logic to show tree analysis process
    no_constraint : bool
        logic to not constrain the weights
    sanity_check : bool
        logic to do full process but keep same weights
    sgd_max_num : int
        number of steps to take for the stocastic gradient optimization
    sgd_t_fix : scalar
        value for fixed t step size for stocastic gradient descent
    all_obs : bool
        logic to use all observations (and therefore do full gradient descent)
    use_full_loss : bool
        logic to use and track full loss (even if doing sgd). If all_obs is
        TRUE, then this will be corrected to being true.
    sgd_n_obs : int
        number of observations for each stocastic gradient descent step - this
        is overrided if all_obs is TRUE.
    initial_lamb_seed : scalar
        initial value for seed (default is None) to randomly select the
        starting point of lambda
    parents_all : bool
        logic to instead include all observations with parent of distance k
        away
    distance_style : string
        style of inner-tree distance to use, see *details* in the
        create_distance_mat_leaves doc-string.
    dist_mat_style : string
        style of inner-tree distance matrix to use, see *details* in the
        create_distance_mat_leaves doc-string.
    distance_style : string
        distance style (use depth or difference in impurity) see *details*
        in the create_distance_mat_leaves doc-string.
    levels : int, or array or None
        levels to discretize distance into (see *details* in the
        create_distance_mat_leaves doc-string.)
    adam : dict (default is None)
        dictionary for input parameters adam SGD (if None, regular SGD is used)
        Note expected structure looks like:
            {"alpha": .001, "beta_1": .9, "beta_2": .999,"eps": 1e-8}

    Returns:
    --------
    inner_rf : RandomForestClassifier or RandomForestRegressor
        updated smoothed random forest with optional lambda weighting, also has
        a .lamb parameter that stores weighting
    c : list
        list of cost value for each stocastic gradient descent (full or partial
         depending upon use_full_loss)
    """
    if type(random_forest) is sklearn.ensemble.RandomForestClassifier:
        rf_type = "class"
        raise ValueError("currently just coded for regression case.")
    elif type(random_forest) is sklearn.ensemble.RandomForestRegressor:
        rf_type = "reg"
    else:
        raise ValueError("random_forest needs to be either a " +\
                   "sklearn.ensemble.RandomForestClassifier " +\
                   "or a sklearn.ensemble.RandomForestRegressor")


    if rf_type == "class" and no_constraint == True:
        raise ValueError("for probabilities for the smoothed classification "+\
                         "rf to be consistent, you need to constrain the " +\
                         "lambda in the simplex (no_constraint = False) - " +\
                         "or at least have only positive values.")





    inner_rf = copy.deepcopy(random_forest)

    forest = inner_rf.estimators_

    # getting structure from built trees
    Gamma, eta, _ = smooth_base.create_Gamma_eta_forest(random_forest,
                                            verbose=verbose,
                                            parents_all=parents_all,
                                            dist_mat_style=dist_mat_style,
                                            distance_style=distance_style,
                                            levels=levels)

    idx_mat, oob_weights = generate_oob_info(forest, X_trained)

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

    # if rf_type == "class": # to avoid initial problems in cross-entropy loss
    #     lamb = lamb + np.ones(lamb.shape[0]) * class_eps / lamb.shape[0]
    #     lamb = lamb / lamb.sum()
    #     lamb = lamb / lamb.sum() # need to do twice...

    if all_obs:
        sgd_n_obs = X_trained.shape[0]
        use_full_loss = True

    # if rf_type == "class":
    #     Gamma_shape = Gamma.shape
    #     num_classes = Gamma.shape[0]
    #     Gamma = Gamma.reshape((Gamma.shape[0]*Gamma.shape[1],
    #                                  Gamma.shape[2]))

    #     eta = np.tile(eta, (num_classes,1))
    #     y_all = y_all.T.reshape((-1,))
    #     weight_all = np.tile(weight_all, num_classes)
    #     t_idx_vec = np.tile(t_idx_vec, num_classes)

    if not sanity_check:
        if rf_type == "reg":
            lamb, lamb_last, c = stocastic_grad_descent_clean(
                                Gamma, eta, idx_mat,
                                oob_weights, y_trained,
                                n_obs=sgd_n_obs, use_full_loss=use_full_loss,
                                lamb_init=lamb, t_fix=sgd_t_fix,
                                num_steps=sgd_max_num,
                                no_constraint=no_constraint,verbose=verbose,
                                adam=adam)
    else:
        c = []
        # for sanity check
        lamb = np.zeros(Gamma.shape[-1])
        lamb[0] = 1


    #---
    # update random forest object (correct estimates from new lambda)
    #---
    # to avoid divide by 0 errors (this may be a problem relative to the
    #   observed values)


    eta_fill = (eta @ lamb)
    eta_fill[eta_fill == 0] = 1
    y_leaf_new_all = (Gamma @ lamb) / eta_fill
    y_leaf_new_all[(eta @ lamb) == 0] = 0

    if rf_type == "class":
        y_leaf_new_all = y_leaf_new_all.reshape((-1,num_classes),
                                                order = "F")
                                                # ^order by column, not row

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

    return inner_rf, c




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
        a,b = smooth_clean(random_forest,
                 X_trained, y_trained,
                 verbose=verbose,
                 sgd_max_num=n_steps,
                 parents_all=parents_all,
                 dist_mat_style=dist_mat_style)

    except:
        assert False, \
            "error running smoothing_clean for a random forest regressor"

    # sanity check
    a,b = smooth_clean(random_forest,
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
        a,b = smooth_clean(random_forest, X_trained, y_trained,
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
        a,b = smooth_clean(random_forest, X_trained, y_trained,
                                    sgd_max_num=n_steps,
                                    parents_all=parents_all, verbose=verbose,
                                    dist_mat_style=dist_mat_style,
                                    adam={"alpha": .001, "beta_1": .9,
                                            "beta_2": .999,"eps": 1e-8})

    except:
        assert False, \
            "error running smoothing_function for a random forest regressor"
