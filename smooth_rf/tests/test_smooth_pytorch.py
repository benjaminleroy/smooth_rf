import numpy as np
import pandas as pd
import scipy.sparse
import sparse
import sklearn
from sklearn.ensemble import RandomForestRegressor
import sklearn.datasets
import sys, os

from collections import Counter

import torch

import smooth_rf

#from trees_for_test import moon_fix

def test_pytorch_numpy_prep():
    """
    test for pytorch_numpy_prep
    """
    data, y = sklearn.datasets.make_moons(n_samples=350, noise=.3)

    data_test, y_test = sklearn.datasets.make_moons(10000, noise=.3)

    model_type = sklearn.ensemble.RandomForestClassifier

    n_tree = 10
    model = model_type(n_estimators=n_tree)
    model_fit = model.fit(data, y)
    random_forest = model_fit

    y_all, g, n, weights_all, t, \
    one_d_dict, two_d_dict, lamb_dim, num_classes \
    = smooth_rf.pytorch_numpy_prep(random_forest,
                         X_trained=data,
                         y_trained=y,
                         verbose=False)

    # gamma, eta, t_idx structure
    assert g.shape == n.shape, \
        "Gamma and eta matrices are not the correct shared size"
    assert g.shape[0] == t.shape[0], \
        "the tree index vector doesn't have the correct number of observations"

    assert np.all(
        np.array(list(dict(Counter(t)).keys())) == np.arange(n_tree)),\
        "tree index doesn't contain expected tree index values"

    Gamma_inner, eta_inner, _, _, _, _, _, _ = \
        smooth_rf.create_Gamma_eta_forest_more(random_forest,verbose = False)

    assert (Gamma_inner.shape[0]*Gamma_inner.shape[1], Gamma_inner.shape[2])==\
        g.shape, \
        "Gamma shape doesn't match the transformation of "+\
        "create_Gamma_eta_forest_more function output"

    assert n.shape == g.shape, \
        "eta and gamma shape should be same shape"

    assert np.all(n >= 0) and np.all(np.ceil(n) == n), \
        "eta should be all non-negative integers"

    assert np.all(g >= 0) ,\
        "gamma should be non-negative" #classification


    # y_all, weights_all structure
    assert y_all.shape == weights_all.shape and \
        y_all.shape == t.shape, \
        "y and/or weights don't have to correct dimensions"
    assert np.all(weights_all >= 0) and \
        np.all(np.ceil(weights_all) == weights_all), \
        "weights should be all positive integers" # non-zero for regression
    # assert np.min(y_all) >= np.min(y) and \ #for regression
    #     np.max(y_all) <= np.max(y), \
    #     "leaf y values are constrained by true values"
    assert np.min(y_all) >= 0 and np.max(y_all) <= 1,\
        "y are all probability values, and should be bounded"
         # ^classification

    # lamb_dim and num_classes structure
    assert num_classes == random_forest.n_classes_, \
        "number of classes doesn't match inputed random forest"
    assert lamb_dim == g.shape[-1], \
        "lamb_dim doesn't match gamma structure"

    # one_d_dict, two_d_dict structure
    if one_d_dict is not None:
        assert type(one_d_dict) == dict, \
            "one_d_dict should be a dictionary or None"
        for key, X in one_d_dict.items():
            assert X.shape == (y_all.shape[0], ), \
                "%s's data entry in one_d_dict doesn't have the correct shape" %key
    if two_d_dict is not None:
        assert type(two_d_dict) == dict, \
            "two_d_dict should be a dictionary or None"
        for key, X in two_d_dict.items():
            assert X.shape == (y_all.shape[0], lamb_dim), \
                "%s's data entry in two_d_dict doesn't have the correct shape" %key


def test_ForestDataset():
    """
    tests for ForestDataset

    check of some attributes, model creation and forward function
    """
    data, y = sklearn.datasets.make_moons(n_samples=350, noise=.3)

    data_test, y_test = sklearn.datasets.make_moons(10000, noise=.3)

    model_type = sklearn.ensemble.RandomForestClassifier

    n_tree = 10
    model = model_type(n_estimators=n_tree)
    model_fit = model.fit(data, y)
    random_forest = model_fit

    y_all, g, n, weights_all, t, \
    one_d_dict, two_d_dict, lamb_dim, num_classes \
    = smooth_rf.pytorch_numpy_prep(random_forest,
                         X_trained=data,
                         y_trained=y,
                         verbose=False)
    # general catch
    #(also implicitly checks if __create_softmax_structure errors)
    try:
        myForest = smooth_rf.ForestDataset(y_all, g, n, weights_all, t,
                        one_d_dict, two_d_dict, lamb_dim)
    except:
        assert False, \
            "error creating smooth_rf.ForestDataset"

    assert myForest.num_trees == n_tree and \
        len(myForest) == n_tree and \
        myForest.lamb_dim == lamb_dim and \
        myForest.n_obs == n.shape[0], \
        "basic attributes (num_trees, lamb_dim, n_obs), don't match input"

    num_vars = len(one_d_dict) + len(two_d_dict)


    for _ in range(n_tree):
        idx = np.random.choice(n_tree)

        num_leaves = np.sum(t == idx)
        y_item, Gamma_item, eta_item, weights_item, \
            softmax_structure_item = myForest[idx]

        assert y_item.shape == (num_leaves),\
            "y output shape is not correct"
        assert Gamma_item.shape == (num_leaves,g.shape[1]),\
            "Gamma output shape is not correct"
        assert eta_item.shape == Gamma_item.shape,\
            "eta output shape is not correct"
        assert weights_item.shape == (num_leaves),\
            "weights output shape is not correct"

        assert softmax_structure_item.shape == (num_leaves,
                                                num_vars * lamb_dim),\
            "softmax structure shape is incorrect"


def test_SoftmaxTreeFit():
    """
    test SoftmaxTreeFit creation
    """

    # model structure
    num_vars = 5
    lamb_dim = 6

    smoothing_model = smooth_rf.SoftmaxTreeFit(num_vars = num_vars,
                                               lamb_dim = lamb_dim)

    assert len(smoothing_model.linear_list) == 6, \
        "number of linear models is not correct"

    param_list = list(smoothing_model.parameters())

    assert len(param_list) == lamb_dim * 2,\
        "model parameters length should be 2 times linear model length, " +\
        "due to softmax construction"

    for p_idx, param in enumerate(param_list):
        if p_idx % 2 == 1:
            assert len(param) == 1, \
                "softmax parameter structure is violated"
        if p_idx % 2 == 0:
            assert len(param[0]) == num_vars, \
                "linear model structure is violated"

    # forward step
    data, y = sklearn.datasets.make_moons(n_samples=350, noise=.3)

    data_test, y_test = sklearn.datasets.make_moons(10000, noise=.3)

    model_type = sklearn.ensemble.RandomForestClassifier

    n_tree = 10
    model = model_type(n_estimators=n_tree)
    model_fit = model.fit(data, y)
    random_forest = model_fit

    y_all, g, n, weights_all, t, \
    one_d_dict, two_d_dict, lamb_dim, num_classes \
    = smooth_rf.pytorch_numpy_prep(random_forest,
                         X_trained=data,
                         y_trained=y,
                         verbose=False)

    myForest = smooth_rf.ForestDataset(y_all, g, n, weights_all, t,
                        one_d_dict, two_d_dict, lamb_dim)

    num_vars = len(one_d_dict) + len(two_d_dict)
    smoothing_model = smooth_rf.SoftmaxTreeFit(num_vars = num_vars,
                                               lamb_dim = lamb_dim)

    for _ in range(n_tree):
        idx = np.random.choice(n_tree)

        y_item, Gamma_item, eta_item, weights_item, \
            softmax_structure_item = myForest[idx]

        # it need to have first dimension 1 for sampling structure...
        y_item = y_item.reshape((1, y_item.shape[0]))
        Gamma_item = Gamma_item.reshape((1, *Gamma_item.shape))
        eta_item = eta_item.reshape((1, *eta_item.shape))
        weights_item = weights_item.reshape((1, weights_item.shape[0]))
        softmax_structure_item = softmax_structure_item.reshape((1, *softmax_structure_item.shape))

        yhat_item, weights_item = smoothing_model.forward(
                                        (y_item, Gamma_item,
                                         eta_item, weights_item,
                                         softmax_structure_item))

        assert yhat_item.shape == (1,np.sum(t == idx),1), \
            "yhat shape should be the same size as number of leaves in trees"

        assert yhat_item.shape == (*weights_item.shape,1), \
            "yhat and weights should have the same shape" # odd...

def test_weighted_l2():
    """
    test of weighted_l2 (for torch objects)
    """
    for _ in range(5):
        y, y_pred, weights = torch.rand(5), torch.rand(5), torch.rand(5)
        loss = smooth_rf.weighted_l2(y,y_pred,weights).item()
        assert loss == np.sum(weights.numpy() * (y.numpy() - y_pred.numpy())**2),\
            "weighted l2 loss is different than expected"
def test_weighted_l2_np():
    """
    test of weighted_l2_np
    """
    for _ in range(5):
        y, y_pred, weights = torch.rand(5), torch.rand(5), torch.rand(5)
        loss = smooth_rf.weighted_l2(y,y_pred,weights).item()
        assert smooth_rf.weighted_l2_np(y.numpy(), y_pred.numpy(), weights.numpy()) == \
            loss, \
            "numpy version of weighted_l2 differs from torch version"
def test_l2_np():
    """
    test of l2_np
    """
    for _ in range(5):
        y, y_pred = np.random.uniform(5),np.random.uniform(5)
        loss = smooth_rf.l2_np(y,y_pred)
        assert loss == np.sum((y-y_pred)**2),\
            "l2_np doesn't correctly calculate l2 loss"


def test_acc_np():
    """
    test of acc_np
    """
    for _ in range(5):
        y, y_pred = np.random.choice(2,5),np.random.choice(2,5)
        loss = smooth_rf.acc_np(y,y_pred)
        assert loss == np.mean(y == y_pred),\
            "acc_np doesn't correctly calculate classification accuracy"


# def test_update_rf():
#     """
#     test update_rf
#     """

#     pass
#     # data, y = smooth_rf.generate_data(650)

#     # data_test, y_test = smooth_rf.generate_data(10000)

#     # model_type = sklearn.ensemble.RandomForestRegressor

#     # model = model_type(n_estimators=2)
#     # model_fit = model.fit(data, y)
#     # random_forest = model_fit

#     # max_iter = 10000

#     # smooth_rf_pytorch, loss_all, loss_min, params_min, best_model, \
#     #     (torch_model, forest_dataset, dataloader) = \
#     #     smooth_rf.smooth_pytorch(random_forest = random_forest,
#     #            X_trained=data, y_trained=y,
#     #            X_tune=None, y_tune=None,
#     #            resample_tune=False,
#     #            sgd_max_num=max_iter,
#     #            all_trees=False,
#     #            parents_all=True,
#     #            distance_style="standard",
#     #            which_dicts=["one_d_dict", "two_d_dict"],
#     #            verbose=False)

#     # y_pred_test_base = random_forest.predict(data_test)
#     # y_pred_test_smooth = smooth_rf_pytorch.predict(data_test)

#     # smooth_rf.acc_np(y_pred_test_base, y_test)
#     # smooth_rf.acc_np(y_pred_test_smooth, y_test)
