import numpy as np
import pandas as pd
import scipy.sparse
import sparse
import progressbar
import copy
import sklearn.ensemble
import sklearn
import pdb
from collections import Counter
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, \
                                SequentialSampler
import torch.nn

import smooth_rf


def pytorch_numpy_prep(random_forest, X_trained, y_trained,
                       X_tune=None, y_tune=None,resample_tune=False,
                       train_only=False,
                       distance_style=["standard","max", "min"],
                       parents_all=True, verbose=True):
    """
    Prepares data structures to be used in the smooth random forest
    optimization. Specifically for the optimization that attempts to provide
    individual lambdas per each leaf as a function of the leaf's attributes

    Arguments:
    ----------
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
    resample_tune: bool
        logic to tune / optimize with another bootstrap same from X
    train_only : bool
        logic to deliver structure relative to base tree training creation
    distance_style : string
        style of inner-tree distance to use, see *details* in the
        create_distance_mat_leaves doc-string.
    parents_all : bool
        logic to instead include all observations with parent of distance k
        away
    class_eps : scalar (default 1e-4)
        For classification modles only. This scalar value makes it such that
        we keep the lamb values stay in the space defined by
        {x:x_i >= (class_eps / n)/(class_eps + 1), sum_i x_i =1}_i=1^n. This
        is done to not let the cross-entropy loss or it's derivative explode.
    verbose : bool
        logic to show tree analysis process

    Returns:
    --------
    y_all : array (TN*,)
        mean value of leaf (or proportion value for leaf for a class)
    Gamma : array (TN*, K)
        A horizontal stacking of Gamma matrices as defined above for all trees
        in the forest. (Regression and Classification is different)
    eta : array (TN*, K)
        A horizontal stacking of eta matrices as defined above for all trees
        in the forest.
    weight_all : array (TN*,)
        weights associated with each leaf (the number of observations that
        fall in that leaf)
    t_idx_vec : array (TN*,)
        integer array will holds which tree in the forest the associated row
        of Gamma_all or eta_all comes from
    one_d_dict : dict with arrays (TN*,)
        dictionary of one dimensional arrays that relate to the leaves
        attributes. (See comments for currently created 1d arrays.)
    two_d_dict : dict with arrays (TN*, K)
        dictionary of two dimensional arrays that relate to the leaves
        attributes. See comments for currently created 2d arrays.
    lamb_dim : int
        dimension of lambda / maximum depth of forest + 1 (value is equal to K)
    num_classes : int
        number of classes in the random forest model

    Comments:
    ---------
    If X_tune and/or y_tune is None then we will optimize each tree with oob
    samples.

    TN*: if the forest is a regression forest, then TN is the number of leafs
    across all the trees, if the forest is a classification forest then TN is
    the number of leaves across all the trees ~times~ the number of classes

    one_d_dict contains:
    ~~~~~~~~~~~~~~~~~~~~
    "leaf_n" : array (TN*,)
        number of observations in each leaf (same as weight_all actually)
    "leaf_d" : array (TN*,)
        depth of each leaf
    "leaf_i" : array (TN*,)
        impurity of each leaf (gini or mse for classification and regression
        respectively)

    two_d_dict contains:
    ~~~~~~~~~~~~~~~~~~~~
    "full_d" : array (TN*, K)
        depth of leaf's parents K distance away. After K > depth leaf, value
        is 0.
    "full_i" : array (TN*, K)
        impurity of leaf's parents K distance away (gini or mse for
        classification and regression respectively). After K > depth leaf,
        value is gini value of root node.

    """

    if type(random_forest) is sklearn.ensemble.RandomForestClassifier:
        rf_type = "class"
    elif type(random_forest) is sklearn.ensemble.RandomForestRegressor:
        rf_type = "reg"
    else:
        ValueError("random_forest needs to be either a " +\
                   "sklearn.ensemble.RandomForestClassifier " +\
                   "or a sklearn.ensemble.RandomForestRegressor")

    if (X_tune is None or y_tune is None) and not resample_tune and \
        not train_only:
        oob_logic = True
    else:
        oob_logic = False


    if (oob_logic or resample_tune or train_only) and \
        (X_trained is None or y_trained is None):
        raise TypeError("X_trained and y_trained need to be inserted for "+\
                        "provided input of X_tune/y_tune and resample_tune "+\
                        "parameters.")

    if oob_logic or resample_tune or train_only:
        n_obs_trained = X_trained.shape[0]

    _, max_depth = smooth_rf.calc_depth_for_forest(random_forest,verbose=False)
    max_depth = np.int(max_depth)

    forest = random_forest.estimators_

    Gamma, eta, t_idx_vec, leaf_n, leaf_d, leaf_i, full_d, full_i = \
        smooth_rf.create_Gamma_eta_forest_more(random_forest,
                                     verbose=verbose,
                                     parents_all=parents_all,
                                     dist_mat_style=distance_style)

    first_iter = forest
    if verbose:
        bar = progressbar.ProgressBar()
        first_iter = bar(first_iter)

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

        if train_only:
            random_state = t.random_state
            oob_indices = \
                sklearn.ensemble.forest._generate_sample_indices(random_state,
                                                                 n_obs_trained)
            X_tune = X_trained[oob_indices,:]
            y_tune = y_trained[oob_indices]

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


    if rf_type == "class":
        Gamma_shape = Gamma.shape
        num_classes = Gamma.shape[0]
        Gamma = Gamma.reshape((Gamma.shape[0]*Gamma.shape[1],
                                     Gamma.shape[2]))

        eta = np.tile(eta, (num_classes,1))
        full_d = np.tile(full_d, (num_classes,1))
        full_i = np.tile(full_i, (num_classes,1))

        y_all = y_all.T.reshape((-1,))
        weight_all = np.tile(weight_all, num_classes)
        t_idx_vec = np.tile(t_idx_vec, num_classes)
        leaf_n = np.tile(leaf_n, num_classes)
        leaf_d = np.tile(leaf_d, num_classes)
        leaf_i = np.tile(leaf_i, num_classes)

    else:
        num_classes = 1

    y_all, weight_all
    one_d_dict = {"leaf_n":leaf_n, "leaf_d":leaf_d, "leaf_i":leaf_i}
    two_d_dict = {"full_d":full_d,"full_i":full_i}
    n_obs = Gamma.shape[0]
    lamb_dim = Gamma.shape[-1]

    return y_all, Gamma, eta, weight_all, t_idx_vec, \
        one_d_dict, two_d_dict, lamb_dim, num_classes


def node_spatial_structure_update(random_forest, X_trained,
                                  one_d_dict=None,
                                  two_d_dict=None):
    """
    updates one_d_dict and/or two_d_dict with center of nodes

    Arguments:
    ----------
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
    one_d_dict : dict with arrays (TN*,)
        dictionary of one dimensional arrays that relate to the leaves
        attributes. (Default is None - if None no update is done)
    two_d_dict : dict with arrays (TN*, K)
        dictionary of two dimensional arrays that relate to the leaves
        attributes. (Default is None - if None no update is done)

    Returns:
    --------
    one_d_dict : dict
        updated one_d_dict with addition entries with keys of the form:
        "center0", ... , "centerK" (where K is max number of dimensions of X)
    two_d_dict : dict
        updated two_d_dict with addition entries with keys of the form:
        "center0", ... , "centerK" (where K is max number of dimensions of X)
    """

    if one_d_dict is None:
        if two_d_dict is None:
            ValueError("either 'one_d_dict' or 'two_d_dict' must not be "+\
                       "None")
        else:
            if type(two_d_dict) is not dict:
                ValueError("'two_d_dict' must be either None or a dictionary")

    if two_d_dict is None:
        if type(one_d_dict) is not dict:
                ValueError("'one_d_dict' must be either None or a dictionary")


    n_col = X_trained.shape[1]

    # analysis
    if one_d_dict is not None:
        # set up of each dimensions' center
        for col_num in range(n_col):
            one_d_dict["center"+str(col_num)] = np.zeros((0,))

        # iterate across trees
        for tree in random_forest.estimators_:
            center_t = smooth_rf.center_tree(tree, X_trained)
            cl = tree.tree_.children_left

            for col_num in range(n_col):
                one_d_dict["center"+str(col_num)] =\
                    np.concatenate((one_d_dict["center"+str(col_num)],
                                    center_t[cl == -1,:][:,col_num]))




    if two_d_dict is not None:
        _, max_depth = smooth_rf.calc_depth_for_forest(random_forest,
                                                       verbose = False) # techincally repeats 1/2 the calculations in depth_per_node_plus_parent
        max_depth = np.int(max_depth) + 1
        # set up of each dimensions' center
        for col_num in range(n_col):
            two_d_dict["center"+str(col_num)] = np.zeros((0,max_depth))

        # iterate across trees
        for t_idx, tree in enumerate(random_forest.estimators_):
            center_t = smooth_rf.center_tree(tree, X_trained)
            _, parent_mat = smooth_rf.depth_per_node_plus_parent(tree)

            cl = tree.tree_.children_left
            parent_mat = parent_mat[cl == -1,:]

            for col_num in range(n_col):
                addon = center_t[:,col_num][parent_mat]
                if parent_mat.shape[1] != max_depth:
                    diff = max_depth - parent_mat.shape[1]

                    extra = np.tile(
                                addon[:,(addon.shape[1]-1)].reshape((-1,1)),
                                (1, diff))
                    addon  = np.concatenate((addon, extra), axis = 1)

                two_d_dict["center"+str(col_num)] = \
                    np.concatenate((two_d_dict["center"+str(col_num)],
                                    addon), axis = 0)



    return one_d_dict, two_d_dict




class ForestDataset(Dataset):
    def __init__(self, y, Gamma, eta, weights, t_vec,
                 one_d_dict, two_d_dict, lamb_dim):
        """
        Intialize Forest dataset for pytorch

        Arguments:
        ----------
        y : array (TN*,)
            mean value of leaf (or proportion value for leaf for a class)
        Gamma : array (TN*, K)
            A horizontal stacking of Gamma matrices as defined above for all
            trees in the forest. (Regression and Classification is different)
        eta : array (TN*, K)
            A horizontal stacking of eta matrices as defined above for all
            trees in the forest.
        weights : array (TN*,)
            weights associated with each leaf (the number of observations that
            fall in that leaf)
        t_vec : array (TN*,)
            integer array will holds which tree in the forest the associated
            row of Gamma_all or eta_all comes from. This determines what the
            batches are for stocastic gradient descent.
        one_d_dict : dict with arrays (TN*,)
            dictionary of one dimensional arrays that relate to the leaves
            attributes.
        two_d_dict : dict with arrays (TN*, K)
            dictionary of two dimensional arrays that relate to the leaves
            attributes.
        lamb_dim : int
            dimension of lambda / maximum depth of forest + 1 (value is equal
            to K)

        Attributes:
        -----------
        __len__ : function that provides length of dataset, equal to
        the number of unique values in t_vec

        __getitem__ : gets all attributes related to a single tree (as defined
        in t_vec)
        """
        self.y = torch.from_numpy(y)
        self.Gamma = torch.from_numpy(Gamma)
        self.eta = torch.from_numpy(eta)
        self.weights = torch.from_numpy(weights)
        self.num_trees = len(dict(Counter(t_vec)).keys())
        self.t_vec = torch.from_numpy(t_vec)
        self.n_obs = eta.shape[0]
        self.lamb_dim = lamb_dim
        self.softmax_structure = self.__create_softmax_structure(one_d_dict,
                                                                 two_d_dict)
    #init data here

    def __len__(self):
        """
        number of leaves/number of batches for batch stocastic gradient descent
        """
        return   self.num_trees

    def __getitem__(self, idx):
        """
        gets a set of data structure for one batch

        Arguments:
        ----------
        idx : int
            integer within the range of 0 to __len__, identifies batch

        Returns:
        --------
        y_item : torch array (N*,)
            mean value of leaf in selected tree (or proportion value for leaf
            for a class)
        Gamma_item : torch array (N*, K)
            Gamma matrix for leaves of selected tree.
        eta_item : torch array (N*, K)
            eta matrix for leaves of selected tree
        weights_item : torch array (N*, )
            weights associated with each leaf in selected tree (the number of
            observations thatfall in that leaf)
        softmax_structure_item : torch array (N*, K * (len(1d_dict) + len(2d_dict)))
            leaf based attributes associated with inputed 1d and 2d dictionary
            arrays.
        """
        y_item = self.y[self.t_vec == idx]

        Gamma_item = self.Gamma[self.t_vec == idx,:]
        eta_item = self.eta[self.t_vec == idx,:]
        weights_item = self.weights[self.t_vec == idx]
        softmax_structure_item = self.softmax_structure[self.t_vec == idx,:]
        #softmax_structure_item = self.softmax_structure[self.t_vec == idx,:,:]

        return y_item, Gamma_item, eta_item, weights_item, \
            softmax_structure_item


    def __create_softmax_structure(self, one_d_dict, two_d_dict):
        """
        inner function to process the one_d_dict and two_d_dict into
        desirable array

        Arguments:
        ----------
        one_d_dict : dict of arrays (TN*,) (can be None)
        two_di_dict : dict of arrays (TN*, K) (can be None)

        Returns:
        --------
        X_nn_pytorch : torch array (N*, K * (len(1d_dict) + len(2d_dict)))
            leaf based attributes associated with inputed 1d and 2d dictionary
            arrays.
        """
        num_vars = len(one_d_dict) + len(two_d_dict)

        X_nn = np.zeros((self.n_obs, self.lamb_dim * num_vars))
        # X_nn = np.zeros((self.n_obs, self.lamb_dim, num_vars))


        var_num = 0
        if len(one_d_dict) > 0:
            for key, item in one_d_dict.items():
                assert item.shape == (self.n_obs,),  \
                    "Error in creating softmax structure, " +\
                    "incorrect shape of a 1d vector, (%s)"%key
                X_nn[:,self.lamb_dim * var_num :( self.lamb_dim *(var_num + 1))] = \
                   np.tile(item.reshape((-1,1)), (1,self.lamb_dim))

                # X_nn[:,:,var_num] = \
                #      np.tile(item.reshape((-1,1)), (1,self.lamb_dim))

                var_num += 1



        if len(two_d_dict) > 0:
            for key, item in two_d_dict.items():
                assert item.shape == (self.n_obs, self.lamb_dim), \
                    "Error in creating softmax structure, " +\
                    "incorrect shape of a 2d matrix, (%s)"%key
                X_nn[:,self.lamb_dim*var_num:(self.lamb_dim*(var_num+1))] = item

                # X_nn[:,:,var_num] = \
                #      np.tile(item.reshape((-1,1)), (1,self.lamb_dim))

                var_num += 1

        X_nn_pytorch = torch.from_numpy(X_nn)
        X_nn_pytorch = X_nn_pytorch.type(torch.float32)

        return X_nn_pytorch


class SoftmaxTreeFit(torch.nn.Module):
    def __init__(self, num_vars, lamb_dim, init = [5, "r"]):
        """
        Pytorch model for smooth random forest optimization. Specifically for
        the optimization that attempts to provide individual lambdas per each
        leaf as a function of the leaf's attributes. The __init__ function
        generates the graphical model

        Arguments:
        ----------
        num_vars : int
            should be equal to len(one_d_dict.keys()) + len(two_d_dict.keys())
        lamb_dim : int
            dimension of lambda
        init : scalar / string
            telling type of initialization for linear models. If "r" then
            all coefficients are randomly initialized under standard pytorch
            protocols. Else, it's expected to be a positive scalar which is
            the weight of the bias terms to make the lambdas very similar to
            the standard random forest weights (see details). Default


        Attributes:
        -----------
        forward : forward function
            function to predicted values

        Details:
        --------
        When init is a positive scalar the weighting of each linear model
        - 1 set relative to each different lambda dimension
        (total count = lamb_dim) is 0 for all values except the bias term of
        the first linear model (related to the first lambda value).
        """
        if type(init) is list:
            init = init[0]
        if init != "r" and init <= 0:
            ValueError("init must either be 'r' or a positive scalar")

        super(SoftmaxTreeFit, self).__init__()

        self.num_vars = num_vars
        self.lamb_dim = lamb_dim

        # set of list of linear models
        self.linear_list = torch.nn.ModuleList([torch.nn.Linear(self.num_vars,1) for _ in range(self.lamb_dim)])
        if init != "r":
            for l_idx in range(self.lamb_dim):
                self.linear_list[l_idx].weight = \
                    torch.nn.Parameter(
                        torch.zeros(self.num_vars).resize_(
                                        self.linear_list[l_idx].weight.shape)
                        )
                if l_idx == 0:
                    self.linear_list[l_idx].bias = \
                        torch.nn.Parameter(torch.ones(1)*init)
                else:
                    self.linear_list[l_idx].bias = \
                        torch.nn.Parameter(torch.zeros(1))

        # softmax structure
        self.soft_max = torch.nn.Softmax(dim = 2)


    def forward(self, x):
        """
        function to create predicted values related to underlying graphical
        structure

        Arguments:
        ----------
        x : tuple
            data tuple as defined by output of ForestDataset.__getitem__()

        Returns:
        --------
        yhat_item : torch array (N*,)
            array of predicted values for each leaf of the x input (either
            probability or regression values)
        weights_item : torch array (N*,)
            array of weights associated with each leaf of x input
        """
        y_item, Gamma_item, eta_item, weights_item, \
            softmax_structure_item = x

        x_linear = []

        for lamb_idx in range(self.lamb_dim):
            c_idx = (lamb_idx + np.array([self.lamb_dim*x for x in range(self.num_vars)]))
            X_nn_pytorch_inner = softmax_structure_item[:,:,c_idx] # weird shape
            x_linear.append(self.linear_list[lamb_idx](X_nn_pytorch_inner))

        linear_layer = torch.stack(x_linear,dim = 2)
        lamb = self.soft_max(linear_layer).type(torch.float64)

        # standard calculations:
        Gamma_fill = Gamma_item.resize_(lamb.shape) * lamb
        eta_fill = eta_item.resize_(lamb.shape)  * lamb
        #eta_fill[eta_fill == 0] = 1 # to avoid divide by 0.

        yhat_item = (Gamma_fill).sum(dim = 2) / (eta_fill).sum(dim = 2)

        return yhat_item, weights_item, lamb



def weighted_l2(y_item, y_pred, weights):
    """
    calculate weighted l2 loss for pytorch input

    Arguments:
    ----------
    y_item : torch array (N*,)
        array of true leave values (either probability or regression values)
    y_pred : torch array (N*,)
        array of predicted values for each leaf of the x input (either
        probability or regression values)
    weights_item : torch array (N*,)
        array of weights associated with each leaf of x input

    Returns:
    --------
    loss : torch scalar
        l2 loss
    """
    loss = torch.sum( weights * (y_item - y_pred)**2) / torch.sum( weights )
    return loss

def weighted_l2_np(y, y_pred, weights):
    """
    calculate weighted l2 loss for numpy input

    Arguments:
    ----------
    y : array (N*,)
        array of true leave values (either probability or regression values)
    y_pred : array (N*,)
        array of predicted values for each leaf of the x input (either
        probability or regression values)
    weights_item : array (N*,)
        array of weights associated with each leaf of x input

    Returns:
    --------
    loss : scalar
        l2 loss
    """
    loss = np.sum(weights * (y - y_pred)**2)/np.sum(weights)
    return loss

def l2_np(y, y_pred):
    """
    calculate un-weighted l2 loss for numpy input

    Arguments:
    ----------
    y : array (N*,)
        array of true leave values (either probability or regression values)
    y_pred : array (N*,)
        array of predicted values for each leaf of the x input (either
        probability or regression values)

    Returns:
    --------
    loss : scalar
        l2 loss
    """
    loss = np.mean((y - y_pred)**2)
    return loss

def acc_np(y, y_pred):
    """
    calculate un-weighted misclassication accuracy for numpy input

    Arguments:
    ----------
    y : array (N*,)
        array of class value
    y_pred : array (N*,)
        array of predicted class values

    Returns:
    --------
    acc : scalar
        misclassification accuracy
    """
    acc = np.mean(y == y_pred)
    return acc






def update_rf(random_forest, pytorch_model,
              X_trained, y_trained,
              parents_all=False,
              distance_style=["standard","max", "min"],
              verbose=True):
    """
    update a sklearn random forest model with tuned pytorch model

    Arguments:
    ----------
    random_forest : sklearn forest
            (sklearn.ensemble.forest.RandomForestRegressor or
            sklearn.ensemble.forest.RandomForestClassifier)
        grown forest
    pytorch_model: SoftmaxTreeFit class object (pytorch model object)
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
    parents_all : bool
        logic to instead include all observations with parent of distance k
        away
    distance_style : string
        style of inner-tree distance to use, see *details* in the
        create_distance_mat_leaves doc-string.
    verbose : bool
        logic to show tree analysis process
    Returns:
    --------
    inner_rf : updated smooth random forest
    """

    if type(random_forest) is sklearn.ensemble.RandomForestClassifier:
        rf_type = "class"
    elif type(random_forest) is sklearn.ensemble.RandomForestRegressor:
        rf_type = "reg"
    else:
        ValueError("random_forest needs to be either a " +\
                   "sklearn.ensemble.RandomForestClassifier " +\
                   "or a sklearn.ensemble.RandomForestRegressor")

    y_all, Gamma, eta, weight_all, t_idx_vec, \
        one_d_dict, two_d_dict, lamb_dim, num_classes = \
        pytorch_numpy_prep(random_forest, X_trained, y_trained,
                       train_only=True,
                       distance_style=distance_style,
                       parents_all=parents_all,
                       verbose=verbose)

    pytorch_forest_train = ForestDataset(y_all, Gamma, eta, weight_all,
                                         t_idx_vec, one_d_dict, two_d_dict,
                                         lamb_dim)

    data_loader_inner = DataLoader(dataset = pytorch_forest_train,
                            sampler = SequentialSampler(pytorch_forest_train))
    # ^ make sure to interate through all trees sequentially

    inner_rf = copy.deepcopy(random_forest)
    forest = inner_rf.estimators_

    if rf_type == "class":
        num_classes = inner_rf.n_classes_

    for i, x in enumerate(data_loader_inner): # not sure this works...
        #y_tree, _, _, _, _ = x # shouldn't this be y_tree, _ = pytorch_model(x)?

        y_tree, _, lamb = pytorch_model(x)

        y_tree = y_tree.detach().numpy()

        t = forest[i]
        tree = t.tree_
        num_leaf = np.sum(tree.children_left == -1)

        if rf_type == "reg":
            tree.value[tree.children_left == -1,:,:] = \
                y_tree.reshape((-1,1,1))
        else:
            tree.value[tree.children_left == -1,:,:] = \
                y_tree.reshape((-1,1,num_classes), order = "F")

    return inner_rf



def update_rf_middle(random_forest, pytorch_forest, pytorch_model):
    """
    update a sklearn random forest model with tuned pytorch model

    Arguments:
    ----------
    random_forest : sklearn forest
            (sklearn.ensemble.forest.RandomForestRegressor or
            sklearn.ensemble.forest.RandomForestClassifier)
        grown forest
    pytorch_forest: forest class object (pytorch Dataset object)
    pytorch_model: SoftmaxTreeFit class object (pytorch model object)

    Returns:
    --------
    inner_rf : updated smooth random forest
    """

    if type(random_forest) is sklearn.ensemble.RandomForestClassifier:
        rf_type = "class"
    elif type(random_forest) is sklearn.ensemble.RandomForestRegressor:
        rf_type = "reg"
    else:
        ValueError("random_forest needs to be either a " +\
                   "sklearn.ensemble.RandomForestClassifier " +\
                   "or a sklearn.ensemble.RandomForestRegressor")


    data_loader_inner = DataLoader(dataset = pytorch_forest,
                                   sampler = SequentialSampler(pytorch_forest)
                                   # ^ not sure this is needed...
                                   )
    inner_rf = copy.deepcopy(random_forest)
    forest = inner_rf.estimators_

    if rf_type == "class":
        num_classes = inner_rf.n_classes_

    for i, x in enumerate(data_loader_inner): # not sure this works...
        #y_tree, _, _, _, _ = x # shouldn't this be y_tree, _ = pytorch_model(x)?
        y_tree, _ = pytorch_model(x) # this changes x...

        y_tree = y_tree.detach().numpy()

        t = forest[i]
        tree = t.tree_
        num_leaf = np.sum(tree.children_left == -1)

        if rf_type == "reg":
            tree.value[tree.children_left == -1,:,:] = \
                y_tree.reshape((-1,1,1))
        else:
            tree.value[tree.children_left == -1,:,:] = \
                y_tree.reshape((-1,1,num_classes), order = "F")

    return inner_rf


def update_rf_old(random_forest, pytorch_forest, pytorch_model):
    """
    update a sklearn random forest model with tuned pytorch model

    Arguments:
    ----------
    random_forest : sklearn forest
            (sklearn.ensemble.forest.RandomForestRegressor or
            sklearn.ensemble.forest.RandomForestClassifier)
        grown forest
    pytorch_forest: forest class object (pytorch Dataset object)
    pytorch_model: SoftmaxTreeFit class object (pytorch model object)

    Returns:
    --------
    inner_rf : updated smooth random forest
    """

    if type(random_forest) is sklearn.ensemble.RandomForestClassifier:
        rf_type = "class"
    elif type(random_forest) is sklearn.ensemble.RandomForestRegressor:
        rf_type = "reg"
    else:
        ValueError("random_forest needs to be either a " +\
                   "sklearn.ensemble.RandomForestClassifier " +\
                   "or a sklearn.ensemble.RandomForestRegressor")


    data_loader_inner = DataLoader(dataset = pytorch_forest,
                                   sampler = SequentialSampler(pytorch_forest)
                                   # ^ not sure this is needed...
                                   )
    inner_rf = copy.deepcopy(random_forest)
    forest = inner_rf.estimators_

    if rf_type == "class":
        num_classes = inner_rf.n_classes_

    for i, x in enumerate(data_loader_inner):
        #y_tree, _, _, _, _ = x # shouldn't this be y_tree, _ = pytorch_model(x)?
        # y_item, Gamma_item, eta_item, weights_item, \
        #     softmax_structure_item = x
        y_tree, weights = pytorch_model(x)
        y_tree = y_tree.detach().numpy()

        t = forest[i]
        tree = t.tree_
        num_leaf = np.sum(tree.children_left == -1)

        if rf_type == "reg":
            tree.value[tree.children_left == -1,:,:] = \
                y_tree.reshape((-1,1,1))
        else:
            tree.value[tree.children_left == -1,:,:] = \
                y_tree.reshape((-1,1,num_classes), order = "F")

    return inner_rf


def smooth_pytorch(random_forest, X_trained, y_trained,
               X_tune=None, y_tune=None, resample_tune=False,
               sgd_max_num=1000, all_trees=False, parents_all=False,
               distance_style=["standard","max", "min"],
               which_dicts=["one_d_dict", "two_d_dict"],
               x_dicts=[],
               init = 10,
               verbose=True):
    """
    Fits smooth random forest optimization. Specifically for the optimization
    that attempts to provide individual lambdas per each leaf as a function of
    the leaf's attributes.

    Arguments:
    ----------
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
    resample_tune: bool
        logic to tune / optimize with another bootstrap same from X
    sgd_max_num : int
        number of steps to take for the stocastic gradient optimization
    all_trees : bool
        logic to use all trees (and therefore do full gradient descent)
    parents_all : bool
        logic to instead include all observations with parent of distance k
        away
    distance_style : string
        style of inner-tree distance to use, see *details* in the
        create_distance_mat_leaves doc-string.
    which_dicts : list of strings
        which dictionaries that define the features of the leaves we'd like to
        look at should be used, see pytorch_numpy_prep function for more
        details
    x_dicts : list of strings
        which dictionaries should be appended based on centering of nodes
        (same options as "which_dicts")
    verbose : bool
        logic to show pytorch data creation process and iteration of steps of
        Adam SGD (note that iteraions will be requested number // number trees)


    Returns:
    --------
    smooth_rf_pytorch : sklearn forest
        smoothed sklearn forest (same class as random_forest inputed)
    loss_all : list
        list of all full losses along the Adam SGD optimization path
    loss_min : scalar
        minimum full loss obtained when SGD was preformed
    params_min : torch model parameters
        SoftmaxTreeFit model parameters when minimum full loss was obtained
    best_model: SoftmaxTreeFit class torch model
        best SoftmaxTree model (that assoicated with the minimum full loss)
    (torch_model, forest_dataset, dataloader) : tuple
        torch_model is the SoftmaxTreeFit class torch model used for the
        analysis,
        forest_dataset contains the ForestDataset class torch dataset created from
        the data
        dataloader is the torch dataloader class object for the forest_dataset
    """
    y_all, Gamma, eta, weights_all, t_idx_vec, \
        one_d_dict, two_d_dict, lamb_dim, num_classes \
         = pytorch_numpy_prep(random_forest,
                              X_trained = X_trained, y_trained = y_trained,
                              distance_style=distance_style,
                              parents_all=parents_all,
                              verbose=verbose,
                              train_only=True)

    if "one_d_dict" not in which_dicts:
        one_d_dict = dict()
    if "two_d_dict" not in which_dicts:
        two_d_dict = dict()

    x_input_one_d_dict = one_d_dict
    x_input_two_d_dict = two_d_dict

    if "one_d_dict" not in x_dicts:
        x_input_one_d_dict = None
    if "two_d_dict" not in x_dicts:
        x_input_two_d_dict = None

    if x_input_one_d_dict is not None or x_input_two_d_dict is not None:
        od, td = node_spatial_structure_update(random_forest, X_trained,
                                               one_d_dict=x_input_one_d_dict,
                                               two_d_dict=x_input_two_d_dict)

    if od is not None:
        one_d_dict = od
    if td is not None:
        two_d_dict = td

    num_vars = len(one_d_dict) + len(two_d_dict)
    num_trees = random_forest.n_estimators

    forest_dataset = ForestDataset(y_all, Gamma, eta, weights_all,
                            t_idx_vec, one_d_dict, two_d_dict, lamb_dim)
    dataloader = DataLoader(dataset = forest_dataset,
                            sampler = RandomSampler(forest_dataset,
                                                    replacement=True)) # doesn't have to go through all trees in 1 iteration
    if all_trees:
        t_idx_vec = np.zeros(t_idx_vec.shape)
        num_trees = 1

    # for full loss function
    t_idx_all = np.zeros(t_idx_vec.shape)
    dataloader_all = DataLoader(dataset = ForestDataset(y_all, Gamma, eta, weights_all,
                            t_idx_all, one_d_dict, two_d_dict, lamb_dim))


    torch_model = SoftmaxTreeFit(num_vars=num_vars, lamb_dim=lamb_dim,
                                 init=init)
    criterion = weighted_l2
    optimizer = torch.optim.Adam(torch_model.parameters())

    # actual Adam SGD
    loss_all = []
    loss_min = np.inf
    params_min = []
    verbose = True

    first_iter = range(sgd_max_num // num_trees)

    if verbose:
        bar = progressbar.ProgressBar()
        first_iter = bar(np.arange(sgd_max_num // num_trees))


    for epoch in first_iter: # to get the actual number of interations close to max_iter
        for i, x in enumerate(dataloader):
            y_item, Gamma_item, eta_item, weights_item, \
                    softmax_structure_item = x
            y_pred, weights, _ = torch_model(x)

            loss = criterion(y_item, y_pred, weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                x2 = next(iter(dataloader_all))
                y_all, _, _, _, _ = x2
                y_all_pred, weights_all,_ = torch_model(x2)

                current_loss_all = criterion(y_all, y_all_pred, weights_all).item()
                loss_all += [current_loss_all]

                if loss_min >= current_loss_all:
                    loss_min = current_loss_all
                    params_min = list(torch_model.parameters()).copy()

                    best_model = copy.deepcopy(torch_model)

    # updating sklearn random forest
    smooth_rf_pytorch = update_rf(random_forest,
                         pytorch_model=best_model,
                         X_trained=X_trained,
                         y_trained=y_trained,
                         parents_all=parents_all,
                         distance_style=distance_style,
                         verbose=False)

    return smooth_rf_pytorch, loss_all, loss_min, params_min, best_model, \
        (torch_model, forest_dataset, dataloader)
