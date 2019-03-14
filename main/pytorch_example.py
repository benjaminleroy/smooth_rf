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
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torch.nn

import smooth_rf
import sklearn.datasets
import sklearn.metrics


# def pytorch_numpy_prep(random_forest, X_trained, y_trained,
#                        X_tune=None, y_tune=None,resample_tune=False,
#                        distance_style=["standard","max", "min"],
#                        parents_all=True,verbose=True):

#     if type(random_forest) is sklearn.ensemble.RandomForestClassifier:
#         rf_type = "class"
#     elif type(random_forest) is sklearn.ensemble.RandomForestRegressor:
#         rf_type = "reg"
#     else:
#         ValueError("random_forest needs to be either a " +\
#                    "sklearn.ensemble.RandomForestClassifier " +\
#                    "or a sklearn.ensemble.RandomForestRegressor")

#     if (X_tune is None or y_tune is None) and not resample_tune:
#         oob_logic = True
#     else:
#         oob_logic = False


#     if (oob_logic or resample_tune) and \
#         (X_trained is None or y_trained is None):
#         raise TypeError("X_trained and y_trained need to be inserted for "+\
#                         "provided input of X_tune/y_tune and resample_tune "+\
#                         "parameters.")

#     if oob_logic or resample_tune:
#         n_obs_trained = X_trained.shape[0]

#     _, max_depth = smooth_rf.calc_depth_for_forest(random_forest,verbose=False)
#     max_depth = np.int(max_depth)

#     forest = random_forest.estimators_

#     Gamma, eta, t_idx_vec, leaf_n, leaf_d, leaf_i, full_d, full_i = \
#         smooth_rf.create_Gamma_eta_forest_more(random_forest,
#                                      verbose=verbose,
#                                      parents_all=parents_all,
#                                      dist_mat_style=distance_style)

#     first_iter = forest
#     if verbose:
#         bar = progressbar.ProgressBar()
#         first_iter = bar(first_iter)

#     if rf_type == "reg":
#         y_all = np.zeros((0,))
#     else:
#         y_all = np.zeros((0,Gamma.shape[0]))


#     weight_all = np.zeros((0,))


#     for t in first_iter:
#         tree = t.tree_

#         num_leaf = np.sum(tree.children_left == -1)

#         # to grab the OOB:
#         # _generate_sample_indices
#         # from https://github.com/scikit-learn/scikit-learn/blob/bac89c253b35a8f1a3827389fbee0f5bebcbc985/sklearn/ensemble/forest.py#L78
#         # just need to grab the tree's random state (t.random_state)


#         # observed information
#         if oob_logic:
#             random_state = t.random_state
#             oob_indices = \
#                 sklearn.ensemble.forest._generate_unsampled_indices(
#                                                                  random_state,
#                                                                  n_obs_trained)
#             X_tune = X_trained[oob_indices,:]
#             y_tune = y_trained[oob_indices]

#             if rf_type == "class":
#                 y_tune = np.array(pd.get_dummies(y_tune))

#         if resample_tune:
#             resample_indices = \
#                 sklearn.ensemble.forest._generate_sample_indices(None,
#                                                                  n_obs_trained)
#             X_tune = X_trained[resample_indices,:]
#             y_tune = y_trained[resample_indices]

#             if rf_type == "class":
#                 y_tune = np.array(pd.get_dummies(y_tune))

#         # create y_leaf and weights for observed
#         obs_V = t.decision_path(X_tune)
#         obs_V_leaf = obs_V[:,tree.children_left == -1]
#         obs_weight = obs_V_leaf.sum(axis = 0).ravel() # by column (leaf)

#         #---
#         # for clean division without dividing by 0
#         obs_weight_div = obs_weight.copy()
#         obs_weight_div[obs_weight_div == 0] = 1

#         # obs_y_leaf is either \hat{y}_obs or \hat{p}_obs
#         if rf_type == "reg":
#             obs_y_leaf = (obs_V_leaf.T @ y_tune) / obs_weight_div
#             y_all = np.concatenate((y_all, np.array(obs_y_leaf).ravel()))
#         else:
#             obs_y_leaf = (obs_V_leaf.T @ y_tune) / obs_weight_div.T
#             y_all = np.concatenate((y_all, np.array(obs_y_leaf)))


#         weight_all = np.concatenate((weight_all,
#                                      np.array(obs_weight).ravel()))


#     if rf_type == "class":
#         Gamma_shape = Gamma.shape
#         num_classes = Gamma.shape[0]
#         Gamma = Gamma.reshape((Gamma.shape[0]*Gamma.shape[1],
#                                      Gamma.shape[2]))

#         eta = np.tile(eta, (num_classes,1))
#         full_d = np.tile(full_d, (num_classes,1))
#         full_i = np.tile(full_i, (num_classes,1))

#         y_all = y_all.T.reshape((-1,))
#         weight_all = np.tile(weight_all, num_classes)
#         t_idx_vec = np.tile(t_idx_vec, num_classes)
#         leaf_n = np.tile(leaf_n, num_classes)
#         leaf_d = np.tile(leaf_d, num_classes)
#         leaf_i = np.tile(leaf_i, num_classes)

#     else:
#         num_classes = 1

#     y_all, weight_all
#     one_d_dict = {"leaf_n":leaf_n, "leaf_d":leaf_d, "leaf_i":leaf_i}
#     two_d_dict = {"full_d":full_d,"full_i":full_i}
#     n_obs = Gamma.shape[0]
#     lamb_dim = Gamma.shape[-1]

#     return y_all, Gamma, eta, weight_all, t_idx_vec, \
#         one_d_dict, two_d_dict, lamb_dim, num_classes


# class forest(Dataset):
#     def __init__(self, y, Gamma, eta, weights, t_vec,
#                  one_d_dict, two_d_dict, lamb_dim):
#         self.y = torch.from_numpy(y)
#         self.Gamma = torch.from_numpy(Gamma)
#         self.eta = torch.from_numpy(eta)
#         self.weights = torch.from_numpy(weights)
#         self.num_trees = len(dict(Counter(t_vec)).keys())
#         self.t_vec = torch.from_numpy(t_vec)
#         self.n_obs = eta.shape[0]
#         self.lamb_dim = lamb_dim
#         self.softmax_structure = self.__create_softmax_structure(one_d_dict,
#                                                                  two_d_dict)
#     #init data here

#     def __len__(self):
#         return   self.num_trees

#     def __getitem__(self, idx):
#         y_item = self.y[self.t_vec == idx]

#         Gamma_item = self.Gamma[self.t_vec == idx,:]
#         eta_item = self.eta[self.t_vec == idx,:]
#         weights_item = self.weights[self.t_vec == idx]
#         softmax_structure_item = self.softmax_structure[self.t_vec == idx,:]

#         return y_item, Gamma_item, eta_item, weights_item, \
#             softmax_structure_item


#     def __create_softmax_structure(self, one_d_dict, two_d_dict):
#         num_vars = len(one_d_dict) + len(two_d_dict)

#         X_nn = np.zeros((self.n_obs, self.lamb_dim * num_vars))

#         var_num = 0
#         if len(one_d_dict) > 0:
#             for key, item in one_d_dict.items():
#                 assert item.shape == (self.n_obs,),  \
#                     "Error in creating softmax structure, " +\
#                     "incorrect shape of a 1d vector, (%s)"%key
#                 X_nn[:,self.lamb_dim * var_num :( self.lamb_dim *(var_num + 1))] = \
#                     np.tile(item.reshape((-1,1)), (1,self.lamb_dim))
#                 var_num += 1

#         if len(two_d_dict) > 0:
#             for key, item in two_d_dict.items():
#                 assert item.shape == (self.n_obs, self.lamb_dim), \
#                     "Error in creating softmax structure, " +\
#                     "incorrect shape of a 2d matrix, (%s)"%key
#                 X_nn[:,self.lamb_dim*var_num:(self.lamb_dim*(var_num+1))] = item

#                 var_num += 1

#         X_nn_pytorch = torch.from_numpy(X_nn)
#         X_nn_pytorch = X_nn_pytorch.type(torch.float32)

#         return X_nn_pytorch



# class SoftmaxTreeFit(torch.nn.Module):
#     def __init__(self, num_vars, lamb_dim):
#         """
#         Arguments:
#         ----------
#         num_vars : int
#             should be equal to len(one_d_dict.keys()) + len(two_d_dict.keys())
#         lamb_dim : int
#             dimension of lambda

#         """
#         super(SoftmaxTreeFit, self).__init__()

#         self.num_vars = num_vars
#         self.lamb_dim = lamb_dim

#         # set of list of linear
#         self.linear_list = torch.nn.ModuleList([torch.nn.Linear(self.num_vars,1) for _ in range(self.lamb_dim)])

#         # softmax structure
#         self.soft_max = torch.nn.Softmax(dim = 1)


#     def forward(self, x):
#         y_item, Gamma_item, eta_item, weights_item, \
#             softmax_structure_item = x



#         all_zeros_idx = np.arange(softmax_structure_item.shape[-1],
#                                   dtype = np.int)

#         beta_dict = []
#         x_linear = []

#         for lamb_idx in range(self.lamb_dim):
#             c_idx = (lamb_idx + np.array([self.lamb_dim*x for x in range(self.num_vars)]))
#             X_nn_pytorch_inner = softmax_structure_item[:,:,c_idx] # weird shape
#             x_linear.append(self.linear_list[lamb_idx](X_nn_pytorch_inner))

#         linear_layer = torch.stack(x_linear,dim = 1)
#         lamb = self.soft_max(linear_layer).type(torch.float64)

#         # standard calculations:

#         Gamma_fill = Gamma_item.resize_(lamb.shape) * lamb
#         eta_fill = eta_item.resize_(lamb.shape)  * lamb
#         eta_fill[eta_fill == 0] = 1 # to avoid divide by 0.
#         yhat_item = Gamma_fill.sum(dim = 1) /\
#                             eta_fill.sum(dim = 1)
#         #residuals = y_item - Gamma_fill.sum(dim = 1) /\
#         #                    eta_fill.sum(dim = 1)

#         #loss = torch.sum( (residuals**2) * weights_item)

#         return yhat_item, weights_item


# def weighted_l2(y_item, y_pred, weights):
#     """
#     weighted l2 loss, pytorch
#     """
#     loss = torch.sum( weights * (y_item - y_pred)**2)
#     return loss

# def weighted_l2_np(y, y_pred, weights):
#     """
#     weighted l2 loss, numpy
#     """
#     loss = np.sum(weights * (y - y_pred)**2)
#     return loss

# def l2_np(y, y_pred):
#     """
#     weighted l2 loss, numpy
#     """
#     loss = np.sum((y - y_pred)**2)
#     return loss

# def acc_np(y, y_pred):
#     """
#     weighted l2 loss, numpy
#     """
#     acc = np.mean(y == y_pred)
#     return acc



# def update_rf(random_forest, pytorch_forest, pytorch_model):

#     if type(random_forest) is sklearn.ensemble.RandomForestClassifier:
#         rf_type = "class"
#     elif type(random_forest) is sklearn.ensemble.RandomForestRegressor:
#         rf_type = "reg"
#     else:
#         ValueError("random_forest needs to be either a " +\
#                    "sklearn.ensemble.RandomForestClassifier " +\
#                    "or a sklearn.ensemble.RandomForestRegressor")


#     data_loader_inner = DataLoader(dataset = pytorch_forest,
#                                    sampler = SequentialSampler(pytorch_forest))
#     inner_rf = copy.deepcopy(random_forest)
#     forest = inner_rf.estimators_

#     if rf_type == "class":
#         num_classes = inner_rf.n_classes_

#     for i, x in enumerate(data_loader_inner):
#         y_tree, _, _, _, _ = x
#         y_tree= y_tree.numpy()

#         t = forest[i]
#         tree = t.tree_
#         num_leaf = np.sum(tree.children_left == -1)

#         if rf_type == "reg":
#             tree.value[tree.children_left == -1,:,:] = \
#                 y_tree.reshape((-1,1,1))
#         else:
#             tree.value[tree.children_left == -1,:,:] = \
#                 y_tree.reshape((-1,1,num_classes), order = "F")

#     return inner_rf

#### example code:

base_error = []
smooth_error = []
base_acc = []
smooth_acc = []

for sim in np.arange(2):
    print("sim", sim)
    data, y = sklearn.datasets.make_moons(n_samples=350, noise=.3)

    data_test, y_test = sklearn.datasets.make_moons(10000, noise=.3)

    model_type = sklearn.ensemble.RandomForestClassifier

    model = model_type(n_estimators=10)
    model_fit = model.fit(data, y)
    random_forest = model_fit

    # y_all, Gamma, eta, weights_all, t_idx_vec, \
    #     one_d_dict, two_d_dict, lamb_dim, num_classes \
    #      = pytorch_numpy_prep(random_forest,
    #                           X_trained = data, y_trained = y,
    #                           X_tune=None, y_tune=None,resample_tune=False,
    #                           distance_style="standard", parents_all=True,
    #                           verbose=True)
    # #two_d_dict = dict()

    # num_vars = len(one_d_dict) + len(two_d_dict)
    # num_trees = model.n_estimators


    # forest_dataset = forest(y_all, Gamma, eta, weights_all,
    #                         t_idx_vec, one_d_dict, two_d_dict, lamb_dim)
    # dataloader = DataLoader(dataset = forest_dataset,
    #                         sampler = RandomSampler(forest_dataset,
    #                                                 replacement=True)) # doesn't have to go through all trees in 1 iteration

    # t_idx_all = np.zeros(t_idx_vec.shape)
    # dataloader_all = DataLoader(dataset = forest(y_all, Gamma, eta, weights_all,
    #                         t_idx_all, one_d_dict, two_d_dict, lamb_dim))



    # torch_model = SoftmaxTreeFit(num_vars=num_vars, lamb_dim=lamb_dim)
    # criterion = weighted_l2
    # optimizer = torch.optim.Adam(torch_model.parameters())



    max_iter = 10000
    # loss_all = []
    # loss_min = np.inf
    # params_min = []
    # verbose = True

    # first_iter = range(max_iter // num_trees)

    # if verbose:
    #     bar = progressbar.ProgressBar()
    #     first_iter = bar(np.arange(max_iter // num_trees))


    # for epoch in first_iter: # to get the actual number of interations close to max_iter
    #     for i, x in enumerate(dataloader):
    #         y_item, Gamma_item, eta_item, weights_item, \
    #                 softmax_structure_item = x
    #         y_pred, weights = torch_model(x)

    #         loss = criterion(y_item, y_pred, weights)

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         with torch.no_grad():
    #             x2 = next(iter(dataloader_all))
    #             y_all, _, _, _, _ = x2
    #             y_all_pred, weights_all = torch_model(x2)

    #             current_loss_all = criterion(y_all, y_all_pred, weights_all).item()
    #             loss_all += [current_loss_all]

    #             if loss_min >= current_loss_all:
    #                 loss_min = current_loss_all
    #                 params_min = list(torch_model.parameters()).copy()

    #                 best_model = copy.deepcopy(torch_model)



    # smooth_rf_pytorch = update_rf(random_forest,
    #                      pytorch_forest = forest_dataset,
    #                      pytorch_model = best_model)

    # y_pred_test_base = random_forest.predict(data_test)
    # y_pred_test_smooth = smooth_rf_pytorch.predict(data_test)

    # functional representation

    smooth_rf_pytorch, loss_all, loss_min, params_min, best_model, \
        (torch_model, forest_dataset, dataloader) = \
        smooth_rf.smooth_pytorch(random_forest = random_forest,
               X_trained=data, y_trained=y,
               X_tune=None, y_tune=None,
               resample_tune=False,
               sgd_max_num=max_iter,
               all_trees=False,
               parents_all=True,
               distance_style="standard",
               which_dicts=["one_d_dict", "two_d_dict"],
               verbose=True)

    y_pred_test_base = random_forest.predict(data_test)
    y_pred_test_smooth = smooth_rf_pytorch.predict(data_test)

    base_error.append(smooth_rf.l2_np(y_pred_test_base, y_test))
    smooth_error.append(smooth_rf.l2_np(y_pred_test_smooth, y_test))
    base_acc.append(smooth_rf.acc_np(y_pred_test_base, y_test))
    smooth_acc.append(smooth_rf.acc_np(y_pred_test_smooth, y_test))
    print(base_acc[sim], smooth_acc[sim])

csv_out_pd = pd.DataFrame(data = {"base l2": base_error,
                          "smooth l2": smooth_error,
                          "base acc": base_acc,
                          "smooth_acc": smooth_acc})

#csv_out_pd.to_csv("../images/pytorch_moon_both10.csv")

#sklearn.metrics.confusion_matrix(y_pred_test_base,y_test)
#sklearn.metrics.confusion_matrix(y_pred_test_smooth,y_test)
