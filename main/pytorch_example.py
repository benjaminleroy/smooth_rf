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
import sklearn.datasets
import sklearn.metrics

base_error = []
smooth_error = []
smooth_error2 = []

for sim in np.arange(2):#np.arange(20):
    print("sim", sim)
    # data, y = sklearn.datasets.make_moons(n_samples=350, noise=.3)

    # data_test, y_test = sklearn.datasets.make_moons(10000, noise=.3)

    # model_type = sklearn.ensemble.RandomForestClassifier

    data, y = smooth_rf.generate_data(650)
    y = y + 100

    data_test, y_test = smooth_rf.generate_data(10000)
    y_test = y_test + 100

    model_type = sklearn.ensemble.RandomForestRegressor

    model = model_type(n_estimators=10)
    model_fit = model.fit(data, y)
    random_forest = model_fit

    max_iter = 10000


    smooth_rf_standard, _ , _, loss_all_standard = \
      smooth_rf.smooth(random_forest=random_forest,
               X_trained=data, y_trained=y,
               X_tune=None, y_tune=None,
               resample_tune=False,
               sgd_max_num=max_iter,
               all_trees=False,
               parents_all=True,
               distance_style="standard",
               verbose=True,
               adam={"alpha": .001, "beta_1": .9,
                     "beta_2": .999,"eps": 1e-8})

    smooth_rf_pytorch, loss_all, loss_min, params_min, best_model, \
        (torch_model, forest_dataset, dataloader) = \
        smooth_rf.smooth_pytorch(random_forest=random_forest,
               X_trained=data, y_trained=y,
               X_tune=None, y_tune=None,
               resample_tune=False,
               sgd_max_num=max_iter,
               all_trees=False,
               parents_all=True,
               distance_style="standard",
               which_dicts=[],
               x_dicts=["one_d_dict", "two_d_dict"],
               init=5,
               verbose=True)

    y_pred_test_base = random_forest.predict(data_test)
    y_pred_test_smooth = smooth_rf_pytorch.predict(data_test)
    y_pred_test_smooth2 = smooth_rf_standard.predict(data_test)

    base_error.append(smooth_rf.l2_np(y_pred_test_base, y_test))
    smooth_error.append(smooth_rf.l2_np(y_pred_test_smooth, y_test))
    smooth_error2.append(smooth_rf.l2_np(y_pred_test_smooth2, y_test))

    print(base_error, smooth_error, smooth_error2)
    #base_acc.append(smooth_rf.acc_np(y_pred_test_base, y_test))
    #smooth_acc.append(smooth_rf.acc_np(y_pred_test_smooth, y_test))
    #print(base_acc[sim], smooth_acc[sim])

# csv_out_pd = pd.DataFrame(data = {"base l2": base_error,
#                           "smooth l2": smooth_error,
#                           "base acc": base_acc,
#                           "smooth_acc": smooth_acc})

# csv_out_pd.to_csv("../images/pytorch_moon_both10.csv")


#sklearn.metrics.confusion_matrix(y_pred_test_base,y_test)
#sklearn.metrics.confusion_matrix(y_pred_test_smooth,y_test)
