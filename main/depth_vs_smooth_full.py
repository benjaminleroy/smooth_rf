import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.ensemble
import sklearn
import progressbar
import sklearn.model_selection
from plotnine import *
import pdb
import sys

import smooth_rf
path = "../images/"

def depth_search(X_train, y_train, depth_range = np.arange(2,50,2),
                 n_estimators = 100,
                 type = "reg", verbose = True):
    """
    recommend larger number of trees for oob error to stabilize
    """
    if type=="reg":
        model = sklearn.ensemble.RandomForestRegressor
    elif type == "class":
        model = sklearn.ensemble.RandomForestClassifier
    else:
        NameError("type must be 1 of the 2 options")


    first_iter = depth_range
    if verbose:
        bar = progressbar.ProgressBar()
        first_iter = bar(first_iter)

    best_depth = np.inf
    if type == "reg":
        best_oob_score = np.inf
    else:
        best_oob_score = -np.inf

    for depth in first_iter:
        rf = model(n_estimators = n_estimators,
                   max_depth = depth,
                   oob_score = True)

        rf = rf.fit(X = np.array(X_train),
                    y = y_train)

        oob_score = rf.oob_score_

        if type == "reg":
            if oob_score < best_oob_score:
                best_oob_score = oob_score
                best_depth = depth
        else:
            if oob_score > best_oob_score:
                best_oob_score = oob_score
                best_depth = depth

    return best_depth, best_oob_score


def test_depth_search():
    # data creation
    X_train = np.concatenate(
        (np.random.normal(loc = (1,2), scale = .6, size = (100,2)),
        np.random.normal(loc = (-1.2, -.5), scale = .6, size = (100,2))),
    axis = 0)
    y_train = np.concatenate((np.zeros(100, dtype = np.int),
                             np.ones(100, dtype = np.int)))

    b_depth = depth_search(X_train, y_train)




############
# Analysis #
############


# input
# 1: data_set = ["microsoft", "online_news"]

data_set = sys.argv[1]
if data_set == "online_news":
    data = pd.read_csv(path +\
                       "data/OnlineNewsPopularity/OnlineNewsPopularity.csv")

    y_all = np.log10(data[" shares"])
    data_all = data.drop(columns = ["url"," timedelta"," shares"])
    data_all = np.array(data_all)
elif data_set == "microsoft":
    data_generator = smooth_rf.generate_data
else:
    NameError("data_set need sto be 1 of the 2 options")



sim_amount = 20
n_data = 650
n_large = 1000
n_estimators = 100
n_steps = 10000
scoring = sklearn.metrics.mean_squared_error


first_iter = np.arange(sim_amount)
if verbose:
    bar = progressbar.ProgressBar()
    first_iter = bar(first_iter)


sim_info = np.zeros((sim_amount, 4))

for sim_idx in first_iter:

    # data simulation
    if data_set == "online_news":
        data, data_test, y, y_test = \
                    sklearn.model_selection.train_test_split(data_all,
                                                             y_all,
                                                             test_size = .5)
        data_tune = None
        y_tune = None
    else:
        all_dat = data_generator(large_n=n_data)
        data, y = all_dat[0], all_dat[1]
        y = y + 100

        all_dat_test = data_generator(large_n=n_large)
        data_test, y_test = all_dat_test[0], all_dat_test[1]
        y_test = y_test + 100


    # depth_selection:
    d_best, d_value = depth_search(data, y,
                                   depth_range = np.arange(2,50,2),
                                   n_estimators = n_estimators,
                                   type = "reg", verbose = False)

    d_rf = random.forest(n_estimators = n_estimators,
                         max_depth = d_best)
    d_rf_fit = d_rf.fit(data,y)
    yhat_test_depth = d_rf_fit.predict(data_test)



    # smoothed
    rf_base = random.forest(n_estimators = n_estimators)
    rf_fit = rf_base.fit(data,y)
    yhat_test_base = rf_fit.predict(data_test)


    _, rf_smooth, _, _ = smooth_rf.smooth(rf_fit,
                                          X_trained = data,
                                          y_trained = y,
                                          subgrad_max_num = n_steps)
    yhat_test_smooth = rf_smooth.predict(data_test)

    # depth_smoothed
    _, d_rf_smooth, _, _ = smooth_rf.smooth(d_rf_fit,
                                          X_trained = data,
                                          y_trained = y,
                                          subgrad_max_num = n_steps)
    yhat_test_smooth_depth = d_rf_smooth.predict(data_test)


    sim_info[sim_idx,:] = [scoring(y_test,yhat_test_depth),
                           scoring(y_test,yhat_test_base),
                           scoring(y_test,yhat_test_smooth),
                           scoring(y_test,yhat_test_smooth_depth)]


np.savetxt(fname = path + data_set + ".csv", X = sim_info)



